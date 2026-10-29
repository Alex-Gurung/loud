"""E13's token statistics (plan v2): a streaming n-gram model of the training stream and a copy detector, from the
token ids alone. Nothing here enters the model; they only decide how many probe rows a token gets.

NGram: orders 1 to 4 with Witten-Bell interpolation, P(w | h) = (c(h, w) + T(h) P(w | h')) / (c(h) + T(h)), h' the
history without its oldest token, T(h) the number of distinct tokens seen after h, down to the unigram, which
interpolates with the uniform distribution the same way; where a history was never seen, its order passes the lower
order's probability on. Every level is a proper distribution over the vocabulary, with no free parameter. Streaming
and causal: a batch is scored under the counts of the batches before it, then inserted (score, then insert). Counts
are exact, in dicts keyed by the packed token ids (12 bits each), one per order; each batch costs a few dict operations
per unique n-gram.

Copy (copies): the token's preceding 3-gram occurred earlier in the same sequence followed by the same token.

What the 30M speedrun record (PR #52) already does with token statistics: it orders each Cut batch by difficulty, one
row from each of 8 strata of rows ranked by the mean add-one unigram surprisal of their targets. That balances whole
rows across batches; it gives every token of a batch the same probe budget. E13 gives tokens different budgets.

tiers / tiers_tool: E13's diagnostic, the share of the exact credit energy (summed over the nine sites, from H8's cut
graph) and of the tokens in the low tier, max(n-gram loss, model loss) <= theta, for each theta.
"""
import copy

import numpy as np
import torch

from .common import read_json
from .data import benchmark_split

ORDER = 4
BITS = 12          # bits per token id (vocabulary 4,096)
THETAS = (2.0, 3.0, 4.0)


def pack(rows, length, end):
    """The ids rows[:, end - length + 1 .. end] packed into one int64 per position `end` (oldest in the high bits)."""
    key = np.zeros(end.shape, dtype=np.int64)
    for j in range(length, 0, -1):
        key = (key << BITS) | rows[:, end - j + 1]
    return key


def keys(rows, order):
    """For every target of rows [B, L + 1] (the token at position t + 1, t = 0..L-1): the packed history (the order - 1
    ids before the target) and the packed (history, target) pair; and where the history fits in the row."""
    rows = np.asarray(rows, dtype=np.int64)
    L = rows.shape[1] - 1
    t = np.arange(L)
    n = order - 1
    if n > L:  # rows too short for this order's history anywhere
        none = np.zeros((rows.shape[0], L), dtype=np.int64)
        return none, none, np.zeros(none.shape, dtype=bool)
    end = np.maximum(t, n - 1) if n else t  # positions without a full history read a placeholder, masked below
    history = pack(rows, n, end) if n else np.zeros((rows.shape[0], L), dtype=np.int64)
    return history, (history << BITS) | rows[:, 1:], np.broadcast_to(t >= n - 1, (rows.shape[0], L))


def lookup(table, query):
    unique, inverse = np.unique(query, return_inverse=True)
    values = np.fromiter((table.get(k, 0) for k in unique.tolist()), dtype=np.float64, count=len(unique))
    return values[inverse].reshape(query.shape)


class NGram:
    def __init__(self, vocab, order=ORDER):
        self.vocab, self.order = vocab, order
        self.pairs = [{} for _ in range(order)]      # c(h, w), by order 1..order
        self.contexts = [{} for _ in range(order)]   # c(h)
        self.types = [{} for _ in range(order)]      # T(h)

    def probabilities(self, rows):
        """P(target | history) of every target of rows [B, L + 1] under the counts so far."""
        p = np.full((len(rows), np.asarray(rows).shape[1] - 1), 1.0 / self.vocab)
        for n in range(self.order):
            history, pair, valid = keys(rows, n + 1)
            c_h, t_h = lookup(self.contexts[n], history), lookup(self.types[n], history)
            seen = valid & (c_h > 0)
            mixed = (lookup(self.pairs[n], pair) + t_h * p) / np.maximum(c_h + t_h, 1)
            p = np.where(seen, mixed, p)
        return p

    def losses(self, rows):
        """Each target's n-gram loss in nats, -log P."""
        return -np.log(self.probabilities(rows))

    def insert(self, rows):
        for n in range(self.order):
            _, pair, valid = keys(rows, n + 1)
            unique, counts = np.unique(pair[valid], return_counts=True)
            pairs, contexts, types = self.pairs[n], self.contexts[n], self.types[n]
            for k, c in zip(unique.tolist(), counts.tolist()):
                h = k >> BITS
                old = pairs.get(k)
                if old is None:
                    pairs[k] = c
                    types[h] = types.get(h, 0) + 1
                else:
                    pairs[k] = old + c
                contexts[h] = contexts.get(h, 0) + c

    def distribution(self, history):
        """P(w | history) over the whole vocabulary for one history (a list of ids, oldest first): for checks."""
        rows = np.array([list(history) + [w] for w in range(self.vocab)], dtype=np.int64)
        return self.probabilities(rows)[:, -1]


def copies(rows):
    """Whether each target of rows [B, L + 1] is a copy: its preceding 3-gram occurred earlier in the same row followed
    by the same token (the same 4-gram ending earlier)."""
    _, pair, valid = keys(rows, 4)
    pair = np.where(valid, pair, -1 - np.arange(pair.shape[1]))  # positions without a 3-gram match nothing
    out = np.zeros(pair.shape, dtype=bool)
    for b in range(len(pair)):
        _, first, inverse = np.unique(pair[b], return_index=True, return_inverse=True)
        out[b] = first[inverse] < np.arange(pair.shape[1])
    return out & valid


def tiers(model, rows, seen, thetas=THETAS, chunk=4):
    """E13's diagnostic: with an n-gram table of the training rows `seen` so far, for each theta, the share of the
    tokens of `rows` in the low tier (max(n-gram loss, model loss) <= theta) and their share of the exact credit
    energy, sum over the nine sites of |g|^2 (H8's cut graph, float32), overall and per site; and the copy share."""
    from .h8 import cut_forward
    model = copy.deepcopy(model).float().requires_grad_(True)
    table = NGram(model.config.vocab_size)
    for b in range(0, len(seen), 64):
        table.insert(seen[b:b + 64].numpy())
    rows = rows.cpu()
    ngram = torch.from_numpy(table.losses(rows.numpy())).float()
    device = next(model.parameters()).device
    losses, energy = [], []
    for b in range(0, len(rows), chunk):
        part = rows[b:b + chunk].to(device)
        x, y = part[:, :-1].contiguous(), part[:, 1:].contiguous()
        token_losses, sites, *_ = cut_forward(model, x, y)
        credits = torch.autograd.grad(token_losses.sum(), sites)
        losses.append(token_losses.detach().float().cpu())
        energy.append(torch.stack([g.float().square().sum(-1) for g in credits], -1).cpu())  # [b, T, 9]
    losses, energy = torch.cat(losses), torch.cat(energy)
    hardness = torch.maximum(ngram, losses)
    summary = dict(tokens=int(hardness.numel()), copy_share=float(copies(rows.numpy()).mean()),
                   ngram_loss_mean=float(ngram.mean()), model_loss_mean=float(losses.mean()), per_site={})
    for theta in thetas:
        low = hardness <= theta
        name = f'theta{theta:g}'
        summary[name] = dict(energy=float(energy.sum(-1)[low].sum() / energy.sum()), tokens=float(low.float().mean()))
        summary['per_site'][name] = [float(energy[..., k][low].sum() / energy[..., k].sum())
                                     for k in range(energy.shape[-1])]
    return summary, dict(ngram_loss=ngram.half(), model_loss=losses.half())


def checkpoint_rows(checkpoint):
    """The training rows the run had fed into updates when it saved `checkpoint` (its config.json's step x batch, in
    the run's data order)."""
    config = read_json(checkpoint.parent / 'config.json')
    o, step = config['options'], config['checkpoints'][checkpoint.name]['step']
    return benchmark_split(o['sequences'], o['heldout']).train[:step * o['batch']]


def tiers_tool(model, rows, checkpoint):
    """tiers on a checkpoint of a training run: the diagnose task's entry point."""
    return tiers(model, rows, checkpoint_rows(checkpoint))


def low_tier(ngram_losses, model_losses, theta):
    """E13's rule: tokens whose loss under both the n-gram model and the network is at most theta nats get the low
    budget."""
    return torch.maximum(ngram_losses, model_losses.float()) <= theta

