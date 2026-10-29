"""H8: an autograd emulation of the record's Boost and Cut (plan v2, section 3). Development only: autograd here
stands in for the forward-only estimator and never enters a submission.

The fact it rests on: with the other tokens' keys and values held at their clean values and each token's own query
live (PR #40's own_query), every path from a site to a token's loss stays inside that token's row. So the gradient of
the fixed network with every stream-derived input to the keys, the values and the value-embedding gate detached
(queries, the value-embedding tables and every parameter path live) is what the chain measures, at every site at once.
That network is the cut graph (cut_forward). Its sites: S0, the raw embedding (wte's output), and S1..S8, block i's mid
stream (after attention, before the MLP) as S(i+1); the top stream, the last block's output before the final norm.

Modes of the Cut phase (option mode):
  cut        one backward of the cut graph. Block i's MLP also receives the own-query path through block i + 1, which
             PR #40's out-credit (lambda times the next mid credit) leaves out: a slightly more complete variant.
  synthesis  the exact credits at the nine sites and the top stream (one autograd.grad on the cut graph), transformed
             by a probing scheme (Scheme: partial rows, a control variate, ...), then parameter gradients block-locally,
             one autograd.grad per sublayer whose inputs are only that sublayer's parameters, so transformed credit
             never leaks between blocks (nor from a block's MLP into its attention): block i's attention and value_embeds[i] from the credit at S(i+1); block i's MLP
             from resid_lambdas[i+1] times the credit at S(i+2) (block 7's from the top-stream credit); wte from S0;
             the head and the mixing scalars from the cut graph. With the identity scheme this is PR #40's core estimate.
The Boost phase (the first 20% of updates as PR #40 schedules it; boost: false starts Cut at update 1) is
boost_gradients: PR #40's Boost, each sublayer against the shared head's readout of the stream right after it.

Synthesis-mode experiments, one per run, each a fixed rule (no knob but E13's theta):
  E10 (a) clip_credit   each token's credit norm at each site capped at CLIP times that site's batch median (Scheme).
  E10 (b) agree         each block nn.Linear's weight-update entry kept only where at least AGREE of the batch's
                        tokens agree in the sign of their contribution, from sign(G)^T sign(X) in one matrix multiply
                        (agreed_gradients).
  E11     runway        each token's credit at each site weighted by its runway: how far its loss keeps falling along
                        -g/|g| (runway_weights, through Replay).
  E13     budget_theta  LOW_ROWS rows per site for tokens whose n-gram and model losses are both at most theta
                        (loud/ngram.py), the full cycle for the rest (Scheme.read_tiers).

H8 has neither the record's cross-token patches (value path, input-key and gate probes, path scalars, future-token
probes) nor its top-512 MLP routes (it uses every unit), so H8 and backprop bracket the cross-token credit.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import sys

import torch
from torch import nn
import torch.nn.functional as F

from . import refs
from .ngram import NGram, low_tier
from .options import h8_options
from .train import current_rates, rate_labels, train

SOFTCAP = 15
SCHEME_SEED = 7907   # the probing scheme's signs: the job seed plus this
CLIP = 3.0           # E10 (a): a token's credit norm at a site is capped at this times the site's batch median
AGREE = 0.7          # E10 (b): the share of tokens that must agree in sign for a weight-update entry to be kept
LOW_ROWS = 64        # E13: rows per token per site in the low tier (the high tier reads the full cycle)
RUNWAY = (0.01, 0.02, 0.04, 0.08)  # E11: moves along -g/|g|, in the token's activation RMS per coordinate
RUNWAY_EVERY = 10    # E11: updates between logs of the runway distribution


def um():
    """utils.model, the fixed model's module (whichever pinned checkout imported it; the CPU checks patch its attention
    there)."""
    return sys.modules['utils.model']


def logits(model, x, head=None):
    """Soft-capped logits of a stream x before the final norm, as GPT.forward; `head` overrides the head's weight."""
    z = F.linear(um().norm(x), model.lm_head.weight if head is None else head)[..., :model.config.vocab_size].float()
    return SOFTCAP * torch.tanh(z / SOFTCAP)


def token_losses(model, x, targets, head=None):
    z = logits(model, x, head)
    return F.cross_entropy(z.flatten(0, 1), targets.flatten(), ignore_index=-1, reduction='none').view_as(targets)


def attention(attn, x, ve, cos_sin, window, held, freeze_queries=False):
    """CausalSelfAttention.forward on the normed block input x. With `held`, the keys', values' and gate's input is
    detached: every token's keys and values are their clean ones while its query stays live (freeze_queries detaches
    the query's input too: record #9's Cut, without own queries)."""
    m = um()
    B, T, _ = x.shape
    kv = x.detach() if held else x
    q = attn.c_q(x.detach() if freeze_queries else x).view(B, T, attn.n_head, attn.head_dim)
    k = attn.c_k(kv).view(B, T, attn.n_kv_head, attn.head_dim)
    v = attn.c_v(kv).view(B, T, attn.n_kv_head, attn.head_dim)
    if ve is not None:
        gate = 2 * torch.sigmoid(attn.ve_gate(kv[..., :attn.ve_gate_channels]))
        v = v + gate.unsqueeze(-1) * ve.view(B, T, attn.n_kv_head, attn.head_dim)
    cos, sin = cos_sin
    q, k = m.norm(m.apply_rotary_emb(q, cos, sin)), m.norm(m.apply_rotary_emb(k, cos, sin))
    return attn.c_proj(m.attention(q, k, v, window[0]).contiguous().view(B, T, -1))


def cut_forward(model, idx, targets, held=True, freeze_queries=False):
    """GPT.forward on the cut graph (held=False: the model's own forward). Returns the token losses [B, T], the nine
    sites, the top stream and each block's attention and MLP outputs (what synthesis mode differentiates)."""
    m = um()
    cos_sin = (model.cos[:, :idx.shape[1]], model.sin[:, :idx.shape[1]])
    e = model.transformer.wte(idx)
    x = x0 = m.norm(e)
    sites, attn_out, mlp_out = [e], [], []
    for i, block in enumerate(model.transformer.h):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        ve = model.value_embeds[str(i)](idx) if str(i) in model.value_embeds else None
        a = attention(block.attn, m.norm(x), ve, cos_sin, model.window_sizes[i], held, freeze_queries)
        x = x + a
        sites.append(x)
        attn_out.append(a)
        f = block.mlp(m.norm(x))
        x = x + f
        mlp_out.append(f)
    return token_losses(model, x, targets), sites, x, attn_out, mlp_out


def own_path_attention(attn, x, cos_sin, window):
    """Block 0's attention as Boost.embedding_block0 moves it: each token's own query, key and value live, the other
    positions' keys and values held. Dense (one T x T map per head), with the held parts entering by value only: the
    diagonal carries the live key's and value's difference from their held copies, zero in value and exact in gradient."""
    m = um()
    B, T, _ = x.shape
    q = m.norm(m.apply_rotary_emb(attn.c_q(x).view(B, T, attn.n_head, attn.head_dim), *cos_sin)).transpose(1, 2)
    k = m.norm(m.apply_rotary_emb(attn.c_k(x).view(B, T, attn.n_kv_head, attn.head_dim), *cos_sin)).transpose(1, 2)
    v = attn.c_v(x).view(B, T, attn.n_kv_head, attn.head_dim).transpose(1, 2)
    kd, vd = k.detach(), v.detach()
    scores = q @ kd.transpose(-1, -2) + torch.diag_embed((q * (k - kd)).sum(-1))
    i = torch.arange(T, device=x.device)
    allowed = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] <= window[0])
    p = (scores / attn.head_dim ** 0.5).masked_fill(~allowed, float('-inf')).softmax(-1)
    out = p @ vd + p.diagonal(dim1=-2, dim2=-1).unsqueeze(-1) * (v - vd)
    return attn.c_proj(out.transpose(1, 2).contiguous().view(B, T, -1))


def boost_loss(model, idx, targets):
    """PR #40's Boost as one loss to differentiate: each sublayer trained by the shared head's readout of the stream
    right after it, its input detached (attention left whole: Boost's internals measure it exactly); the head by the
    readouts after every sublayer of blocks 0..L-2 and the final one (Boost.estimate; not block L-1's attention
    readout); the mixing scalars by their block's attention readout with the block's inputs held (stage_losses); the
    embedding by block 0's attention readout along each token's own path (embedding_block0). Returns (loss, the
    model's own token losses)."""
    m = um()
    L, head = model.config.n_layer, model.lm_head.weight
    cos_sin = (model.cos[:, :idx.shape[1]], model.sin[:, :idx.shape[1]])
    e = model.transformer.wte(idx)
    x0 = m.norm(e)
    readout = lambda stream, live: token_losses(model, stream, targets, None if live else head.detach()).sum()
    total, x = 0.0, x0
    for i, block in enumerate(model.transformer.h):
        xin = model.resid_lambdas[i] * x.detach() + model.x0_lambdas[i] * x0.detach()
        ve = model.value_embeds[str(i)](idx) if str(i) in model.value_embeds else None
        mid = xin + attention(block.attn, m.norm(xin), ve, cos_sin, model.window_sizes[i], held=False)
        total = total + readout(mid, i < L - 1)
        x = mid.detach() + block.mlp(m.norm(mid.detach()))
        final = token_losses(model, x, targets) if i == L - 1 else None
        total = total + (final.sum() if i == L - 1 else readout(x, True))
    attn0, mix = model.transformer.h[0].attn, (model.resid_lambdas[0] + model.x0_lambdas[0]).detach()
    with frozen(attn0):
        xin0 = mix * m.norm(e)
        mid0 = xin0 + own_path_attention(attn0, m.norm(xin0), cos_sin, model.window_sizes[0])
    total = total + readout(mid0, False)
    return total / (targets != -1).sum(), final


class Replay:
    """The probe primitive `reforward` (plan v2, H5/H8): per-token losses with one site's activation moved, run on from
    that site with every key and value held at its clean value (the cut) and each token's own query live, as the chain
    runs its probes. Built from one clean forward of a batch (no grad).

        replay = Replay(model, idx, targets)
        losses, at = replay(site, delta, sites=True)   # delta [B, T, D, W]: D moves per token at site (0..L)

    losses [B, T, D]; with sites, also each later mid stream's change [B, T, D, W] per site index (E5's readings)."""

    @torch.no_grad()
    def __init__(self, model, idx, targets, chunk=128):
        m = um()
        self.model, self.targets, self.chunk = model, targets, chunk
        T = idx.shape[1]
        self.cos, self.sin = model.cos[:, :T], model.sin[:, :T]
        e = model.transformer.wte(idx)
        x = self.x0 = m.norm(e)
        self.sites, self.keys, self.values = [e], [], []
        for i, block in enumerate(model.transformer.h):
            attn = block.attn
            x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * self.x0
            h = m.norm(x)
            k = m.norm(m.apply_rotary_emb(attn.c_k(h).view(*h.shape[:2], attn.n_kv_head, attn.head_dim),
                                          self.cos, self.sin))
            v = attn.c_v(h).view(*h.shape[:2], attn.n_kv_head, attn.head_dim)
            if str(i) in model.value_embeds:
                gate = 2 * torch.sigmoid(attn.ve_gate(h[..., :attn.ve_gate_channels]))
                v = v + gate.unsqueeze(-1) * model.value_embeds[str(i)](idx).view_as(v)
            self.keys.append(k.transpose(1, 2))    # [B, H, T, d]
            self.values.append(v.transpose(1, 2))
            x = x + attention(attn, h, model.value_embeds[str(i)](idx) if str(i) in model.value_embeds else None,
                              (self.cos, self.sin), model.window_sizes[i], held=True)
            self.sites.append(x)
            x = x + block.mlp(m.norm(x))

    def held_attention(self, i, x, t0):
        """Block i's attention for moved block inputs x [B, C, D, W] of tokens t0..t0+C: their queries against every
        position's clean keys and values in their window."""
        m, attn = um(), self.model.transformer.h[i].attn
        B, C, D, _ = x.shape
        pos = torch.arange(t0, t0 + C, device=x.device)
        cos, sin = self.cos[:, pos].unsqueeze(2), self.sin[:, pos].unsqueeze(2)
        q = attn.c_q(m.norm(x)).view(B, C, D, attn.n_head, attn.head_dim)
        half = attn.head_dim // 2
        q1, q2 = q[..., :half], q[..., half:]
        q = m.norm(torch.cat([q1 * cos + q2 * sin, q1 * -sin + q2 * cos], -1))  # apply_rotary_emb at each position
        k, v = self.keys[i], self.values[i]
        T = k.shape[2]
        scores = torch.einsum('bcdhk,bhsk->bhcds', q.to(k.dtype), k) / attn.head_dim ** 0.5
        s = torch.arange(T, device=x.device)
        allowed = (s[None, :] <= pos[:, None]) & (pos[:, None] - s[None, :] <= self.model.window_sizes[i][0])
        p = scores.masked_fill(~allowed[None, None, :, None, :], float('-inf')).softmax(-1)
        out = torch.einsum('bhcds,bhsk->bcdhk', p.to(v.dtype), v)
        return attn.c_proj(out.reshape(B, C, D, -1))

    @torch.no_grad()
    def __call__(self, site, delta, sites=False):
        m, model = um(), self.model
        L = model.config.n_layer
        B, T, D, W = delta.shape
        losses = torch.empty(B, T, D, device=delta.device)
        at = {k: torch.empty_like(delta) for k in range(site + 1, L + 1)} if sites else None
        for t0 in range(0, T, self.chunk):
            sl = slice(t0, t0 + self.chunk)
            dl = delta[:, sl]
            x0 = self.x0[:, sl, None]
            if site == 0:
                x0 = m.norm(self.sites[0][:, sl, None] + dl)
                x, first = x0, 0
            else:
                mid = self.sites[site][:, sl, None] + dl
                x, first = mid + model.transformer.h[site - 1].mlp(m.norm(mid)), site
            for i in range(first, L):
                xin = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
                mid = xin + self.held_attention(i, xin, t0)
                if sites:
                    at[i + 1][:, sl] = mid - self.sites[i + 1][:, sl, None]
                x = mid + model.transformer.h[i].mlp(m.norm(mid))
            y = self.targets[:, sl, None].expand(-1, -1, D)
            losses[:, sl] = token_losses(model, x.flatten(1, 2), y.flatten(1, 2)).view(B, -1, D)
        return (losses, at) if sites else losses


class frozen:
    """A module's parameters as constants for the duration (their gradient comes from elsewhere)."""

    def __init__(self, module):
        self.params = [p for p in module.parameters() if p.requires_grad]

    def __enter__(self):
        for p in self.params:
            p.requires_grad_(False)

    def __exit__(self, *exc):
        for p in self.params:
            p.requires_grad_(True)


def identity_guess(model, sites, g_top):
    """The identity path's guess at every site: the top-stream credit carried down the residual scalars (at S(i+1):
    the product of resid_lambdas above block i), and at S0 through x0's skips (each block reads x0 with its
    x0_lambda) and one step back through the RMS norm. Detached: a computed prior, not a measurement."""
    lam_r, lam_0 = model.resid_lambdas.detach().float(), model.x0_lambdas.detach().float()
    L = len(lam_r)
    above = torch.ones(L + 1, device=lam_r.device)  # above[i] = product of lam_r[i:]
    for i in reversed(range(L)):
        above[i] = above[i + 1] * lam_r[i]
    guess = [None] + [above[i + 1] * g_top for i in range(L)]
    x0_coefficient = above[0] + sum(lam_0[i] * above[i + 1] for i in range(L))
    e = sites[0].detach().float().requires_grad_(True)
    with torch.enable_grad():
        guess[0] = torch.autograd.grad(um().norm(e), e, grad_outputs=x0_coefficient * g_top)[0]
    return guess


class Scheme:
    """A probing scheme for synthesis mode: how the chain's probes would read each site's credit. rows (m): each token
    reads m rows of the site's signed Hadamard basis (the record's: rows of +-1 entries with random column signs,
    fresh per update, site and sequence), credit = (1/m) sum_j (v_j . g) v_j, unbiased because E[v v^T] = I.
    row_assign 'time': every token of an update reads rows [(s mod K) m, (s mod K + 1) m), K = 512 / m; 'token': token
    t reads rows offset by (t mod K) m, so every update covers the basis across tokens. prior 'identity' (E7): the
    estimate is p + (1/m) sum_j (v_j . (g - p)) v_j with p the identity path's guess, computed, not measured.
    clip_credit (E10 a) caps each token's credit norm at CLIP times the site's batch median. budget_theta (E13): the
    low tier's tokens read LOW_ROWS rows, token-assigned among the tier's tokens of a sequence (read_tiers), the others
    the full cycle. m = 512 with none of these is the identity (the exact credit)."""

    def __init__(self, o, width, device, seed):
        self.rows, self.assign, self.prior = o['rows'] or width, o['row_assign'], o['prior']
        self.clip, self.tiers = o['clip_credit'], bool(o['budget_theta'])
        self.width = width
        self.hadamard = hadamard(width, device)
        self.generator = torch.Generator(device).manual_seed(seed + SCHEME_SEED)

    @property
    def identity(self):
        return self.rows == self.width and self.prior == 'none' and not self.clip and not self.tiers

    def __call__(self, credits, step, guess=None, low=None):
        """credits: per site [B, T, W]; guess: the identity path's (prior); low: E13's low tier [B, T]."""
        if self.identity:
            return credits
        out = []
        with torch.autocast(credits[0].device.type, enabled=False):  # the decoding in float32, as the record's
            for k, g in enumerate(credits):
                g = g.float()
                if self.clip:
                    out.append(clip(g))
                elif self.tiers:
                    out.append(self.read_tiers(g, low))
                else:
                    p = guess[k].float() if self.prior == 'identity' else None
                    out.append(self.read(g if p is None else g - p, step) + (0 if p is None else p))
        return out

    def signs(self, g):
        shape = (g.shape[0], 1, g.shape[-1])
        return torch.randint(0, 2, shape, device=g.device, generator=self.generator).float() * 2 - 1

    def read(self, g, step):
        """The partial-row estimate of g [B, T, W] for one site."""
        B, T, W = g.shape
        m, K = self.rows, W // self.rows
        signs = self.signs(g)
        if m == W:
            return g
        blocks = self.hadamard.view(K, m, W)
        out = torch.empty_like(g)
        if self.assign == 'time':
            rows = blocks[step % K] * signs  # [B, m, W]
            return (g @ rows.transpose(1, 2)) @ rows / m
        for k in range(K):  # token t reads block t mod K
            rows = blocks[k] * signs
            part = g[:, k::K]
            out[:, k::K] = (part @ rows.transpose(1, 2)) @ rows / m
        return out

    def read_tiers(self, g, low):
        """E13: the low tier's tokens read LOW_ROWS rows (the j-th low token of a sequence block j mod K, so the tier
        covers the basis), scaled to stay unbiased; the high tier reads the full cycle, which is the exact credit."""
        B, T, W = g.shape
        m = min(LOW_ROWS, W)
        K = W // m
        signs, blocks, out = self.signs(g), self.hadamard.view(K, m, W), g.clone()
        for b in range(B):
            tokens = low[b].nonzero().squeeze(1)
            for k in range(K):
                chosen = tokens[k::K]
                rows = blocks[k] * signs[b]
                out[b, chosen] = (g[b, chosen] @ rows.T) @ rows / m
        return out


def clip(g):
    """E10 (a): each token's credit [B, T, W] scaled down to at most CLIP times the median token norm of the batch."""
    norms = g.norm(dim=-1)
    return g * (CLIP * norms.flatten().median() / norms.clamp_min(1e-30)).clamp(max=1).unsqueeze(-1)


@contextmanager
def capture_linears(model, io):
    """E10 (b): io[module] = (input, output) of every block nn.Linear for one forward (the input as the matmul sees it,
    in the output's dtype)."""
    def hook(module, args, out):
        io[module] = (args[0].to(out.dtype), out)
    handles = [m.register_forward_hook(hook) for m in model.transformer.h.modules() if isinstance(m, nn.Linear)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def agreed_gradients(module, params, out, credit, io, tokens, kept):
    """E10 (b): a sublayer's block-local gradients with each nn.Linear weight's entries kept only where the tokens
    agree. G (the credit at the layer's output, from this sublayer's credit alone) and X (its input) give the weight
    gradient G^T X and, in one more matrix multiply, S = sign(G)^T sign(X), the net votes for each entry's sign; an
    entry is kept if |S| >= (2 AGREE - 1) x tokens, which is 'at least AGREE of the tokens agree' when every
    contribution is nonzero. A zero contribution (a unit off after the ReLU^2, an exact zero credit) votes for neither
    sign, so it counts against agreement, as a rule computed in one matrix multiply must count it. Other parameters
    (the value-embedding table) keep their gradients. Appends each weight's kept share to `kept`."""
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    weights = {id(m.weight) for m in linears}
    others = [p for p in params if id(p) not in weights]
    grads = torch.autograd.grad(out, [io[m][1] for m in linears] + others, grad_outputs=credit.to(out.dtype),
                                retain_graph=True)
    found = dict(zip(map(id, others), grads[len(linears):]))
    with torch.autocast(out.device.type, enabled=False):  # float32: the vote counts must be exact
        for m, G in zip(linears, grads):
            G2, X2 = G.flatten(0, -2).float(), io[m][0].flatten(0, -2).float()
            keep = (G2.sign().T @ X2.sign()).abs() >= (2 * AGREE - 1) * tokens
            found[id(m.weight)] = ((G2.T @ X2) * keep).to(m.weight.dtype)
            kept.append(keep.float().mean())
    return found


def runway_weights(model, idx, targets, sites, credits):
    """E11: per site and token, the move along -g/|g| of d x RMS(h) per coordinate (length d x RMS(h) x sqrt(W)), d in
    RUNWAY, with the lowest loss (Replay, on the cut graph's terms; the clean loss from the same replay, so both share
    its arithmetic), or none if no move lowers the loss; alpha = that d (0 for none), normalized to mean 1 per site over
    the batch's tokens. Returns the alphas [B, T] per site and the chosen move's index (0: none, 1..4: RUNWAY)."""
    replay = Replay(model, idx, targets)
    grid = torch.tensor((0.0,) + RUNWAY, device=idx.device)
    valid = targets != -1
    alphas, chosen = [], []
    for k, (h, g) in enumerate(zip(sites, credits)):
        h, g = h.detach().float(), g.detach().float()
        reach = h.square().mean(-1, keepdim=True).sqrt() * h.shape[-1] ** 0.5
        u = -g / g.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        losses = replay(k, grid.view(1, 1, -1, 1) * (reach * u).unsqueeze(2))
        best = losses[..., 1:].min(-1)
        falls = (best.values < losses[..., 0]) & valid
        alpha = torch.where(falls, grid[1:][best.indices], torch.zeros_like(best.values))
        mean = alpha[valid].mean()
        alphas.append(torch.where(mean > 0, alpha / mean.clamp_min(1e-30), alpha))
        chosen.append(torch.where(falls, best.indices + 1, torch.zeros_like(best.indices)))
    return alphas, chosen


def hadamard(d, device):
    """The Sylvester Hadamard matrix of order d (a power of two), entries +-1, as PR #40 builds it."""
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


def synthesis_gradients(model, idx, targets, scheme, step, freeze_queries=False, compare=False, agree=False,
                        runway=False, tiers=None, log=False):
    """Synthesis mode's parameter gradients (module docstring), the model's token losses and metrics fields: with
    `compare`, the cosine of the summed block-local weight gradient to the identity's (E6's log); E10 (b) `agree`
    (the share of entries kept logged when `log`); E11 `runway` (its distribution logged when `log`); E13 `tiers`, a
    function of the token losses giving the low tier (its share logged)."""
    L = model.config.n_layer
    io = {}
    with capture_linears(model, io) if agree else nullcontext():
        losses, sites, top, attn_out, mlp_out = cut_forward(model, idx, targets, True, freeze_queries)
    loss = losses.sum() / (targets != -1).sum()
    shared = [model.lm_head.weight, model.resid_lambdas, model.x0_lambdas]
    *credits, g_top, g_head, g_lr, g_l0 = torch.autograd.grad(loss, sites + [top] + shared, retain_graph=True)
    guess = identity_guess(model, sites, g_top.float()) if scheme.prior == 'identity' else None
    fields, weighted = {}, credits
    if runway:
        alphas, chosen = runway_weights(model, idx, targets, sites, credits)
        weighted = [g.float() * a.unsqueeze(-1) for g, a in zip(credits, alphas)]
        if log:
            share = lambda c: (torch.bincount(c.flatten(), minlength=len(RUNWAY) + 1) / c.numel()).tolist()
            fields['runway'] = {f'S{k}': [round(v, 4) for v in share(c)] for k, c in enumerate(chosen)}
    low = tiers(losses.detach()) if tiers is not None else None
    if low is not None:
        fields['low_tier'] = round(low.float().mean().item(), 4)
    shaped = scheme(weighted, step, guess, low)
    lam = model.resid_lambdas.detach()
    tokens = int((targets != -1).sum()) if agree else None
    kept = []

    def block_local(credit, masked):
        """Each sublayer's parameters from its own output's credit alone: the attention's call stops at its
        parameters, and so does the MLP's (one call for both would carry the MLP's credit back through its input,
        the mid stream, into the attention)."""
        found = {}
        for i, block in enumerate(model.transformer.h):
            ve = [model.value_embeds[str(i)].weight] if str(i) in model.value_embeds else []
            for module, params, out, g in ((block.attn, list(block.attn.parameters()) + ve, attn_out[i], credit[i + 1]),
                                           (block.mlp, list(block.mlp.parameters()), mlp_out[i],
                                            lam[i + 1] * credit[i + 2] if i < L - 1 else g_top)):
                if masked:
                    found.update(agreed_gradients(module, params, out, g, io, tokens, kept))
                else:
                    found.update(zip(map(id, params), torch.autograd.grad(out, params, grad_outputs=g.to(out.dtype),
                                                                          retain_graph=True)))
        found[id(model.transformer.wte.weight)] = torch.autograd.grad(
            sites[0], model.transformer.wte.weight, grad_outputs=credit[0].to(sites[0].dtype), retain_graph=True)[0]
        return found

    grads = block_local(shaped, agree)
    grads.update(zip(map(id, shared), (g_head, g_lr, g_l0)))
    if agree and log:
        fields['agree_kept'] = round(torch.stack(kept).mean().item(), 6)  # entries kept, mean over block weights
    if compare and (not scheme.identity or agree or runway):
        exact = block_local(credits, False)
        keys = list(exact)  # block-local only: the head and scalars are the cut graph's in both
        a = torch.cat([grads[k].float().flatten() for k in keys])
        b = torch.cat([exact[k].float().flatten() for k in keys])
        fields['cosine_to_identity'] = float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30))
    return grads, losses, fields


class H8:
    """The trainer (loud/train.py's interface) on PR #40's recipe: its model, init, optimizers (ANVIL II on the block
    matrices, Adam elsewhere), rates, warmdown, Boost -> Cut schedule (stage_schedule) and optimizer handoff, at the
    record's 4 sequences per update. The whole batch goes through one forward (H8's credits are per token, so there is
    no per-GPU layout to emulate)."""

    def __init__(self, o, device, seed, steps, heldout):
        from .cut import record_settings
        m = self.m = refs.trainer_module('pr40')
        self.o, self.device, self.seed, self.steps = o, device, seed, steps
        self.settings = s = record_settings(m, o['preset'], o['batch'])
        if not o['boost']:
            s.switch = 0.0  # Cut from the first update
        self.schedule = m.stage_schedule(steps * o['batch'], s)
        self.horizons = {phase: horizon for phase, _, _, _, horizon in self.schedule}
        self.reset()

    def reset(self, phase=None):
        m, s = self.m, self.settings
        self.phase = phase or self.schedule[0][0]
        self.model = m.make_model(self.device, self.seed, s.model, s.projection_std, s.head_std).requires_grad_(True)
        self.params = list(self.model.parameters())
        self.optimizer = m.make_optimizer(self.model, s, self.horizons[self.phase])
        self.labels = rate_labels(self.optimizer, s.lr)
        self.scheme = Scheme(self.o, s.model.n_embd, self.device, self.seed)
        self.ngram = NGram(s.model.vocab_size) if self.o['budget_theta'] else None  # E13: the stream so far

    def warmup(self, batch):
        """Every phase's code path once on random tokens, from fresh models (loud/train.py then resets)."""
        for phase in self.horizons:
            self.reset(phase)
            self.update(next(i for i, b in enumerate(self.schedule, 1) if b[0] == phase), batch)

    def describe(self):
        return dict(name='h8', file=refs.TRAINERS['pr40'], settings={k: v for k, v in asdict(self.settings).items()
                                                                       if k != 'model'},
                    model=asdict(self.settings.model), schedule=dict(phases=self.horizons, steps=len(self.schedule)))

    def update(self, step, batch):
        m, o = self.m, self.o
        phase, phase_step, _, _, horizon = self.schedule[step - 1]
        if phase != self.phase:
            optimizer = m.make_optimizer(self.model, self.settings, horizon)
            m.handoff_optimizer(self.optimizer, optimizer, phase_step)
            self.optimizer, self.phase = optimizer, phase
            self.labels = rate_labels(optimizer, self.settings.lr)
        m.set_rate(self.optimizer, phase_step, horizon, self.settings.warmdown, self.settings.final)
        fields = dict(current_rates(self.optimizer, self.labels), phase=phase)
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        compare = bool(o['compare_every']) and step % o['compare_every'] == 0
        tiers = None
        if self.ngram is not None:  # E13: score the batch under the batches before it, then add it (every phase)
            rows = batch.cpu().numpy()
            if phase != 'boost':
                ngram = torch.from_numpy(self.ngram.losses(rows)).float().to(self.device)
                tiers = lambda losses: low_tier(ngram, losses, o['budget_theta'])
            self.ngram.insert(rows)
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            if phase == 'boost':
                loss, losses = boost_loss(self.model, x, y)
                grads = dict(zip(map(id, self.params), torch.autograd.grad(loss, self.params, allow_unused=True)))
            elif o['mode'] == 'cut':
                losses = cut_forward(self.model, x, y, True, o['freeze_queries'])[0]
                loss = losses.sum() / (y != -1).sum()
                grads = dict(zip(map(id, self.params), torch.autograd.grad(loss, self.params, allow_unused=True)))
            else:
                grads, losses, extra = synthesis_gradients(self.model, x, y, self.scheme, step, o['freeze_queries'],
                                                           compare, o['agree'], o['runway'], tiers,
                                                           log=step % RUNWAY_EVERY == 0)
                fields.update(extra)
        for p in self.params:
            g = grads.get(id(p))
            p.grad = torch.zeros_like(p) if g is None else g.to(p.dtype)
        if o['rank_every'] and step % o['rank_every'] == 0:
            fields['stable_rank'] = stable_ranks(self.model, self.optimizer)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return losses.detach().float()[y != -1].mean(), fields

    def diagnose(self, step):
        return {}


@torch.no_grad()
def stable_ranks(model, optimizer):
    """E6's log: the stable rank |M|_F^2 / |M|_2^2 of what ANVIL II's next step orthogonalizes, per matrix type (the
    median over blocks), M rebuilt from its rails and the current gradient as Anvil.step forms it."""
    anvil = optimizer.optimizers[0]
    beta, engaged = anvil.rail_beta(), anvil.t >= anvil.engage
    fast_beta, weight = (anvil.fast_beta, anvil.fast_weight) if engaged else (beta, 1.0)
    out = {}
    for name, p in model.named_parameters():
        state = anvil.state.get(p)
        if not state or p.grad is None:
            continue
        g = p.grad.float()
        fast = state['fast'].lerp(g, 1 - fast_beta)
        slow = state['slow'].lerp(g, 1 - anvil.slow_beta)
        M = g.lerp(weight * fast + (1 - weight) * slow, beta)
        rank = (M.square().sum() / torch.linalg.matrix_norm(M, ord=2).square().clamp_min(1e-30)).item()
        out.setdefault('.'.join(name.split('.')[3:5]), []).append(rank)
    return {kind: round(sorted(v)[len(v) // 2], 3) for kind, v in out.items()}


def run(spec, run_dir):
    o = h8_options(spec.get('args', {}))
    return train(spec, run_dir, o, lambda device, seed, steps, heldout: H8(o, device, seed, steps, heldout),
                 method=f'H8 ({o["mode"]} mode' + ('' if o['boost'] else ', no Boost') + ')')
