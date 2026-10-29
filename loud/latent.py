"""E14 and E15's latent objectives (plan v2), in H8's cut and boost modes.

E14, NextLat (github.com/JaydenTeoh/NextLat at 3770be6, models/model_nextlat.py, followed except where v2 departs):
the latent h_t is the final-norm output; Dynamics predicts h_{t+1} from h_t and the next token's raw embedding (its
`token_embedding(batch)` output, here wte's): a LayerNorm without bias over [next embedding, h_t], an MLP, and the
result added to h_t. NextLat's MLP has two GELU hidden layers of width 128 * round(2 W / 128); v2 keeps one (width
1,024, the same for W = 512) so that a forward-only version can train it with one nonlinear step. Weights normal with
std 0.02 and no biases, as NextLat. The loss is NextLat's MSE term: smooth-L1 with beta 1 against the target latent
(NextLat: the live network's, detached), averaged over the unmasked (batch, position, dimension) elements, masked
where token t+1 is <|endoftext|> (it opens the next document, so h_{t+1} belongs to another one). Horizon 1, no KL.
Its gradient reaches the trunk through h_t and wte through the next token's embedding, as in NextLat, never the
head (h_t does not pass through it).

E15, Boost with a latent target: during Boost, a predictor per sublayer (one GELU hidden layer, 512 -> 1024 -> 512)
maps the stream after the sublayer to the EMA teacher's final latent for the same token, with loss 1 - cosine; the
last sublayer (block L-1's MLP, read by the final readout) has none. The predictors are discarded at the end.
"""
import torch
from torch import nn
import torch.nn.functional as F

EOT = 0          # <|endoftext|> in Noise's pinned BPE-4096 tokenizer; utils/prepare_data.py opens each document with it
HIDDEN = 1024    # the one hidden layer of E14's dynamics model and E15's predictors
INIT_STD = 0.02  # NextLat's init of every Linear


def mlp(width_in, width_out):
    """Linear -> GELU -> Linear, no biases, NextLat's init."""
    layers = nn.Sequential(nn.Linear(width_in, HIDDEN, bias=False), nn.GELU(), nn.Linear(HIDDEN, width_out, bias=False))
    for layer in layers[::2]:
        nn.init.normal_(layer.weight, std=INIT_STD)
    return layers


class Dynamics(nn.Module):
    """E14's latent dynamics model: h_t + MLP(LayerNorm([e_{t+1}, h_t]))."""

    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(2 * width, bias=False)
        self.mlp = mlp(2 * width, width)

    def forward(self, latent, next_embedding):
        return latent + self.mlp(self.norm(torch.cat([next_embedding, latent], -1)))


def nextlat_loss(dynamics, latent, embedding, idx, target):
    """NextLat's horizon-1 loss for latents [B, T, W] (their gradient flows), the raw token embeddings [B, T, W] of the
    inputs idx [B, T] and the target latents (no gradient): smooth-L1 between dynamics(h_t, e_{t+1}) and target_{t+1},
    the mean over the elements whose token t+1 is not <|endoftext|>."""
    pred = dynamics(latent[:, :-1], embedding[:, 1:])
    error = F.smooth_l1_loss(pred.float(), target[:, 1:].detach().float(), beta=1.0, reduction='none')
    keep = (idx[:, 1:] != EOT).unsqueeze(-1).to(error.dtype)
    return (error * keep).sum() / (keep.sum() * error.shape[-1]).clamp_min(1.0)


class LatentTargets(nn.Module):
    """E15: one predictor per Boost sublayer but the last (sublayer 2i is block i's attention, 2i + 1 its MLP), and
    their loss against the teacher's final latents (`target`, set each update). `lens` keeps Boost's logit-lens
    readouts beside them ('plus') or drops them ('only'); `weight` is the latent loss's lambda. While `logging`, `seen`
    gathers each sublayer's mean readout and latent losses."""

    def __init__(self, n_layer, width, lens, weight):
        super().__init__()
        self.predictors = nn.ModuleList(mlp(width, width) for _ in range(2 * n_layer - 1))
        self.lens, self.weight = lens, weight
        self.target, self.logging, self.seen = None, False, {}

    def forward(self, k, stream, own=False):
        """lambda times the summed (1 - cosine) of sublayer k's predicted latents against the teacher's. own: the
        embedding's own-path stream (block 0's attention, as Boost's embedding readout), through predictor 0 held
        fixed, so it trains the embedding only (the predictor learns from the sublayer's own stream)."""
        predictor = self.predictors[k]
        if own:
            frozen = {n: p.detach() for n, p in predictor.named_parameters()}
            predicted = torch.func.functional_call(predictor, frozen, (stream,))
        else:
            predicted = predictor(stream)
        miss = 1 - F.cosine_similarity(predicted.float(), self.target.float(), dim=-1)
        if self.logging and not own:
            self.seen.setdefault(k, {})['latent'] = miss.detach().mean()
        return self.weight * miss.sum()

    def note(self, k, readout_sum, count):
        if self.logging:
            self.seen.setdefault(k, {})['readout'] = readout_sum.detach() / count


def effective_rank(x):
    """exp of the entropy of the normalized singular values of x [N, W] (Roy and Vetterli)."""
    s = torch.linalg.svdvals(x.float())
    p = s / s.sum().clamp_min(1e-30)
    return torch.exp(-(p * p.clamp_min(1e-30).log()).sum()).item()
