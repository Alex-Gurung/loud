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
             never leaks between blocks (nor from a block's MLP into its attention): block i's attention and
             value_embeds[i] from the credit at S(i+1); block i's MLP from resid_lambdas[i+1] times the credit at
             S(i+2) (block 7's from the top-stream credit); wte from S0; the head and the mixing scalars from the cut
             graph. With the identity scheme this is PR #40's core estimate.
The Boost phase (the first 20% of updates as PR #40 schedules it; boost: false starts Cut at update 1) is
boost_loss: PR #40's Boost, each sublayer against the shared head's readout of the stream right after it.

Latent objectives (loud/latent.py; cut mode, one per run): E14's NextLat loss joins the active phase's loss from its
onset (the handoff, or the first update), trained into the cut graph (or, during Boost, the Boost graph) and into its
dynamics model; E15's sublayer predictors join or replace Boost's readouts. Their targets come from H4's EMA teacher
(loud/ema.py), updated after every step; the dynamics model and predictors have their own Adam at the record's base
Adam rate, betas and schedule. Neither enters final.pt.

Synthesis-mode experiments, one per run, each a fixed rule (no knob but E13's theta):
  E10 (a) clip_credit   each token's credit norm at each site capped at CLIP times that site's batch median (Scheme).
  E10 (b) agree         each block nn.Linear's weight-update entry kept only where at least AGREE of the batch's
                        tokens agree in the sign of their contribution, from sign(G)^T sign(X) in one matrix multiply
                        (agreed_gradients).
  E11     runway        each token's credit at each site weighted by its runway: how far its loss keeps falling along
                        -g/|g| (runway_weights, through Replay).
  E13     budget_theta  LOW_ROWS rows per site for tokens whose n-gram and model losses are both at most theta
                        (loud/ngram.py), the full cycle for the rest (Scheme.read_tiers).
  E8      snr_scaling   in the Cut phase, ANVIL II's polar factor replaced by U diag(r) V^T from each sequence's
                        synthesized gradient (sequence_gradients; loud/snr.py), on a partial-row scheme, every arm
                        (control, snr, rate) without the lane gain (ShapedBatchedAnvil, loud/anvil.py).

H8 has neither the record's cross-token patches (value path, input-key and gate probes, path scalars, future-token
probes) nor its top-512 MLP routes (it uses every unit), so H8 and backprop bracket the cross-token credit.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F

from . import arch, latent, refs
from .anvil import shaped_optimizer
from .ema import EMAWeights
from .ngram import NGram, low_tier
from .options import h8_options
from .snr import SNRScaling
from .train import current_rates, rate_labels, train

SOFTCAP = 15
SCHEME_SEED = 7907   # the probing scheme's signs: the job seed plus this
KEY_SEED = 7919      # E17's key-probe rows: the job seed plus this
PROBE_ROWS = 4       # held-out sequences E14's and E15's untimed logs measure
AUX_LOG_EVERY = 50   # E14's cosine of its gradient to the next-token one at the top stream, every this many updates
LATENT_LOG_EVERY = 10  # E15's per-sublayer readout and latent losses, every this many Boost updates
CLIP = 3.0           # E10 (a): a token's credit norm at a site is capped at this times the site's batch median
AGREE = 0.7          # E10 (b): the share of tokens that must agree in sign for a weight-update entry to be kept
# (CLIP, AGREE and RUNWAY are v2's fixed rules; a run's clip_factor, agree_share and runway_grid set them, one trainer
# per process)
LOW_ROWS = 64        # E13: rows per token per site in the low tier (the high tier reads the full cycle)
RUNWAY = RUNWAY_V2 = (0.01, 0.02, 0.04, 0.08)  # E11: moves along -g/|g|, in the token's RMS per coordinate
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


class ReadCredit(torch.autograd.Function):
    """The identity forward; backward passes on only the partial-row read of the incoming credit g [B, T, W]:
    (g R^T) R / m for the m rows R (+-1 entries, E[R^T R / m] = I), in float32. What m probes per token would measure.
    Rows [m, W]: every token's; [B, m, W]: each sequence's own; [B, K, m, W]: each sequence's, token t reading block
    t mod K (E6's token assignment)."""

    @staticmethod
    def forward(ctx, x, rows):
        ctx.save_for_backward(rows)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        rows, = ctx.saved_tensors
        with torch.autocast(g.device.type, enabled=False):
            r, g32 = rows.float(), g.float()
            if r.dim() < 4:  # [m, W] broadcasts over the sequences; [B, m, W] reads each with its own
                read = (g32 @ r.transpose(-1, -2)) @ r / r.shape[-2]
            else:
                K, m = r.shape[1:3]
                read = torch.empty_like(g32)
                for k in range(K):
                    read[:, k::K] = (g32[:, k::K] @ r[:, k].transpose(-1, -2)) @ r[:, k] / m
        return read.to(g.dtype), None


def attention(attn, x, ve, cos_sin, window, held, freeze_queries=False, live=(), key_x=None):
    """CausalSelfAttention.forward on the normed block input x. With `held`, the keys', values' and gate's input is
    detached: every token's keys and values are their clean ones while its query stays live (freeze_queries detaches
    the query's input too: record #9's Cut, without own queries). `live` names cross-token paths held keeps open
    ('keys', 'values': the values and their gate), to measure what each carries. key_x (with the keys live): the key
    branch's input in place of x, the same value (cut_forward's partial read of the key credit)."""
    m = um()
    B, T, _ = x.shape
    kx = x if not held or 'keys' in live else x.detach()
    if key_x is not None and held and 'keys' in live:
        kx = key_x
    vx = x if not held or 'values' in live else x.detach()
    q = attn.c_q(x.detach() if freeze_queries else x).view(B, T, attn.n_head, attn.head_dim)
    k = attn.c_k(kx).view(B, T, attn.n_kv_head, attn.head_dim)
    v = attn.c_v(vx).view(B, T, attn.n_kv_head, attn.head_dim)
    if ve is not None:
        gate = 2 * torch.sigmoid(attn.ve_gate(vx[..., :attn.ve_gate_channels]))
        v = v + gate.unsqueeze(-1) * ve.view(B, T, attn.n_kv_head, attn.head_dim)
    cos, sin = cos_sin
    q, k = m.norm(m.apply_rotary_emb(q, cos, sin)), m.norm(m.apply_rotary_emb(k, cos, sin))
    return attn.c_proj(m.attention(q, k, v, window[0]).contiguous().view(B, T, -1))


def cut_forward(model, idx, targets, held=True, freeze_queries=False, live=(), live_blocks=None, key_rows=None):
    """GPT.forward on the cut graph (held=False: the model's own forward). Returns the token losses [B, T] (None
    without targets), the nine sites, the top stream and each block's attention and MLP outputs (what synthesis mode
    differentiates). key_rows: per block, the rows that read its live key path's credit at the block input, before
    the key branch's RMSNorm, where the record's input-key probes move it (None: exact)."""
    m = um()
    cos_sin = (model.cos[:, :idx.shape[1]], model.sin[:, :idx.shape[1]])
    e = model.transformer.wte(idx)
    x = x0 = m.norm(e)
    ngram = arch.ngram_features(model, idx)
    sites, attn_out, mlp_out = [e], [], []
    for i, block in enumerate(model.transformer.h):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        ve = arch.value_embedding(model, idx, i, ngram)
        open_ = live if live_blocks is None or i in live_blocks else ()  # E17: the live paths of chosen blocks only
        key_x = None
        if key_rows is not None and held and 'keys' in open_:  # norm(x) in value; the read before the norm
            key_x = m.norm(ReadCredit.apply(x, key_rows[i]))
        a = attention(block.attn, m.norm(x), ve, cos_sin, model.window_sizes[i], held, freeze_queries, open_, key_x)
        x = x + a
        sites.append(x)
        attn_out.append(a)
        f = block.mlp(m.norm(x))
        x = x + f
        mlp_out.append(f)
    return None if targets is None else token_losses(model, x, targets), sites, x, attn_out, mlp_out


def own_path_attention(attn, x, cos_sin, window, ve=None):
    """Block 0's attention as Boost.embedding_block0 moves it: each token's own query, key and value live, the other
    positions' keys and values held. Dense (one T x T map per head), with the held parts entering by value only: the
    diagonal carries the live key's and value's difference from their held copies, zero in value and exact in gradient.
    A value embedding (the fixed model has none at block 0) enters held, through its live gate."""
    m = um()
    B, T, _ = x.shape
    q = m.norm(m.apply_rotary_emb(attn.c_q(x).view(B, T, attn.n_head, attn.head_dim), *cos_sin)).transpose(1, 2)
    k = m.norm(m.apply_rotary_emb(attn.c_k(x).view(B, T, attn.n_kv_head, attn.head_dim), *cos_sin)).transpose(1, 2)
    v = attn.c_v(x).view(B, T, attn.n_kv_head, attn.head_dim)
    if ve is not None:
        gate = 2 * torch.sigmoid(attn.ve_gate(x[..., :attn.ve_gate_channels]))
        v = v + gate.unsqueeze(-1) * ve.detach().view_as(v)
    v = v.transpose(1, 2)
    kd, vd = k.detach(), v.detach()
    scores = q @ kd.transpose(-1, -2)
    scores.diagonal(dim1=-2, dim2=-1).add_((q * (k - kd)).sum(-1))  # in place: no second T x T map
    i = torch.arange(T, device=x.device)
    allowed = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] <= window[0])
    p = (scores / attn.head_dim ** 0.5).masked_fill(~allowed, float('-inf')).softmax(-1)
    out = p @ vd + p.diagonal(dim1=-2, dim2=-1).unsqueeze(-1) * (v - vd)
    return attn.c_proj(out.transpose(1, 2).contiguous().view(B, T, -1))


def boost_loss(model, idx, targets, side=None, keep=None):
    """PR #40's Boost as one loss to differentiate: each sublayer trained by the shared head's readout of the stream
    right after it, its input detached (attention left whole: Boost's internals measure it exactly); the head by the
    readouts after every sublayer of blocks 0..L-2 and the final one (Boost.estimate; not block L-1's attention
    readout); the mixing scalars by their block's attention readout with the block's inputs held (stage_losses); the
    embedding by block 0's attention readout along each token's own path (embedding_block0). Returns (loss, the
    model's own token losses).
    side (E15, loud.latent.LatentTargets): every sublayer's stream but the last's also, or instead of its readout
    (side.lens), predicts the teacher's latent. keep: a dict that receives the top stream and the raw embedding
    (E14)."""
    m = um()
    L, head = model.config.n_layer, model.lm_head.weight
    count = (targets != -1).sum()
    cos_sin = (model.cos[:, :idx.shape[1]], model.sin[:, :idx.shape[1]])
    e = model.transformer.wte(idx)
    x0 = m.norm(e)
    ngram = arch.ngram_features(model, idx)

    def sublayer(k, stream, live, own=False):
        """Sublayer k's terms on its stream (k = 2i: block i's attention, 2i + 1: its MLP); own: the embedding's
        path."""
        if side is not None and not side.lens:
            if side.logging and not own:
                with torch.no_grad():
                    side.note(k, token_losses(model, stream, targets).sum(), count)
            return side(k, stream, own)
        readout = token_losses(model, stream, targets, None if live else head.detach()).sum()
        if side is None:
            return readout
        if not own:
            side.note(k, readout, count)
        return readout + side(k, stream, own)

    total, x = 0.0, x0
    for i, block in enumerate(model.transformer.h):
        xin = model.resid_lambdas[i] * x.detach() + model.x0_lambdas[i] * x0.detach()
        ve = arch.value_embedding(model, idx, i, ngram)
        mid = xin + attention(block.attn, m.norm(xin), ve, cos_sin, model.window_sizes[i], held=False)
        total = total + sublayer(2 * i, mid, i < L - 1)
        x = mid.detach() + block.mlp(m.norm(mid.detach()))
        final = token_losses(model, x, targets) if i == L - 1 else None
        total = total + (final.sum() if i == L - 1 else sublayer(2 * i + 1, x, True))
    attn0, mix = model.transformer.h[0].attn, (model.resid_lambdas[0] + model.x0_lambdas[0]).detach()
    with frozen(attn0):
        xin0 = mix * m.norm(e)
        mid0 = xin0 + own_path_attention(attn0, m.norm(xin0), cos_sin, model.window_sizes[0],
                                         arch.value_embedding(model, idx, 0, ngram))
    total = total + sublayer(0, mid0, False, own=True)
    if keep is not None:
        keep.update(top=x, embedding=e)
    return total / count, final


class Replay:
    """The probe primitive `reforward` (plan v2, H5/H8): per-token losses with one site's activation moved, run on from
    that site with every key and value held at its clean value (the cut) and each token's own query live, as the chain
    runs its probes. Built from one clean forward of a batch (no grad).

        replay = Replay(model, idx, targets)
        losses, at = replay(site, delta, sites=True)   # delta [B, T, D, W]: D moves per token at site (0..L)
        losses = replay(site, delta[:, :16], start=32)  # or tokens 32..47 only

    losses [B, T, D]; with sites, also each later mid stream's change [B, T, D, W] per site index (E5's readings)."""

    @torch.no_grad()
    def __init__(self, model, idx, targets, chunk=128):
        m = um()
        self.model, self.targets, self.chunk = model, targets, chunk
        T = idx.shape[1]
        self.cos, self.sin = model.cos[:, :T], model.sin[:, :T]
        e = model.transformer.wte(idx)
        x = self.x0 = m.norm(e)
        ngram = arch.ngram_features(model, idx)
        self.sites, self.keys, self.values = [e], [], []
        for i, block in enumerate(model.transformer.h):
            attn, ve = block.attn, arch.value_embedding(model, idx, i, ngram)
            x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * self.x0
            h = m.norm(x)
            k = m.norm(m.apply_rotary_emb(attn.c_k(h).view(*h.shape[:2], attn.n_kv_head, attn.head_dim),
                                          self.cos, self.sin))
            v = attn.c_v(h).view(*h.shape[:2], attn.n_kv_head, attn.head_dim)
            if ve is not None:
                gate = 2 * torch.sigmoid(attn.ve_gate(h[..., :attn.ve_gate_channels]))
                v = v + gate.unsqueeze(-1) * ve.view_as(v)
            self.keys.append(k.transpose(1, 2))    # [B, H, T, d]
            self.values.append(v.transpose(1, 2))
            x = x + attention(attn, h, ve, (self.cos, self.sin), model.window_sizes[i], held=True)
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
    def __call__(self, site, delta, sites=False, start=0):
        """delta [B, C, D, W]: D moves of tokens start..start + C at the site."""
        m, model = um(), self.model
        L = model.config.n_layer
        B, C, D, W = delta.shape
        losses = torch.empty(B, C, D, device=delta.device)
        at = {k: torch.empty_like(delta) for k in range(site + 1, L + 1)} if sites else None
        for c0 in range(0, C, self.chunk):
            dl = delta[:, c0:c0 + self.chunk]
            local, sl = slice(c0, c0 + dl.shape[1]), slice(start + c0, start + c0 + dl.shape[1])
            x0 = self.x0[:, sl, None]
            if site == 0:
                x0 = m.norm(self.sites[0][:, sl, None] + dl)
                x, first = x0, 0
            else:
                mid = self.sites[site][:, sl, None] + dl
                x, first = mid + model.transformer.h[site - 1].mlp(m.norm(mid)), site
            for i in range(first, L):
                xin = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
                mid = xin + self.held_attention(i, xin, sl.start)
                if sites:
                    at[i + 1][:, local] = mid - self.sites[i + 1][:, sl, None]
                x = mid + model.transformer.h[i].mlp(m.norm(mid))
            y = self.targets[:, sl, None].expand(-1, -1, D)
            losses[:, local] = token_losses(model, x.flatten(1, 2), y.flatten(1, 2)).view(B, -1, D)
        return (losses, at) if sites else losses


class DeltaReplay(Replay):
    """Replay as the chain's kernels compute it: each probe's change carried through the network, every product of a
    change taken in `dtype` (bf16 as the chain runs them) and every nonlinear step (the RMS norms, the own query's
    rotary and norm, the softmax, ReLU^2, the soft cap and the loss) evaluated in float32 around the clean float32
    values. Replay instead reruns whole moved copies, whose bf16 rounding of full-size activations and logits swamps a
    1% probe's change (H6 linearity_check's bf16 rows). With dtype float32 the two agree to rounding. Same interface:
    losses [B, C, D] (and each later mid stream's change), keys and values held, each token's own query live."""

    def __init__(self, model, idx, targets, chunk=128, dtype=torch.bfloat16):
        super().__init__(model, idx, targets, chunk)
        self.dtype = dtype

    def lin(self, layer, d):
        """A linear map applied to a change, in the products' dtype, returned in float32."""
        return F.linear(d.to(self.dtype), layer.weight.to(self.dtype)).float()

    def query(self, attn, h, pos):
        """The own query (rotary and norm) of normed inputs h [B, C, D, W] at positions pos, in float32."""
        m = um()
        B, C, D, _ = h.shape
        q = F.linear(h, attn.c_q.weight.float()).view(B, C, D, attn.n_head, attn.head_dim)
        return self.rotate(m, attn, q, pos)

    def rotate(self, m, attn, q, pos):
        cos, sin = self.cos[:, pos].unsqueeze(2).float(), self.sin[:, pos].unsqueeze(2).float()
        half = attn.head_dim // 2
        q1, q2 = q[..., :half], q[..., half:]
        return m.norm(torch.cat([q1 * cos + q2 * sin, q1 * -sin + q2 * cos], -1))

    def held_attention_change(self, i, xin, dxin, t0):
        """Block i's attention output change for clean block inputs xin and their changes dxin [B, C, D, W]."""
        m, attn = um(), self.model.transformer.h[i].attn
        B, C, D, _ = xin.shape
        pos = torch.arange(t0, t0 + C, device=xin.device)
        h, dh = m.norm(xin), m.norm(xin + dxin) - m.norm(xin)
        q = self.query(attn, h, pos)
        dq_pre = self.lin(attn.c_q, dh).view(B, C, D, attn.n_head, attn.head_dim)
        q_pre = F.linear(h, attn.c_q.weight.float()).view(B, C, D, attn.n_head, attn.head_dim)
        dq = self.rotate(m, attn, q_pre + dq_pre, pos) - q
        k, v = self.keys[i].float(), self.values[i].float()
        T = k.shape[2]
        scores = torch.einsum('bcdhk,bhsk->bhcds', q, k) / attn.head_dim ** 0.5
        dscores = torch.einsum('bcdhk,bhsk->bhcds', dq.to(self.dtype), k.to(self.dtype)).float() / attn.head_dim ** 0.5
        s = torch.arange(T, device=xin.device)
        allowed = (s[None, :] <= pos[:, None]) & (pos[:, None] - s[None, :] <= self.model.window_sizes[i][0])
        mask = ~allowed[None, None, :, None, :]
        p = scores.masked_fill(mask, float('-inf')).softmax(-1)
        dp = (scores + dscores).masked_fill(mask, float('-inf')).softmax(-1) - p
        dout = torch.einsum('bhcds,bhsk->bcdhk', dp.to(self.dtype), v.to(self.dtype)).float()
        return self.lin(attn.c_proj, dout.reshape(B, C, D, -1))

    def mlp_change(self, block, mid, dmid):
        m = um()
        u = F.linear(m.norm(mid), block.mlp.c_fc.weight.float())
        du = self.lin(block.mlp.c_fc, m.norm(mid + dmid) - m.norm(mid))
        return self.lin(block.mlp.c_proj, F.relu(u + du).square() - F.relu(u).square())

    @torch.no_grad()
    def __call__(self, site, delta, sites=False, start=0):
        m, model = um(), self.model
        L, vocab = model.config.n_layer, model.config.vocab_size
        B, C, D, W = delta.shape
        losses = torch.empty(B, C, D, device=delta.device)
        at = {k: torch.empty_like(delta, dtype=torch.float32) for k in range(site + 1, L + 1)} if sites else None
        head = model.lm_head.weight
        for c0 in range(0, C, self.chunk):
            dl = delta[:, c0:c0 + self.chunk].float()
            n = dl.shape[1]
            local, sl = slice(c0, c0 + n), slice(start + c0, start + c0 + n)
            expand = lambda t: t[:, sl, None].float().expand(-1, -1, D, -1)
            x0 = expand(self.x0)
            if site == 0:
                e = expand(self.sites[0])
                dx0 = m.norm(e + dl) - m.norm(e)
                x, dx, first = x0, dx0, 0
            else:
                mid = expand(self.sites[site])
                block = model.transformer.h[site - 1]
                dx0 = torch.zeros_like(dl)
                x = mid + F.linear(F.relu(F.linear(m.norm(mid), block.mlp.c_fc.weight.float())).square(),
                                   block.mlp.c_proj.weight.float())
                dx, first = dl + self.mlp_change(block, mid, dl), site
            for i in range(first, L):
                block = model.transformer.h[i]
                lam_r, lam_0 = model.resid_lambdas[i].float(), model.x0_lambdas[i].float()
                xin, dxin = lam_r * x + lam_0 * x0, lam_r * dx + lam_0 * dx0
                mid = expand(self.sites[i + 1])
                dmid = dxin + self.held_attention_change(i, xin, dxin, sl.start)
                if sites:
                    at[i + 1][:, local] = dmid
                x = mid + F.linear(F.relu(F.linear(m.norm(mid), block.mlp.c_fc.weight.float())).square(),
                                   block.mlp.c_proj.weight.float())
                dx = dmid + self.mlp_change(block, mid, dmid)
            z = F.linear(m.norm(x), head.float())[..., :vocab]
            dz = F.linear((m.norm(x + dx) - m.norm(x)).to(self.dtype), head.to(self.dtype)).float()[..., :vocab]
            capped = SOFTCAP * torch.tanh((z + dz) / SOFTCAP)
            y = self.targets[:, sl, None].expand(-1, -1, D)
            losses[:, local] = F.cross_entropy(capped.flatten(0, 2), y.flatten(), ignore_index=-1,
                                               reduction='none').view(B, n, D)
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
                    # E7's prior at the mid sites only: the chain's (cut.py) S0 keeps the plain estimate
                    p = guess[k].float() if self.prior == 'identity' and k > 0 else None
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
    """E10 (b): a sublayer's block-local gradients with each nn.Linear weight's entries kept only where at least AGREE
    of the tokens agree in sign. G (the credit at the layer's output, from this sublayer's credit alone) and X (its
    input) give the weight gradient G^T X; S = sign(G)^T sign(X) is each entry's net votes and
    N = [G != 0]^T [X != 0] its nonzero contributions, so (N + |S|) / 2 tokens back the majority sign, and an entry is
    kept if that is at least AGREE x tokens. A zero contribution (a unit off after the ReLU^2, an exact zero credit)
    backs neither sign, so it counts against agreement. Other parameters (the value-embedding table) keep their
    gradients. Appends each weight's kept share to `kept`."""
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    weights = {id(m.weight) for m in linears}
    others = [p for p in params if id(p) not in weights]
    grads = torch.autograd.grad(out, [io[m][1] for m in linears] + others, grad_outputs=credit.to(out.dtype),
                                retain_graph=True)
    found = dict(zip(map(id, others), grads[len(linears):]))
    with torch.autocast(out.device.type, enabled=False):  # float32: the vote counts must be exact
        for m, G in zip(linears, grads):
            G2, X2 = G.flatten(0, -2).float(), io[m][0].flatten(0, -2).float()
            votes, nonzero = G2.sign().T @ X2.sign(), (G2 != 0).float().T @ (X2 != 0).float()
            keep = (nonzero + votes.abs()) / 2 >= AGREE * tokens
            found[id(m.weight)] = ((G2.T @ X2) * keep).to(m.weight.dtype)
            kept.append(keep.float().mean())
    return found


def sequence_gradients(module, params, out, credit, io, each):
    """E8's H8 setting: a sublayer's block-local gradients, each nn.Linear weight's from G_b^T X_b per sequence b (G
    its output's credit from this sublayer's credit alone, X its input), summed over b for the step and kept per
    sequence in `each` [B, out, in] for the SNR shaper. Other parameters (the value-embedding table) as autograd's."""
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    weights = {id(m.weight) for m in linears}
    others = [p for p in params if id(p) not in weights]
    grads = torch.autograd.grad(out, [io[m][1] for m in linears] + others, grad_outputs=credit.to(out.dtype),
                                retain_graph=True)
    found = dict(zip(map(id, others), grads[len(linears):]))
    with torch.autocast(out.device.type, enabled=False):
        for m, G in zip(linears, grads):
            per = torch.einsum('bto,bti->boi', G.float(), io[m][0].float())
            each[id(m.weight)] = per
            found[id(m.weight)] = per.sum(0).to(m.weight.dtype)
    return found


class SequenceGrads:
    """The per-sequence gradients synthesis_gradients kept, by parameter name: SNRScaling's view of H2."""

    def __init__(self, model, each):
        self.by_name = {n: each[id(p)] for n, p in model.named_parameters() if id(p) in each}

    def stack(self, name):
        return self.by_name[name]


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


class Untimed:
    """Logging work inside an update (E6's cosine and stable ranks, E14's gradient cosine), timed apart so that
    loud/train.py leaves it out of the clock, as it does the paused work between updates."""

    def __init__(self, device):
        self.device, self.seconds = device, 0.0

    @contextmanager
    def __call__(self):
        sync = self.device.type == 'cuda'
        if sync:
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if sync:
                torch.cuda.synchronize()
            self.seconds += time.perf_counter() - start

    def take(self):
        seconds, self.seconds = self.seconds, 0.0
        return seconds


def synthesis_gradients(model, idx, targets, scheme, step, freeze_queries=False, compare=False, agree=False,
                        runway=False, tiers=None, log=False, each=None, untimed=nullcontext):
    """Synthesis mode's parameter gradients (module docstring), the model's token losses and metrics fields: with
    `compare`, the cosine of the summed block-local weight gradient to the identity's (E6's log); E10 (b) `agree`
    (the share of entries kept logged when `log`); E11 `runway` (its distribution logged when `log`); E13 `tiers`, a
    function of the token losses giving the low tier (its share logged); E8 `each`, a dict that receives every block
    weight's per-sequence gradient (sequence_gradients)."""
    L = model.config.n_layer
    io = {}
    with capture_linears(model, io) if agree or each is not None else nullcontext():
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

    def block_local(credit, masked, kept_each=None):
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
                elif kept_each is not None:
                    found.update(sequence_gradients(module, params, out, g, io, kept_each))
                else:
                    found.update(zip(map(id, params), torch.autograd.grad(out, params, grad_outputs=g.to(out.dtype),
                                                                          retain_graph=True)))
        found[id(model.transformer.wte.weight)] = torch.autograd.grad(
            sites[0], model.transformer.wte.weight, grad_outputs=credit[0].to(sites[0].dtype), retain_graph=True)[0]
        return found

    grads = block_local(shaped, agree, each)
    grads.update(zip(map(id, shared), (g_head, g_lr, g_l0)))
    if agree and log:
        fields['agree_kept'] = round(torch.stack(kept).mean().item(), 6)  # entries kept, mean over block weights
    if compare and (not scheme.identity or agree or runway):
        with untimed():
            exact = block_local(credits, False)
            keys = list(exact)  # block-local only: the head and scalars are the cut graph's in both
            with torch.autocast(idx.device.type, enabled=False):  # float32: under bf16 the dot product exceeds 1
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
        global CLIP, AGREE, RUNWAY  # the rules' constants for this run (v2's unless the config sets them)
        CLIP, AGREE = o['clip_factor'], o['agree_share']
        RUNWAY = tuple(o['runway_grid']) if o['runway_grid'] else RUNWAY_V2
        self.model_file = arch.model_file() if o['arch'] == 'ngram' else None  # what the official scorer builds
        self.untimed = Untimed(device)  # loud/train.py takes it off the clock after each update
        self.probe = heldout[:PROBE_ROWS].to(device) if len(heldout) else None  # E14/E15's untimed logs
        self.settings = s = record_settings(m, o['preset'], o['batch'])
        s.lr, s.anvil_lr = s.lr * o['lr_mult'], s.anvil_lr * o['lr_mult']  # E19's rate scale (1: the record's)
        if not o['boost']:
            s.switch = 0.0  # Cut from the first update
        self.schedule = m.stage_schedule(steps * o['batch'], s)
        self.horizons = {phase: horizon for phase, _, _, _, horizon in self.schedule}
        self.reset()

    def reset(self, phase=None):
        m, s = self.m, self.settings
        self.phase = phase or self.schedule[0][0]
        if self.o['arch'] == 'ngram':  # E16: track-3 record #5's design at this preset's shape
            config = arch.config(s.model.n_layer, s.model.n_embd, self.o['ngram_rows'])
            self.model = arch.make_model(self.device, self.seed, config, s.projection_std, s.head_std)
        else:
            self.model = m.make_model(self.device, self.seed, s.model, s.projection_std, s.head_std)
        self.model.requires_grad_(True)
        self.params = list(self.model.parameters())
        # E8: one shaper for the run, acting in the Cut phase (every arm, control included, without the lane gain)
        self.shaper = SNRScaling(self.o['snr_scaling'], self.model) if self.o['snr_scaling'] != 'none' else None
        self.optimizer = self.make_optimizer(self.phase, self.horizons[self.phase])
        self.labels = rate_labels(self.optimizer, s.lr)
        self.scheme = Scheme(self.o, s.model.n_embd, self.device, self.seed)
        self.key_generator = torch.Generator(self.device).manual_seed(self.seed + KEY_SEED)
        self.ngram = NGram(s.model.vocab_size) if self.o['budget_theta'] else None  # E13: the stream so far
        self.reset_latent()

    def make_optimizer(self, phase, horizon):
        """PR #40's optimizers; with E8 on, ANVIL II's Cut-phase stack is ShapedBatchedAnvil with the SNR shaper
        (Boost keeps the record's, since partial-row noise exists only in the Cut phase)."""
        if self.shaper is not None and phase == 'cut':
            return shaped_optimizer(self.m, self.model, self.settings, horizon, self.shaper, 'BatchedAnvil')
        return arch.embedding_rates(self.m.make_optimizer(self.model, self.settings, horizon), self.model)

    def reset_latent(self):
        """E14's dynamics model, E15's predictors, the EMA teacher either reads (H4), and one Adam for the two at the
        record's base Adam rate and betas, following its schedule."""
        o, s, width = self.o, self.settings, self.settings.model.n_embd
        self.dynamics = latent.Dynamics(width).to(self.device) if o['nextlat'] else None
        self.targets = (latent.LatentTargets(s.model.n_layer, width, o['latent_boost'] == 'plus', o['latent_lambda'])
                        .to(self.device) if o['latent_boost'] != 'none' else None)
        modules = [mod for mod in (self.dynamics, self.targets) if mod is not None]
        self.aux_params = [p for mod in modules for p in mod.parameters()]
        self.aux_optimizer = (torch.optim.Adam(self.aux_params, lr=s.lr, betas=s.betas, eps=1e-8)
                              if self.aux_params else None)
        teacher = (o['nextlat'] and o['nextlat_target'] == 'ema') or self.targets is not None
        self.teacher = EMAWeights(self.model, self.steps) if teacher else None

    def warmup(self, batch):
        """Every phase's code path once on random tokens, from fresh models (loud/train.py then resets)."""
        for phase in self.horizons:
            self.reset(phase)
            self.update(next(i for i, b in enumerate(self.schedule, 1) if b[0] == phase), batch)

    def key_rows(self, sequences):
        """E17's key draws: for each block and sequence, key_draws rows of the Hadamard basis chosen at random with
        random column signs, fresh each update, as the record draws them (key_input_credits, each sequence from its
        own generator). key_assign 'shared' (the record's): every token of the sequence reads the same rows; 'token':
        the sequence's basis, permuted, split into K = width / m blocks of m rows, token t reading block t mod K, so
        neighbouring tokens' errors are uncorrelated. None: exact."""
        m = self.o['key_draws']
        if not m:
            return None
        W, H, out = self.settings.model.n_embd, self.scheme.hadamard, []
        for _ in range(self.settings.model.n_layer):
            per = []
            for _ in range(sequences):
                order = torch.randperm(W, device=self.device, generator=self.key_generator)
                signs = torch.randint(0, 2, (W,), device=self.device, generator=self.key_generator).float() * 2 - 1
                per.append((H[order] * signs).view(W // m, m, W) if self.o['key_assign'] == 'token'
                           else H[order[:m]] * signs)
            out.append(torch.stack(per))
        return out

    def describe(self):
        return dict(name='h8', file=refs.TRAINERS['pr40'], settings={k: v for k, v in asdict(self.settings).items()
                                                                       if k != 'model'},
                    model=asdict(self.settings.model), schedule=dict(phases=self.horizons, steps=len(self.schedule)))

    def update(self, step, batch):
        m, o = self.m, self.o
        phase, phase_step, _, _, horizon = self.schedule[step - 1]
        if phase != self.phase:
            optimizer = self.make_optimizer(phase, horizon)
            m.handoff_optimizer(self.optimizer, optimizer, phase_step)
            self.optimizer, self.phase = optimizer, phase
            self.labels = rate_labels(optimizer, self.settings.lr)
        m.set_rate(self.optimizer, phase_step, horizon, self.settings.warmdown, self.settings.final)
        fields = dict(current_rates(self.optimizer, self.labels), phase=phase)
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        compare = bool(o['compare_every']) and step % o['compare_every'] == 0
        nextlat = bool(o['nextlat']) and (o['nextlat_onset'] == 'start' or phase == 'cut')  # E14 acts this update
        side = self.targets if phase == 'boost' else None                                 # E15 acts this update
        aux_params = self.aux_params if nextlat or side is not None else []
        tiers = None
        if self.ngram is not None:  # E13: score the batch under the batches before it, then add it (every phase)
            rows = batch.cpu().numpy()
            if phase != 'boost':
                ngram = torch.from_numpy(self.ngram.losses(rows)).float().to(self.device)
                tiers = lambda losses: low_tier(ngram, losses, o['budget_theta'])
            self.ngram.insert(rows)
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            teacher = None
            if self.teacher is not None and ((nextlat and o['nextlat_target'] == 'ema') or side is not None):
                with torch.no_grad():
                    teacher = um().norm(cut_forward(self.teacher.model, x, None, held=False)[2])
            if side is not None:
                side.target, side.logging, side.seen = teacher, step % LATENT_LOG_EVERY == 0, {}
            if phase == 'boost':
                keep = {}
                loss, losses = boost_loss(self.model, x, y, side, keep)
                top, embedding = keep['top'], keep['embedding']
            elif o['mode'] == 'cut':
                losses, sites, top, _, _ = cut_forward(self.model, x, y, True, o['freeze_queries'], o['live_paths'],
                                                       o['live_blocks'] or None, self.key_rows(len(x)))
                loss, embedding = losses.sum() / (y != -1).sum(), sites[0]
            else:
                each = {} if self.shaper is not None else None
                grads, losses, extra = synthesis_gradients(self.model, x, y, self.scheme, step, o['freeze_queries'],
                                                           compare, o['agree'], o['runway'], tiers,
                                                           log=step % RUNWAY_EVERY == 0, each=each,
                                                           untimed=self.untimed)
                fields.update(extra)
                if each is not None:
                    self.shaper.before_step(step, SequenceGrads(self.model, each))
            if phase == 'boost' or o['mode'] == 'cut':
                if nextlat:
                    h = um().norm(top)
                    aux = o['nextlat'] * latent.nextlat_loss(self.dynamics, h, embedding, x,
                                                             h.detach() if teacher is None else teacher)
                    if step % AUX_LOG_EVERY == 0:
                        with self.untimed():
                            fields['nextlat'] = dict(loss=aux.item() / o['nextlat'], top_cosine=gradient_cosine(
                                loss, aux, top))
                    loss = loss + aux
                found = torch.autograd.grad(loss, self.params + aux_params, allow_unused=True)
                grads = dict(zip(map(id, self.params + aux_params), found))
        if side is not None and side.logging:
            fields['latent_boost'] = {k: {name: round(v.item(), 5) for name, v in kinds.items()}
                                      for k, kinds in sorted(side.seen.items())}
        for p in self.params + aux_params:
            g = grads.get(id(p))
            p.grad = torch.zeros_like(p) if g is None else g.to(p.dtype)
        if o['rank_every'] and step % o['rank_every'] == 0:
            with self.untimed():
                fields['stable_rank'] = stable_ranks(self.model, self.optimizer)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.shaper is not None and phase == 'cut':
            fields.update(self.shaper.report(step))
            self.shaper.pergrad = None  # the batch's gradients, freed
        if aux_params:  # only while its modules act: Adam's momentum would move them on zero gradients
            group = self.optimizer.param_groups[0]
            for aux_group in self.aux_optimizer.param_groups:  # the record's schedule on the base rate
                aux_group['lr'] = self.settings.lr * group['lr'] / group['initial_lr']
            self.aux_optimizer.step()
            self.aux_optimizer.zero_grad(set_to_none=True)
        if self.teacher is not None and (o['nextlat'] or phase == 'boost'):  # while anything still reads it
            self.teacher.update(step)
        return losses.detach().float()[y != -1].mean(), fields

    def diagnose(self, step):
        """Untimed, on PROBE_ROWS held-out sequences: E14's effective ranks of the top-layer latents and the site
        credits at 20% and at the end; E15's cosine of the cut graph's gradient to the full one at the handoff."""
        record = {}
        if self.probe is None:
            return record
        x, y = self.probe[:, :-1].contiguous(), self.probe[:, 1:].contiguous()
        if step in (round(0.2 * self.steps), self.steps):  # every run, so E14's arms have their baseline's
            record['effective_rank'] = effective_ranks(self.model, x, y, self.device)
        handoff = next((i for i, b in enumerate(self.schedule, 1) if b[0] == 'cut'), None)
        # every run at the Boost -> Cut handoff (E15's arms and e1.h8_cut), and a run without Boost at the update where
        # the record hands off (e1.h8_cut_noboost: E15's no-Boost reference)
        at = None if handoff is None else handoff - 1 if handoff > 1 else round(self.m.Settings().switch * self.steps)
        if at and step == at:
            record['handoff_cut_cosine'] = cut_cosine(self.model, x, y, self.device)
        return record


def gradient_cosine(a, b, at):
    """The cosine of two losses' gradients at a tensor of their graph (E14: the next-token and the latent loss at the
    top stream)."""
    ga, gb = (torch.autograd.grad(loss, at, retain_graph=True)[0].float().flatten() for loss in (a, b))
    return round((ga @ gb / (ga.norm() * gb.norm()).clamp_min(1e-30)).item(), 5)


def effective_ranks(model, x, y, device):
    """E14's log: the effective rank of the top-layer latents [B * T, W] and of each site's credits."""
    with torch.autocast(device.type, dtype=torch.bfloat16):
        losses, sites, top, _, _ = cut_forward(model, x, y)
        credits = torch.autograd.grad(losses.sum(), sites)
    ranks = dict(top=latent.effective_rank(um().norm(top.detach()).flatten(0, 1)))
    ranks.update({f'S{k}': latent.effective_rank(c.flatten(0, 1)) for k, c in enumerate(credits)})
    return {k: round(v, 2) for k, v in ranks.items()}


def cut_cosine(model, x, y, device):
    """E15's log: the cosine of the cut graph's weight gradient to the full graph's (backprop's) on the same rows."""
    params = [p for p in model.parameters() if p.requires_grad]
    flat = []
    for held in (True, False):
        with torch.autocast(device.type, dtype=torch.bfloat16):
            losses = cut_forward(model, x, y, held)[0]
            grads = torch.autograd.grad(losses.sum(), params, allow_unused=True)
        flat.append(torch.cat([(torch.zeros_like(p) if g is None else g).float().flatten()
                               for p, g in zip(params, grads)]))
    return round((flat[0] @ flat[1] / (flat[0].norm() * flat[1].norm()).clamp_min(1e-30)).item(), 5)


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
