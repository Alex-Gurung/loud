"""E9 (proxy test): a damped true-Fisher left factor in place of ANVIL II's orthogonalization on the block matrices.

For each block matrix W (rows = outputs), S is an exponential mean (FISHER_DECAY per update) of E_t[d_t d_t^T], where
d_t is the gradient of the summed loss with respect to W's output at token t when the labels are sampled from the
model's own softmax: the true Fisher's output-side Kronecker factor (K-FAC's G). Each sequence's training forward
feeds one extra backward for these labels, to the layer outputs only (no weight gradients). Every FISHER_EVERY updates
(from update 1), R = (S + fisher_damping * mean eig(S) * I)^(-1/2) by eigh. ANVIL II's polar(pending) becomes
R @ pending, scaled to the Frobenius norm polar(pending) has (about sqrt(min(rows, cols)), less where the pending update
has small singular values), so the step's size is ANVIL II's and only its direction changes; the
rails, per-lane gain, decay and rates are unchanged. fisher_damping is the one knob: large values flatten R towards
the identity (the raw pending update), small ones whiten the output side fully.

The labels come from a generator seeded by the job's seed. Cost per sequence: one backward to the outputs (about a
third of a training step) and the d^T d products (about 180 GFLOP for the 38M model); every 10 updates, 48 eighs.
The plan's kernel version (the curvature term and sampled-label loss from the Cut chain's probes) waits on this test:
if it fails, that work is skipped.
"""
import torch
import torch.nn.functional as F

from .anvil import is_anvil_matrix

FISHER_DECAY = 0.95   # exponential mean of S per update, as K-FAC's running factors
FISHER_EVERY = 10     # updates between inverse roots
SAMPLE_SEED = 7919    # offset of the label generator's seed from the job's


class FisherFactor:
    """The E9 shaper of ShapedAnvil (loud/anvil.py), and the hooks that measure S on the training forwards."""

    def __init__(self, o, model, seed, polar):
        self.damping, self.polar = o['fisher_damping'], polar
        self.model, self.vocab = model, model.config.vocab_size
        named = [(n, p) for n, p in model.named_parameters() if is_anvil_matrix(n, p)]
        self.index = {id(p): i for i, (_, p) in enumerate(named)}
        modules = dict(model.named_modules())
        self.modules = [modules[n.removesuffix('.weight')] for n, _ in named] + [model.lm_head]
        self.outputs = [None] * len(self.modules)
        self.hooks = [module.register_forward_hook(self.keep(i)) for i, module in enumerate(self.modules)]
        self.batch_sum = [torch.zeros(p.size(0), p.size(0), device=p.device) for _, p in named]
        self.tokens, self.S, self.R = None, None, None  # tokens: a device tensor, so counting never syncs
        self.generator = torch.Generator(next(model.parameters()).device).manual_seed(seed + SAMPLE_SEED)

    def keep(self, i):
        def hook(module, args, out):
            if out.requires_grad:  # a training forward (not the held-out scorer's)
                self.outputs[i] = out
        return hook

    def observe(self, targets):
        """After a training forward, before its backward: add d^T d for labels sampled from the model's softmax."""
        *outputs, head = self.outputs
        logits = 15 * torch.tanh(head[..., :self.vocab].float() / 15)  # utils/model.py's soft cap, on the same graph
        with torch.no_grad():
            p = logits.detach().softmax(-1).flatten(0, 1)
            sampled = torch.multinomial(p, 1, generator=self.generator).view_as(targets)
            sampled = sampled.masked_fill(targets == -1, -1)
        loss = F.cross_entropy(logits.flatten(0, 1), sampled.flatten(), ignore_index=-1, reduction='sum')
        deltas = torch.autograd.grad(loss, outputs, retain_graph=True)
        for total, d in zip(self.batch_sum, deltas):
            d = d.flatten(0, 1).float()
            total.addmm_(d.T, d)
        count = (targets != -1).sum()
        self.tokens = count if self.tokens is None else self.tokens + count
        self.outputs = [None] * len(self.modules)

    def before_step(self, step, pergrad=None):
        """Fold this update's sums into S; on the refresh updates, recompute R."""
        batch = [total / self.tokens for total in self.batch_sum]
        self.S = batch if self.S is None else [s.lerp_(b, 1 - FISHER_DECAY) for s, b in zip(self.S, batch)]
        for total in self.batch_sum:
            total.zero_()
        self.tokens = None
        if (step - 1) % FISHER_EVERY == 0:
            self.R = [inverse_root(s, self.damping) for s in self.S]

    def direction(self, p, pending):
        update = self.R[self.index[id(p)]] @ pending
        return update * (self.polar(pending).norm() / update.norm().clamp_min(1e-30))

    def after_step(self, model, batch, pergrad=None):
        pass

    def report(self, step):
        return {}

    def close(self):
        for hook in self.hooks:
            hook.remove()


def inverse_root(S, damping):
    """(S + damping * mean eig(S) * I)^(-1/2) for a symmetric positive semi-definite S."""
    eig, Q = torch.linalg.eigh(S)
    eig = eig.clamp_min(0)
    return (Q * (eig + damping * eig.mean()).clamp_min(1e-30).rsqrt()) @ Q.T
