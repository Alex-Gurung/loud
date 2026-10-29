"""E9 (v2's proxy test, P3): each block matrix's momentum left-preconditioned by a damped true-Fisher factor before
ANVIL II's orthogonalization.

For each block matrix W (rows = outputs), F is an exponential mean (FISHER_DECAY per update) of E_t[d_t d_t^T], where
d_t is the gradient of the summed loss with respect to W's output at token t when the labels are sampled from the
model's own softmax: the true Fisher's output-side Kronecker factor (K-FAC's G). Each sequence's training forward
feeds one extra backward for these labels, to the layer outputs only (no weight gradients). Every FISHER_EVERY updates
(from update 1), R = (F + delta I)^(-1/2) by eigh, with delta fixed at DAMPING x tr(F) / rows (v2: 1e-3 tr(F) / 512,
which is the same for the 512-row matrices; c_fc's F has 2,048 rows). ANVIL II then orthogonalizes R @ pending
instead of the pending update, and everything after (polar, the per-lane gain, the decay, the rates) is unchanged.

The labels come from a generator seeded by the job's seed. Cost per sequence: one backward to the outputs (about a
third of a training step) and the d^T d products (about 180 GFLOP for the 38M model); every 10 updates, 48 eighs.
The kernel version in the Cut port (the curvature term and the sampled-label loss from the chain's probes) is built
only if this test shows per-direction scaling matters.
"""
import torch
import torch.nn.functional as F

from .anvil import is_anvil_matrix

DAMPING = 1e-3        # delta = DAMPING x the mean eigenvalue of F (v2's fixed 1e-3 tr(F) / 512)
FISHER_DECAY = 0.95   # exponential mean of F per update, as K-FAC's running factors
FISHER_EVERY = 10     # updates between inverse roots
SAMPLE_SEED = 7919    # offset of the label generator's seed from the job's


class FisherFactor:
    """The E9 shaper of ShapedAnvil (loud/anvil.py), and the hooks that measure F on the training forwards."""
    lane_gain = True  # ANVIL II after the orthogonalization is unchanged

    def __init__(self, model, seed, polar):
        self.polar = polar
        self.model, self.vocab = model, model.config.vocab_size
        named = [(n, p) for n, p in model.named_parameters() if is_anvil_matrix(n, p)]
        self.index = {id(p): i for i, (_, p) in enumerate(named)}
        modules = dict(model.named_modules())
        self.modules = [modules[n.removesuffix('.weight')] for n, _ in named] + [model.lm_head]
        self.outputs = [None] * len(self.modules)
        self.hooks = [module.register_forward_hook(self.keep(i)) for i, module in enumerate(self.modules)]
        self.batch_sum = [torch.zeros(p.size(0), p.size(0), device=p.device) for _, p in named]
        self.tokens, self.F, self.R = None, None, None  # tokens: a device tensor, so counting never syncs
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
        """Fold this update's sums into F; on the refresh updates, recompute R."""
        batch = [total / self.tokens for total in self.batch_sum]
        self.F = batch if self.F is None else [f.lerp_(b, 1 - FISHER_DECAY) for f, b in zip(self.F, batch)]
        for total in self.batch_sum:
            total.zero_()
        self.tokens = None
        if (step - 1) % FISHER_EVERY == 0:
            self.R = [inverse_root(fisher) for fisher in self.F]

    def direction(self, p, pending, before=None):
        return self.polar(self.R[self.index[id(p)]] @ pending)

    def after_step(self, model, batch, pergrad=None):
        pass

    def report(self, step):
        return {}

    def close(self):
        for hook in self.hooks:
            hook.remove()


def inverse_root(fisher):
    """(F + delta I)^(-1/2) for a symmetric positive semi-definite F, delta = DAMPING x its mean eigenvalue."""
    eig, Q = torch.linalg.eigh(fisher)
    eig = eig.clamp_min(0)
    return (Q * (eig + DAMPING * eig.mean()).clamp_min(1e-30).rsqrt()) @ Q.T
