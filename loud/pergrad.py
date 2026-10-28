"""H2: per-sequence gradients. Each sequence's gradient is kept separately (bfloat16) before averaging into the update,
then discarded once the update's consumers have read it. Needs one sequence per micro-batch.

Consumers read `stack(name)` -> [B, *shape] during the update. The built-in summary, logged every `every` updates,
is per matrix type: tr(Sigma) = sum_i |g_i - g|^2 / (B - 1), the across-sequence spread, and |g|^2 for the batch mean
g. |g|^2 overestimates the true |G|^2 by tr(Sigma) / B, so the simple noise scale B_simple = tr(Sigma) / |G|^2
(McCandlish et al. 2018, the quantity E8's direction statistics refine) uses the unbiased |G|^2 = |g|^2 - tr(Sigma) / B.
One update's estimate is noisy (and can be negative); average the two sums over updates before taking the ratio.
"""
import re

import torch

TYPES = ('attn.c_q', 'attn.c_k', 'attn.c_v', 'attn.c_proj', 'attn.ve_gate', 'mlp.c_fc', 'mlp.c_proj')


def matrix_type(name):
    for t in TYPES:
        if name.endswith(t + '.weight'):
            return t
    if name.startswith('value_embeds.'):
        return 'value_embeds'
    return re.sub(r'\.weight$', '', name)


class PerSequenceGrads:
    """Every matrix's per-sequence gradients, in bfloat16."""

    def __init__(self, named_params):
        self.names = [n for n, p in named_params if p.ndim == 2]
        self.store = {n: [] for n in self.names}

    def capture(self, model):
        """Call after one sequence's backward, before its gradients are accumulated and cleared."""
        params = dict(model.named_parameters())
        for n in self.names:
            g = params[n].grad
            self.store[n].append(torch.zeros_like(params[n], dtype=torch.bfloat16) if g is None else g.bfloat16())

    def stack(self, name):
        return torch.stack(self.store[name])

    def __len__(self):
        return len(next(iter(self.store.values()), []))

    def clear(self):
        for n in self.names:
            self.store[n].clear()

    @torch.no_grad()
    def summary(self):
        """Per matrix type: trace_cov, mean_sq (|g|^2), true_sq (unbiased |G|^2) and b_simple = trace_cov / true_sq."""
        batch = len(self)
        if batch < 2:
            return {}
        sums = {}
        for n in self.names:
            g = self.stack(n).float()
            mean = g.mean(0)
            row = sums.setdefault(matrix_type(n), torch.zeros(2, device=g.device))
            row[0] += (g - mean).square().sum() / (batch - 1)
            row[1] += mean.square().sum()
        out = {}
        for kind, (trace_cov, mean_sq) in ((k, v.tolist()) for k, v in sums.items()):  # one sync per matrix type
            true_sq = mean_sq - trace_cov / batch
            out[kind] = dict(trace_cov=trace_cov, mean_sq=mean_sq, true_sq=true_sq,
                             b_simple=trace_cov / true_sq if true_sq > 0 else None)
        return out
