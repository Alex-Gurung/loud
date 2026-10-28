"""E8 (proxy): steps set by curvature-to-noise along the pending update's top directions.

Every `damp_every` updates (a refresh, starting at update 1), the top `damp_directions` singular directions
u = a b^T (unit Frobenius norm) of each block matrix's pending update (what ANVIL II orthogonalizes) replace the
tracked ones, and right after that update:
  h, the curvature along u: a central difference of the gradient of the mean loss, <u, g(W + e u) - g(W - e u)> / 2e,
     e = 1e-3 * max(|W|, 1), in float32 at full matmul precision on the first CURVATURE_ROWS sequences of the batch
     (flex attention has no double backward; H6's direction_stats measures it the same way).
c, the noise along u, is the across-sequence variance of the per-sequence gradients' projections <g_i, u> (H2),
averaged over every update since the refresh (the variance within each update, so the drift of the mean between
updates is not counted as noise).

Each update then scales its component along every tracked u by
    f = 1 / (1 + c / (B h^2 lambda)),    lambda = curvature_damping * delta^2
with B the batch and delta = ANVIL II's base rate * sqrt(max(1, rows / cols)), the length of its step along a unit
direction. c / (B h^2) is the variance of the Newton step's noise along u (the noisy quadratic model's ratio), so
curvature_damping, the one knob, is that variance, in squared optimizer steps, at which a direction's step is halved.
Other directions are left alone.

Where the factors act (damp_where): the plan says before the orthogonalization ('pending'). But polar() maps every
singular value of the pending update to about 1, so scaling the pending update along its own top singular directions
leaves them at about 1 and slightly enlarges the rest, through polar's Frobenius normalization (the method check
curvature_damping shows it). The default, 'step', applies the factors right after polar(), before the per-lane gain:
the step along u is f times what it would have been. 'pending' is kept as the plan's literal control.

Cost per refresh at the defaults (16 directions, 48 matrices): 1,536 forward-backward passes on 2 sequences in float32
(about 30 s on an H100), 48 SVDs, and a float32 copy of the model; every update, the projections of the per-sequence
gradients (a few GFLOP).
"""
import copy

import torch

from .anvil import is_anvil_matrix
from .diagnostics import exact
from .pergrad import matrix_type

CURVATURE_ROWS = 2   # sequences of the refresh update's batch that the curvature is measured on
STEP = 1e-3          # central-difference step, relative to the matrix's norm (as direction_stats)


def damp(x, a, b, f):
    """x with its component along each a_k b_k^T (orthonormal columns a, b) scaled by f_k."""
    coefficients = ((a.T @ x) * b.T).sum(1)
    return x - (a * ((1 - f) * coefficients)) @ b.T


class CurvatureDamping:
    """The E8 shaper of ShapedAnvil (loud/anvil.py) and the hooks Backprop.update calls around the optimizer step."""

    def __init__(self, o, model, anvil_lr, polar):
        self.polar = polar
        self.k, self.every, self.where, self.batch = o['damp_directions'], o['damp_every'], o['damp_where'], o['batch']
        self.names = [n for n, p in model.named_parameters() if is_anvil_matrix(n, p)]
        params = dict(model.named_parameters())
        self.index = {id(params[n]): i for i, n in enumerate(self.names)}
        self.scale = [o['curvature_damping'] * anvil_lr ** 2 * max(1, params[n].size(0) / params[n].size(1))
                      for n in self.names]
        self.left = self.right = self.h = self.factors = None  # per matrix: [rows, k], [cols, k], [k], [k]
        self.noise, self.samples = None, 0   # per matrix: summed within-update variances [k], over `samples` updates
        self.recorded, self.refreshing, self.summary = {}, False, None

    def before_step(self, step, pergrad):
        """After the update's gradient: add its per-sequence noise along the tracked directions, then set the
        factors this update's step uses."""
        self.refreshing = (step - 1) % self.every == 0
        if self.left is None:
            return
        self.observe(pergrad)
        self.factors = [self.factor(i, c / self.samples) for i, c in enumerate(self.noise)]

    def observe(self, pergrad):
        for i, name in enumerate(self.names):
            g = pergrad.stack(name).float()                                       # [B, rows, cols]
            projections = torch.einsum('brc,rk,ck->bk', g, self.left[i], self.right[i])
            self.noise[i] += projections.var(0)
        self.samples += 1

    def direction(self, p, pending):
        """ShapedAnvil's orthogonalization: polar(), with the factors before or after it; on a refresh update, also
        record the pending update's top directions."""
        i = self.index[id(p)]
        if self.refreshing:
            u, _, vh = torch.linalg.svd(pending, full_matrices=False)
            self.recorded[i] = (u[:, :self.k].contiguous(), vh[:self.k].T.contiguous())
        damped = self.factors is not None
        if damped and self.where == 'pending':
            pending = damp(pending, self.left[i], self.right[i], self.factors[i])
        update = self.polar(pending)
        if damped and self.where == 'step':
            update = damp(update, self.left[i], self.right[i], self.factors[i])
        return update

    def after_step(self, model, batch, pergrad):
        """On a refresh update: the recorded directions become the tracked ones, their curvature is measured at the
        new weights, and their noise starts from this update's per-sequence gradients."""
        if not self.refreshing:
            return
        previous = self.left
        self.left = [self.recorded[i][0] for i in range(len(self.names))]
        self.right = [self.recorded[i][1] for i in range(len(self.names))]
        self.recorded.clear()
        self.h = self.curvature(model, batch[:CURVATURE_ROWS])
        self.noise, self.samples = [torch.zeros_like(h) for h in self.h], 0
        self.observe(pergrad)
        self.summary = self.describe(previous)

    def curvature(self, model, rows):
        """h along every tracked direction (module docstring), on a float32 copy of the model."""
        ref = copy.deepcopy(model).float().requires_grad_(False)
        params = dict(ref.named_parameters())
        x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
        out = []
        with exact():
            for name, a, b in zip(self.names, self.left, self.right):
                p = params[name]
                w = p.detach().clone()
                e = STEP * max(w.norm().item(), 1.0)
                p.requires_grad_(True)
                h = torch.empty(a.shape[1], device=w.device)
                for k in range(a.shape[1]):
                    d = torch.outer(a[:, k], b[:, k])
                    sides = []
                    for sign in (1, -1):
                        with torch.no_grad():
                            p.copy_(w + sign * e * d)
                        sides.append(torch.autograd.grad(ref(x, y), p)[0])
                    h[k] = a[:, k] @ (sides[0] - sides[1]) @ b[:, k] / (2 * e)
                p.requires_grad_(False)
                with torch.no_grad():
                    p.copy_(w)
                out.append(h)
        return out

    def describe(self, previous):
        """Per matrix type, medians over its matrices and directions: h, c (from the refresh update alone), the noisy
        quadratic model's h^2 / c and the factor at that c; and the overlap of the new directions' left
        subspace with the previous one (1 = unchanged), which says whether the refresh keeps up."""
        rows = {}
        for i, name in enumerate(self.names):
            h, c = self.h[i], self.noise[i] / self.samples
            values = dict(h=h, c=c, h2_over_c=h.square() / c.clamp_min(1e-30), factor=self.factor(i, c))
            if previous is not None:
                values['overlap'] = (previous[i].T @ self.left[i]).square().sum().reshape(1) / self.k
            for key, value in values.items():
                rows.setdefault(matrix_type(name), {}).setdefault(key, []).append(value)
        return {kind: {key: round(torch.cat(v).median().item(), 6) for key, v in row.items()}
                for kind, row in rows.items()}

    def factor(self, i, c):
        """1 / (1 + c / (B h^2 lambda)) for matrix i's tracked directions, given their noise c."""
        return 1 / (1 + c / (self.batch * self.h[i].square().clamp_min(1e-30) * self.scale[i]))

    def report(self, step):
        summary, self.summary = self.summary, None
        return {} if summary is None else dict(damping=summary)
