"""ANVIL II with its orthogonalization as a hook, for E8 and E9: boost-cut's Anvil (the backprop proxy's optimizer for
block matrices) and PR #40's BatchedAnvil (the Cut chain's).

Anvil.step forms each matrix's pending update, g.lerp(w * fast + (1 - w) * slow, beta), and orthogonalizes it with
polar() before the per-lane gain, the decay and the step. ShapedAnvil.step is that code with one change: the
orthogonalized update is `shaper.direction(p, pending)`. BatchedAnvil does the same for each stack of same-shape
matrices at once (anvil_update, with polar_batch); ShapedBatchedAnvil runs anvil_update's lines with one change: each
matrix's orthogonalized update becomes `shaper.after_polar(p, pending, update)`. The optimizers keep the unchanged
classes unless E8 or E9 is on; tests/method_checks.py (anvil_hook, batched_anvil_hook) checks that each shaped class
reproduces the original bit for bit when the shaper returns the orthogonalized update.
"""
import torch


def is_anvil_matrix(name, p):
    """The parameters make_optimizer gives ANVIL II: block matrices, not the value-embedding gates."""
    return name.startswith('transformer.h.') and p.ndim == 2 and 've_gate' not in name


def shaped_anvil_class(m):
    class ShapedAnvil(m.Anvil):
        shaper = None

        @torch.no_grad()
        def step(self):
            beta, engaged = self.rail_beta(), self.t >= self.engage
            fast_beta, weight = (self.fast_beta, self.fast_weight) if engaged else (beta, 1.0)
            for group in self.param_groups:
                lr, wd = group['lr'], group['weight_decay']
                for p in group['params']:
                    g = p.grad.float()
                    lanes = 1 if p.size(0) >= p.size(1) else 0
                    state = self.state[p]
                    if not state:
                        state.update(fast=torch.zeros_like(g), slow=torch.zeros_like(g),
                                     energy=torch.zeros_like(g.mean(lanes, keepdim=True)))
                    fast, slow, energy = state['fast'], state['slow'], state['energy']
                    fast.lerp_(g, 1 - fast_beta)
                    slow.lerp_(g, 1 - self.slow_beta)
                    update = self.shaper.direction(p, g.lerp(weight * fast + (1 - weight) * slow, beta))
                    energy.lerp_(update.square().mean(lanes, keepdim=True), 1 - self.lane_beta)
                    gained = update * energy.clamp_min(1e-10).rsqrt()
                    update = gained * (update.norm() / gained.norm().clamp_min(1e-10))
                    w = p.float()
                    decay = (slow * w >= 0) * w * (wd * lr)
                    p.copy_(w - lr * max(1, p.size(0) / p.size(1)) ** 0.5 * (update + decay))
            self.t += 1

    return ShapedAnvil


def shaped_batched_anvil_class(m):
    class ShapedBatchedAnvil(m.BatchedAnvil):
        shaper = None

        @torch.no_grad()
        def step(self):
            beta, engaged = self.rail_beta(), self.t >= self.engage
            fast_beta, weight = (self.fast_beta, self.fast_weight) if engaged else (beta, 1.0)
            device = self.param_groups[0]['params'][0].device
            value = lambda v: torch.tensor(float(v), device=device)
            for group in self.param_groups:
                lr, wd = group['lr'], group['weight_decay']
                by_shape = {}
                for p in group['params']:
                    by_shape.setdefault(tuple(p.shape), []).append(p)
                for shape, params in by_shape.items():
                    stack = self.stacked_state(shape, params)
                    new = self.shaped_update(
                        params, torch.stack([p.grad.float() for p in params]), torch.stack([p.float() for p in params]),
                        stack['fast'], stack['slow'], stack['energy'], value(beta), value(fast_beta),
                        value(self.slow_beta), value(weight), value(self.lane_beta), value(lr), value(wd),
                        1 if shape[0] >= shape[1] else 0, max(1, shape[0] / shape[1]) ** 0.5, self.polar_margin)
                    torch._foreach_copy_(params, list(new.to(params[0].dtype).unbind(0)))
            self.t += 1

        def shaped_update(self, params, g, w, fast, slow, energy, beta, fast_beta, slow_beta, weight, lane_beta, lr,
                          wd, lanes, ratio, margin):
            """PR #40's anvil_update, with the shaper on each matrix's orthogonalized update."""
            fast.lerp_(g, 1 - fast_beta)
            slow.lerp_(g, 1 - slow_beta)
            pending = g.lerp(weight * fast + (1 - weight) * slow, beta)
            update = m.polar_batch(pending, margin)
            # in place, so the reductions below keep polar_batch's layout (a transposed view for tall stacks)
            for i, p in enumerate(params):
                update[i] = self.shaper.after_polar(p, pending[i], update[i])
            energy.lerp_(update.square().mean(lanes + 1, keepdim=True), 1 - lane_beta)
            gained = update * energy.clamp_min(1e-10).rsqrt()
            update = gained * (update.norm(dim=(1, 2), keepdim=True)
                               / gained.norm(dim=(1, 2), keepdim=True).clamp_min(1e-10))
            decay = (slow * w >= 0) * w * (wd * lr)
            return w - lr * ratio * (update + decay)

    return ShapedBatchedAnvil


SHAPED = {'Anvil': shaped_anvil_class, 'BatchedAnvil': shaped_batched_anvil_class}


def shaped_optimizer(m, model, recipe, steps, shaper, name='Anvil'):
    """m.make_optimizer(model, recipe, steps) with m.<name> (boost-cut's Anvil or PR #40's BatchedAnvil) replaced by its
    shaped class: the same groups, rates and state."""
    original = getattr(m, name)
    setattr(m, name, SHAPED[name](m))
    try:
        optimizer = m.make_optimizer(model, recipe, steps)
    finally:
        setattr(m, name, original)
    optimizer.optimizers[0].shaper = shaper
    return optimizer
