"""ANVIL II (boost-cut's Anvil, the backprop proxy's optimizer for block matrices) with its orthogonalization as a hook,
for E8 and E9.

Anvil.step forms each matrix's pending update, g.lerp(w * fast + (1 - w) * slow, beta), and orthogonalizes it with
polar() before the per-lane gain, the decay and the step. ShapedAnvil.step is that code with one change: the
orthogonalized update is `shaper.direction(p, pending, before)`, `before` the blended momentum w * fast + (1 - w) * slow
as it stood before this batch's gradient entered the rails; a shaper with lane_gain False also skips the per-lane gain and
its Frobenius renormalization (E8, whose per-direction scaling they would undo). The proxy keeps the unchanged Anvil
unless E8 or E9 is on; tests/method_checks.py (anvil_hook) checks that ShapedAnvil reproduces Anvil bit for bit when the
shaper returns polar(pending). ShapedBatchedAnvil is the same hook on PR #40's BatchedAnvil, for E8's H8 setting
(batched_anvil_hook: bit for bit when the shaper returns polar_batch's update).
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
                    before = weight * fast + (1 - weight) * slow  # E8's directions: fixed before this batch
                    fast.lerp_(g, 1 - fast_beta)
                    slow.lerp_(g, 1 - self.slow_beta)
                    update = self.shaper.direction(p, g.lerp(weight * fast + (1 - weight) * slow, beta), before)
                    if self.shaper.lane_gain:  # E8 turns the gain and its renormalization off: they undo its scaling
                        energy.lerp_(update.square().mean(lanes, keepdim=True), 1 - self.lane_beta)
                        gained = update * energy.clamp_min(1e-10).rsqrt()
                        update = gained * (update.norm() / gained.norm().clamp_min(1e-10))
                    w = p.float()
                    decay = (slow * w >= 0) * w * (wd * lr)
                    p.copy_(w - lr * max(1, p.size(0) / p.size(1)) ** 0.5 * (update + decay))
            self.t += 1

    return ShapedAnvil


def shaped_batched_anvil_class(m):
    """PR #40's BatchedAnvil (H8's optimizer) with the same hook: each matrix's orthogonalized update is
    `shaper.direction(p, pending, before)` in place of polar_batch's, and a shaper with lane_gain False skips the
    per-lane gain and its renormalization. The rest is anvil_update's lines, run eagerly."""

    class ShapedBatchedAnvil(m.BatchedAnvil):
        shaper = None

        @torch.no_grad()
        def step(self):
            beta, engaged = self.rail_beta(), self.t >= self.engage
            fast_beta, weight = (self.fast_beta, self.fast_weight) if engaged else (beta, 1.0)
            device = self.param_groups[0]['params'][0].device
            value = lambda v: torch.tensor(float(v), device=device)  # as BatchedAnvil hands them to anvil_update
            beta, fast_beta, slow_beta, weight = value(beta), value(fast_beta), value(self.slow_beta), value(weight)
            for group in self.param_groups:
                lr, wd = value(group['lr']), value(group['weight_decay'])
                by_shape = {}
                for p in group['params']:
                    by_shape.setdefault(tuple(p.shape), []).append(p)
                for shape, params in by_shape.items():
                    stack = self.stacked_state(shape, params)
                    g, w = torch.stack([p.grad.float() for p in params]), torch.stack([p.float() for p in params])
                    fast, slow, energy = stack['fast'], stack['slow'], stack['energy']
                    before = weight * fast + (1 - weight) * slow  # E8's directions: fixed before this batch
                    fast.lerp_(g, 1 - fast_beta)
                    slow.lerp_(g, 1 - slow_beta)
                    pending = g.lerp(weight * fast + (1 - weight) * slow, beta)
                    # laid out as polar_batch returns it (a transposed view for tall stacks), so the reductions
                    # below run in the same order
                    update = (pending.new_empty(len(params), shape[1], shape[0]).transpose(1, 2)
                              if shape[0] > shape[1] else torch.empty_like(pending))
                    for i, p in enumerate(params):
                        update[i] = self.shaper.direction(p, pending[i], before[i])
                    if self.shaper.lane_gain:
                        lanes = 1 if shape[0] >= shape[1] else 0
                        energy.lerp_(update.square().mean(lanes + 1, keepdim=True), 1 - value(self.lane_beta))
                        gained = update * energy.clamp_min(1e-10).rsqrt()
                        update = gained * (update.norm(dim=(1, 2), keepdim=True)
                                           / gained.norm(dim=(1, 2), keepdim=True).clamp_min(1e-10))
                    decay = (slow * w >= 0) * w * (wd * lr)
                    new = w - lr * max(1, shape[0] / shape[1]) ** 0.5 * (update + decay)
                    torch._foreach_copy_(params, list(new.to(params[0].dtype).unbind(0)))
            self.t += 1

    return ShapedBatchedAnvil


SHAPED = {'Anvil': shaped_anvil_class, 'BatchedAnvil': shaped_batched_anvil_class}


def shaped_optimizer(m, model, recipe, steps, shaper, name='Anvil'):
    """m.make_optimizer(model, recipe, steps) with m.<name> (boost-cut's Anvil, or PR #40's BatchedAnvil for H8)
    replaced by its shaped class: the same groups, rates and state."""
    original = getattr(m, name)
    setattr(m, name, SHAPED[name](m))
    try:
        optimizer = m.make_optimizer(model, recipe, steps)
    finally:
        setattr(m, name, original)
    optimizer.optimizers[0].shaper = shaper
    return optimizer
