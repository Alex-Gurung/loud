"""ANVIL II (boost-cut's Anvil, the backprop proxy's optimizer for block matrices) with its orthogonalization as a hook,
for E8 and E9.

Anvil.step forms each matrix's pending update, g.lerp(w * fast + (1 - w) * slow, beta), and orthogonalizes it with
polar() before the per-lane gain, the decay and the step. ShapedAnvil.step is that code with one change: the
orthogonalized update is `shaper.direction(p, pending)`. The proxy keeps the unchanged Anvil unless E8 or E9 is on;
tests/method_checks.py (anvil_hook) checks that ShapedAnvil reproduces Anvil bit for bit when the shaper returns
polar(pending).
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


def shaped_optimizer(m, model, recipe, steps, shaper):
    """m.make_optimizer(model, recipe, steps) with ShapedAnvil in place of Anvil (same groups, rates and state)."""
    original, m.Anvil = m.Anvil, shaped_anvil_class(m)
    try:
        optimizer = m.make_optimizer(model, recipe, steps)
    finally:
        m.Anvil = original
    optimizer.optimizers[0].shaper = shaper
    return optimizer
