"""E21 B and C: the valley around a run's weights at fixed fractions of training, from inside the loop (valley: true).

At each fraction's update s, with w the weights before it and u = w_after - w the update the optimizer actually made
(training is untouched: every scan restores the weights exactly):

  B  landing   the held-out loss (all held-out rows) at w + t u, t = 0, 0.1, ..., 3. ANVIL II fixes each update's
               size, so the classic 2/eta edge does not apply; the best t* says where the step lands: near 0.5 it
               overshoots to the far wall (at the edge), near 1 it lands on the floor, above 1 it under-steps.
  C  lines     the held-out loss at w + t d on a long grid of multiples, for d: the drift w - w_{s-k} (k = 25, 100
               updates), u, the exact gradient's descent direction on the update's own batch, and two random
               directions; the last three are scaled tensor by tensor to the 100-update drift's norms, so their t is in
               drift units. How far along the drift the loss keeps falling says whether long jumps along the valley
               have room.

The line toward a second seed's weights is the seed_line diagnostic (loud/diagnostics.py), on the H4 checkpoints.
Results go into the update's metrics record (`valley`) and valley.json beside it.
"""
import json

import torch

from .heldout import mean_loss

FRACTIONS = (0.2, 0.5, 1.0)
DRIFTS = (25, 100)
LANDING = [round(0.1 * i, 1) for i in range(31)]
LINE = (0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 32)
RANDOM_SEED = 4049  # the random directions: the job seed plus this


class Valley:
    def __init__(self, model, rows, steps, device, seed, path):
        self.rows, self.device, self.path = rows, device, path
        self.at = {max(1, round(f * steps)): f for f in FRACTIONS}
        # the weights before update s are copied in the pause after update s - 1: the fraction steps' bases, and each
        # drift's start s - k
        self.wanted = {s for s in self.at} | {s - k for s in self.at for k in DRIFTS if s - k >= 1}
        self.saved, self.out = {}, {}
        self.generator = torch.Generator(device).manual_seed(seed + RANDOM_SEED)
        self.params = [p for p in model.parameters()]

    def _copy(self):
        return [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def _set(self, base, d=None, t=0.0):
        for p, w, v in zip(self.params, base, d or base):
            p.copy_(w if d is None else w + t * v)

    def _loss(self, model):
        return mean_loss(model, self.rows, self.device, 16)

    def _scan(self, model, base, d, grid):
        out = {}
        for t in grid:
            self._set(base, d, t)
            out[str(t)] = round(self._loss(model), 5)
        return out

    def _gradient(self, model, base, batch):
        """The descent direction of the exact (full-graph) mean loss on the update's batch at the base weights."""
        self._set(base)
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        with torch.enable_grad(), torch.autocast(self.device.type, dtype=torch.bfloat16):
            loss = model(x, y)
            grads = torch.autograd.grad(loss, self.params, allow_unused=True)
        return [torch.zeros_like(p) if g is None else -g.float() for p, g in zip(self.params, grads)]

    @staticmethod
    def _scaled(d, like):
        """d scaled tensor by tensor to like's norms."""
        return [v * (w.norm() / v.norm().clamp_min(1e-30)) for v, w in zip(d, like)]

    def after(self, step, model, batch):
        """In the pause after update `step` (0: before the first): scan at a fraction's update, then keep the copies
        later updates need. Returns the metrics fields."""
        record = {}
        if step in self.at:
            base, after = self.saved[step], self._copy()
            u = [a - b for a, b in zip(after, base)]
            landing = self._scan(model, base, u, LANDING)
            best = min(landing, key=landing.get)
            drifts = {k: [w - v for w, v in zip(base, self.saved[step - k])] for k in DRIFTS if step - k in self.saved}
            unit = drifts.get(100) or drifts.get(25)
            directions = {f'drift{k}': d for k, d in drifts.items()}
            directions['update'] = u
            if unit is not None:
                directions['gradient'] = self._scaled(self._gradient(model, base, batch), unit)
                for i in range(2):
                    noise = [torch.randn(p.shape, device=p.device, generator=self.generator) for p in self.params]
                    directions[f'random{i}'] = self._scaled(noise, unit)
            lines = {name: self._scan(model, base, d, LINE) for name, d in directions.items()}
            norms = {name: round(float(torch.sqrt(sum(v.float().square().sum() for v in d))), 5)
                     for name, d in directions.items()}
            self._set(after)  # training goes on from the update it made
            falls = {name: max((float(t) for t, v in line.items() if v < line['0']), default=0.0)
                     for name, line in lines.items()}
            record = dict(fraction=self.at[step], landing=landing, t_star=float(best), lines=lines, norms=norms,
                          falls_to=falls)
            self.out[str(self.at[step])] = record
            self.path.write_text(json.dumps(self.out, indent=1) + '\n')
            for k in [k for k in self.saved if k <= step]:
                if k not in {s - d for s in self.at if s > step for d in DRIFTS}:
                    del self.saved[k]
        if step + 1 in self.wanted:
            self.saved[step + 1] = self._copy()
        return {'valley': record} if record else {}
