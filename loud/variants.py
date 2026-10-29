"""Proxy variants: each changes one thing about how the backprop proxy forms or takes an update (v2's E2 and E4).
Each is off unless its option is set (loud/options.py, PROXY); the plain proxy never enters this module.

    Blur          E2: the gradient of the loss blurred by activation noise, from an exact antithetic pair of passes.
    StepControl   E4: undo an update that raised the loss on held-out rows; optionally scale every rate by the outcomes.
"""
import contextlib

import torch

from .heldout import mean_loss
from .options import JUDGE_ROWS

BLUR_SEED = 7907    # the blur draws from its own generator, seeded with the job seed + BLUR_SEED


class Blur:
    """E2. The sites are the raw embedding (wte's output) and each block's mid stream (after attention, before the
    MLP), nine for the benchmark model. At each site, token t gets r[t] = s * RMS(a[t]) * sqrt(width) * u, u uniform
    on the unit sphere and a[t] its clean activation there: each coordinate's RMS displacement is s times the token's
    activation RMS, the units of PR #40's chain_sigma (its probe step is 0.01). Two forwards add +r and -r (the same
    r) at every site at once. The mean of their gradients has no odd-order terms in r, so over the draws it is the
    gradient of the loss averaged over the activation noise, to second order. The displacement is added in the
    activation's dtype (bf16 on the GPU); a mid-stream displacement is added to the attention output, which is what a
    block adds to its input there."""

    def __init__(self, model, s, seed, device):
        self.model, self.width = model, model.config.n_embd
        self.scale = s * self.width ** 0.5
        self.generator = torch.Generator(device).manual_seed(seed + BLUR_SEED)
        self.shifts = None

    @contextlib.contextmanager
    def _sites(self, visit):
        """For one forward, call visit(site, output, block_input) at each site: output writes the site's activation
        (wte's output; a block's attention output, added to block_input). A returned tensor replaces the output."""
        inputs, handles = {}, []

        def embedding(module, args, output):
            return visit(0, output, None)

        def block_input(i):
            def hook(module, args):
                inputs[i] = args[0]
            return hook

        def attention(i):
            def hook(module, args, output):
                return visit(i + 1, output, inputs.pop(i))
            return hook

        try:
            handles.append(self.model.transformer.wte.register_forward_hook(embedding))
            for i, block in enumerate(self.model.transformer.h):
                handles.append(block.register_forward_pre_hook(block_input(i)))
                handles.append(block.attn.register_forward_hook(attention(i)))
            yield
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def measure(self, forward):
        """Run the clean forward (forward() returns its loss) and draw this micro-batch's displacements from its
        per-token RMS at every site. Returns the clean loss."""
        seen = []

        def visit(site, output, block_input):
            a = output.float() if block_input is None else block_input.float() + output.float()
            seen.append((a.square().mean(-1, keepdim=True).sqrt(), output.dtype))

        with self._sites(visit):
            loss = forward()
        self.shifts = []
        for rms, dtype in seen:  # sites in forward order: the embedding, then each block's mid stream
            u = torch.randn(rms.shape[:-1] + (self.width,), device=rms.device, generator=self.generator)
            self.shifts.append(u.mul_(self.scale * rms / u.norm(dim=-1, keepdim=True)).to(dtype))
        return loss

    def shifted(self, sign):
        """A context in which forwards add sign * r (the last measure()'s) at every site."""
        def visit(site, output, block_input):
            return output + self.shifts[site] if sign > 0 else output - self.shifts[site]
        return self._sites(visit)


class Snapshot:
    """The weights and the optimizer state (every state tensor; ANVIL II's schedule clock is left to the loop),
    copied into buffers that are allocated once per tensor and reused, so a rejected update can be undone exactly."""

    def __init__(self, model, optimizer):
        self.params, self.optimizers = list(model.parameters()), optimizer.optimizers
        self.buffers = {}

    def _tensors(self):
        held = list(self.params)
        for o in self.optimizers:
            for state in o.state.values():
                held += [v for v in state.values() if torch.is_tensor(v)]
        return held

    def save(self):
        self.saved = self._tensors()
        for t in self.saved:
            if id(t) not in self.buffers:
                self.buffers[id(t)] = torch.empty_like(t)
        torch._foreach_copy_([self.buffers[id(t)] for t in self.saved], self.saved)

    def restore(self):
        """Undo the step: weights, rails, energy and Adam's moments and counts. ANVIL II's clock `t` is not a record
        of the data but its schedule (rail-beta ramp, engage, cooldown), which follows the loop's update count as the
        warmdown does, so it stays advanced."""
        torch._foreach_copy_(self.saved, [self.buffers[id(t)] for t in self.saved])
        known = {id(t) for t in self.saved}
        created = [t for t in self._tensors() if id(t) not in known]
        if created:  # the undone step was the first: zero state is what a fresh optimizer starts from
            torch._foreach_zero_(created)


class StepControl:
    """E4. Each update is judged on JUDGE_ROWS of the held-out sequences, rotating (update s takes rows 4(s - 1) ..
    4s - 1 mod 32): their loss under the old weights and under the new ones, a paired comparison costing two forwards
    of 4 sequences. Never on the batch that produced the step, which almost always improves, nor on the next training
    batch, which would then feed two updates. If the held-out loss rose, the weights and the optimizer state go back
    to before the update and the batch feeds no update ('reject', v1). 'adapt' (v2) also keeps a global multiplier on
    every group's rate, on top of the hand schedule: times accept_up after an accept, times accept_down after a
    reject. It settles where the acceptance rate a satisfies a * ln(up) + (1 - a) * ln(down) = 0: 0.59 for v2's 1.1
    and 0.87. With the warmdown removed (warmdown: 0, v2b), a falling acceptance late in training decays the
    multiplier geometrically: the measured warmdown. Both losses of every update are in metrics.jsonl."""

    def __init__(self, model, optimizer, o, heldout, device):
        self.adapt, self.up, self.down = o['step_control'] == 'adapt', o['accept_up'], o['accept_down']
        self.model, self.heldout, self.device = model, heldout, device
        self.multiplier = 1.0
        self.snapshot = Snapshot(model, optimizer)

    def rows(self, step):
        start = (step - 1) * JUDGE_ROWS % len(self.heldout)
        return [(start + i) % len(self.heldout) for i in range(JUDGE_ROWS)]

    def loss(self, step):
        return mean_loss(self.model, self.heldout[self.rows(step)], self.device, JUDGE_ROWS)

    def scale(self, optimizer):
        """Apply the multiplier to the rates set_rate just set (they are recomputed from initial_lr every update)."""
        if self.adapt:
            for group in optimizer.param_groups:
                group['lr'] *= self.multiplier

    def save(self, step):
        """Before the step: the held-out loss at the old weights, and what the step will change."""
        self.before = self.loss(step)
        self.snapshot.save()

    def judge(self, step):
        """After the step: undo it if the held-out loss rose. Returns the metrics fields."""
        after = self.loss(step)
        accepted = after <= self.before
        if not accepted:
            self.snapshot.restore()
        record = dict(accepted=accepted, heldout_before=self.before, heldout_after=after, judge_rows=self.rows(step))
        if self.adapt:
            record['lr_multiplier'] = self.multiplier  # the multiplier this update used
            self.multiplier *= self.up if accepted else self.down
        return record
