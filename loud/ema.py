"""H4's EMA teacher (plan v2): a copy of the model whose weights follow the student's by an exponential moving average,
for E14's target latent and E15's latent targets.

The decay rises linearly in the fraction of updates from 0.996 to 0.9995. ProteinJEPA (arXiv 2605.07554, Table 1)
gives those end points ("EMA decay 0.996 -> 0.9995") but not the curve between them; linear is I-JEPA's. The teacher
is kept in float32 whatever the student's dtypes (bf16 tables on the GPU): an update moves a weight by at most 0.4% of
its distance to the student's, below bf16's resolution, so a bf16 teacher would stop moving.

    teacher = EMAWeights(model, steps)
    ...                                       # after each optimizer step:
    teacher.update(step)                      # step: one-based
    teacher.model                             # the teacher (no gradients), for its forward
"""
import copy

import torch


class EMAWeights:
    def __init__(self, model, steps, start=0.996, end=0.9995):
        self.steps, self.start, self.end = steps, start, end
        self.model = copy.deepcopy(model).float().requires_grad_(False)
        self.pairs = list(zip(self.model.parameters(), model.parameters()))

    def decay(self, step):
        """The weight the teacher keeps at update `step` of `steps`."""
        return self.start + (self.end - self.start) * min(1.0, step / self.steps)

    @torch.no_grad()
    def update(self, step):
        """teacher <- decay * teacher + (1 - decay) * student, for every parameter."""
        teacher, student = zip(*self.pairs)
        torch._foreach_lerp_(list(teacher), [s.float() for s in student], 1 - self.decay(step))
