"""H3: the held-out scorer. The last 32 sequences of the training range feed no update; every in-run decision (which
copy to keep, whether a merge helped, whether to undo a step) reads this, never val.pt. Every call is logged."""
import time

import torch

from .common import JsonlLog


@torch.no_grad()
def mean_loss(model, rows, device, batch_size):
    """Mean next-token loss over `rows`, computed as utils/eval.py does (bf16 autocast, float64 sums)."""
    total = torch.zeros(2, dtype=torch.float64, device=device)
    for batch in rows.split(batch_size):
        batch = batch.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss = model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), loss_reduction='sum')
        total[0] += loss.double()
        total[1] += batch[:, 1:].numel()
    return (total[0] / total[1]).item()


class HeldoutScorer:
    """Scores all held-out sequences, or a subset, in one forward."""

    def __init__(self, rows, log_path, device):
        self.rows, self.device = rows, device
        self.log = JsonlLog(log_path) if len(rows) else None
        self.calls = 0

    def __bool__(self):
        return len(self.rows) > 0

    def loss(self, model, rows=None):
        """The mean loss over all held-out sequences, or the subset `rows` (indices into them), in one forward."""
        chosen = self.rows if rows is None else self.rows[list(rows)]
        return mean_loss(model, chosen, self.device, len(chosen))

    def score(self, model, step, reason, loss=None, rows=None, **fields):
        """Score and log one call. `reason` names the decision (or 'monitor' when nothing depends on it); `rows`
        scores a subset of the held-out sequences (logged with it); `loss`, a function of the model, replaces the
        one-forward mean (the Cut chain's end merge scores as the record does)."""
        if not self:
            raise RuntimeError('this run holds out no sequences (heldout: 0)')
        start = time.perf_counter()
        value = loss(model) if loss is not None else self.loss(model, rows)
        self.calls += 1
        subset = {} if rows is None else dict(rows=list(rows))
        self.log.write(dict(call=self.calls, step=step, reason=reason, heldout_loss=value,
                            seconds=round(time.perf_counter() - start, 3), **subset, **fields))
        return value
