"""H4: weights at fixed fractions of the run, and the record's nine trajectory means.

Checkpoints: ckpt_00.pt (the initialization), ckpt_10, ckpt_20, ckpt_50 after round(f * steps) updates, and ckpt_100
after the last update of a complete run. All five store 2-D weights in one dtype, bfloat16 by default (half the disk;
an identical forward under bf16 autocast, which casts weights to bf16 anyway), so diagnostics that compare them
compare training progress, not precision; `checkpoint_dtype: float32` keeps them exact. final.pt is always the exact
trained state dict, whatever the run's length.

Means: as PR #40's end merge: from 60% of the epoch (the step's first sequence at or past 0.6 of the update sequences),
every 5 updates and at the last update, nine exponential means of every parameter in float32, weight
(0.5, 0.3, 0.2, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01) on the newest; the first qualifying step copies the weights.
"""
import os
import time

import torch

FRACTIONS = (0.0, 0.1, 0.2, 0.5, 1.0)
MERGE_FROM, MERGE_EVERY = 0.6, 5
MERGE_DECAYS = (0.5, 0.3, 0.2, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01)


def state_dict_cpu(model, dtype=None):
    out = {}
    for name, value in model.state_dict().items():
        value = value.detach()
        if dtype is not None and value.ndim == 2 and value.is_floating_point():
            value = value.to(dtype)
        out[name] = value.cpu().clone()
    return out


class FractionCheckpoints:
    def __init__(self, run_dir, steps, dtype='bfloat16', enabled=True):
        self.dir, self.enabled = run_dir, enabled
        self.dtype = None if dtype == 'float32' else getattr(torch, dtype)
        self.at = {}  # step -> fractions due there (several coincide only in very short runs)
        for f in FRACTIONS:
            self.at.setdefault(round(f * steps), []).append(f)
        self.saved = {}

    def due(self, step):
        return self.enabled and step in self.at

    def name(self, fraction, copy=None):
        return f'ckpt_{round(fraction * 100):02d}' + ('' if copy is None else f'.c{copy}') + '.pt'

    def save(self, model, step, optimizer=None, copy=None):
        """Save the checkpoint(s) due at this step; coinciding fractions link to the first file. With `copy` (E3), the
        name is ckpt_<f>.c<copy>.pt. With `optimizer` (the records' Optimizers(ANVIL II, Adam)), also <name>.optim.pt:
        ANVIL II's per-matrix state in float32 and the betas and fast weight of its next step, which H6's
        direction_stats needs to rebuild the pending update."""
        first, *rest = self.at[step]
        path = self.dir / self.name(first, copy)
        torch.save(state_dict_cpu(model, self.dtype), path)
        self.saved[path.name] = dict(step=step, fraction=first, dtype=str(self.dtype or 'float32'))
        if optimizer is not None:
            anvil = optimizer.optimizers[0]
            names = {id(p): n for n, p in model.named_parameters()}
            state = {names[id(p)]: {k: v.detach().float().cpu() for k, v in anvil.state[p].items() if torch.is_tensor(v)}
                     for group in anvil.param_groups for p in group['params'] if p in anvil.state}
            beta, engaged = anvil.rail_beta(), anvil.t >= anvil.engage  # as its next step reads them
            torch.save(dict(step=step, state=state, beta=beta, fast_beta=anvil.fast_beta if engaged else beta,
                            slow_beta=anvil.slow_beta, fast_weight=anvil.fast_weight if engaged else 1.0),
                       path.with_suffix('.optim.pt'))
        for f in rest:
            link = self.dir / self.name(f, copy)
            if not link.exists():
                os.symlink(path.name, link)
            self.saved[link.name] = dict(self.saved[path.name], fraction=f, link=path.name)
        return path

    def save_final(self, model, name='final.pt'):
        """The exact trained weights (final.pt; final_plain.pt, before a trainer's finish)."""
        path = self.dir / name
        torch.save(state_dict_cpu(model), path)
        return path


class TrajectoryMeans:
    def __init__(self, update_sequences, enabled=True):
        self.update_sequences, self.enabled = update_sequences, enabled
        self.means = None
        self.updates = []

    @torch.no_grad()
    def observe(self, model, step, first_sequence, last_step):
        if not self.enabled or not (step % MERGE_EVERY == 0 or step == last_step):
            return
        if first_sequence / self.update_sequences < MERGE_FROM:
            return
        params = [p.detach().float() for p in model.parameters()]
        if self.means is None:
            self.means = [[p.clone() for p in params] for _ in MERGE_DECAYS]
        else:
            for decay, mean in zip(MERGE_DECAYS, self.means):
                for m, p in zip(mean, params):
                    m.lerp_(p, decay)
        self.updates.append(step)

    def state_dicts(self, model):
        """Each mean as a state dict in the model's dtypes (for scoring with the fixed model)."""
        names = [n for n, _ in model.named_parameters()]
        out = []
        for mean in self.means or []:
            out.append({n: m.to(p.dtype) for n, m, p in zip(names, mean, model.parameters())})
        return out

    def save(self, path, model, dtype=torch.bfloat16):
        """2-D weights in `dtype` (bf16 by default, as the intermediate checkpoints), the rest in float32."""
        names = [n for n, _ in model.named_parameters()]
        cast = lambda m: m.to(dtype) if dtype is not None and m.ndim == 2 else m
        torch.save(dict(decays=MERGE_DECAYS, updates=self.updates, dtype=str(dtype or 'float32'),
                        means=[{n: cast(m).cpu() for n, m in zip(names, mean)} for mean in self.means or []]), path)


def end_merge(module, model, means, heldout, step, loss=None):
    """The record's end merge (`module`: PR #40's or PR #46's run.py, whose end_merge and joint_parts look up
    holdout_loss by name) of the nine trajectory means, on the held-out rows; returns its result fields. Each candidate
    it scores goes through H3's log (heldout.jsonl, reason 'merge'); `loss(model, rows)` scores one (default: H3's
    one-forward mean). Timed by the caller's clock, as the records time it."""
    if means.means is None:  # as the records: no means, no merge
        return {}

    def holdout_loss(candidate, rows, rank, world):
        return heldout.score(candidate, step, 'merge', loss=None if loss is None else lambda c: loss(c, rows))
    module.holdout_loss = holdout_loss
    start = time.perf_counter()
    out = module.end_merge(model, means.means, heldout.rows, 0, 1)
    if heldout.device.type == 'cuda':
        torch.cuda.synchronize()
    return dict(out, merge_seconds=round(time.perf_counter() - start, 2))
