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

    def name(self, fraction):
        return f'ckpt_{round(fraction * 100):02d}.pt'

    def save(self, model, step, optimizer=None):
        """Save the checkpoint(s) due at this step; coinciding fractions link to the first file. With `optimizer`
        (the records' Optimizers(ANVIL II, Adam)), also <name>.optim.pt: ANVIL II's per-matrix state in float32 and
        the beta and fast weight of its next step, which H6's direction_stats needs to rebuild the pending update."""
        first, *rest = self.at[step]
        path = self.dir / self.name(first)
        torch.save(state_dict_cpu(model, self.dtype), path)
        self.saved[path.name] = dict(step=step, fraction=first, dtype=str(self.dtype or 'float32'))
        if optimizer is not None:
            anvil = optimizer.optimizers[0]
            names = {id(p): n for n, p in model.named_parameters()}
            state = {names[id(p)]: {k: v.detach().float().cpu() for k, v in anvil.state[p].items() if torch.is_tensor(v)}
                     for group in anvil.param_groups for p in group['params'] if p in anvil.state}
            engaged = anvil.t >= anvil.engage
            torch.save(dict(step=step, state=state, beta=anvil.rail_beta(),
                            fast_weight=anvil.fast_weight if engaged else 1.0), path.with_suffix('.optim.pt'))
        for f in rest:
            link = self.dir / self.name(f)
            if not link.exists():
                os.symlink(path.name, link)
            self.saved[link.name] = dict(self.saved[path.name], fraction=f, link=path.name)
        return path

    def save_final(self, model):
        """final.pt, the exact trained weights."""
        path = self.dir / 'final.pt'
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
