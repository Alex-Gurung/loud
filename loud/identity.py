"""H5: the identity-path forward-only estimator (Noise PR #26, track 2: 4.1772 in 46.96 s on 8xH100) on one GPU.

PR #26's train/speedrun/run.py @2604407 is imported unchanged: its model, init, optimizer, settings and FastRelay
estimator. That script puts one sequence on each of 8 GPUs; each GPU's estimator draws from generators seeded by its
rank, and Boost.step sums the per-sequence gradients (each divided by its token count) with an all-reduce, divides by
the number of sequences, casts to each parameter's dtype and steps the optimizer. Here one estimator per virtual GPU
(rank = the sequence's index in the batch) measures its sequence in turn, and their gradients are summed the same way
before one optimizer step. With the same seed and data, the draws are the 8-GPU run's.

E6's flags (the doc's --rows and --row-offset-per-step) act only on the top-stream probe, the one Hadamard cycle that
reads the loss: `rows` of PR #26's own probe() are used, [start, start + rows) of the signed basis, and the credit is
scaled by width / rows. That is unbiased: the basis carries random column signs, so every row subset's outer products
average to rows / width times the identity. With row_offset_per_step, the K = width / rows consecutive updates of a
cycle read complementary windows of one signed basis, so each cycle measures every direction once, spread over K
updates for momentum to stitch (E6; the first cycle, updates 1 to K - 1, has no window 0); without it, every update
reads rows [0, rows) of a fresh basis, E6's control. The embedding's
RMS-norm step stays a full cycle (a local Jacobian, no loss reading). rows = 512 runs PR #26's probe() itself.
Differences from PR #26, all deliberate:
  - data order: the records' fixed randperm(seed 1) for every seed (PR #26 shuffles with seed + 1; same at seed 0);
  - the shared loop (loud/train.py): held-out split, H4, untimed diagnostics, official scoring in a fresh process.
"""
from dataclasses import asdict

import torch

from . import refs
from .options import PRESETS, identity_options
from .train import current_rates, rate_labels, set_rate, train

DRIFT_SEED = 104729  # the drift check's own estimator, so its draws never touch the training estimators' streams


def estimator_class(m):
    class Estimator(m.FastRelay):
        """PR #26's FastRelay, with the top-stream probe optionally limited to a window of Hadamard rows. All windows
        of one cycle (`cycle`, set by the trainer) read one signed basis, so the K windows of a cycle are complementary
        rows of a single Hadamard cycle. Every probe still draws its basis signs as PR #26 does, so each K consumes
        the same random stream."""
        rows, start, window, cycle = None, 0, None, None
        held = (None, None)  # (cycle, its basis)

        def share(self, total):
            return self.window if self.window is not None else super().share(total)

        def basis(self):
            fresh = super().basis()
            if self.window is None or self.cycle is None:
                return fresh
            if self.held[0] != self.cycle:
                self.held = (self.cycle, fresh)
            return self.held[1]

        def probe(self, x, y, mix=None):
            if self.rows is None:
                return super().probe(x, y, mix)
            self.window = (self.start, self.start + self.rows)
            try:
                return super().probe(x, y, mix) * (self.width / self.rows)
            finally:
                self.window = None

    return Estimator


def attach(capture, model):
    """Register a Capture's hooks again, as PR #26's Capture.__init__ does once. Each estimator's hooks are attached
    only while it measures: eight idle sets would add about 460 hook calls to every forward."""
    capture.hooks = [module.register_forward_hook(capture.output_hook(name)) for name, module in capture.modules.items()]
    capture.hooks += [b.register_forward_pre_hook(capture.block_hook(i)) for i, b in enumerate(model.transformer.h)]


def cosine(a, b):
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-30)).item()


class IdentityPath:
    def __init__(self, o, device, seed, steps, heldout):
        m = self.m = refs.trainer_module('pr26')
        self.o, self.device, self.seed, self.steps = o, device, seed, steps
        n_layer, width = PRESETS[o['preset']]
        self.config = m.GPTConfig(sequence_len=m.SEQ, n_layer=n_layer, n_head=max(1, width // 64),
                                  n_kv_head=max(1, width // 64), n_embd=width)
        hidden = 4 * width
        self.recipe = m.Recipe(anvil_lr=0.023)  # PR #26's --anvil-lr default
        self.settings = m.Settings(fc_k=hidden // 8, route_k=hidden // 4, fast_mlp_mode='all', no_chain=True,
                                   value_path=False, skip_mid=True)  # PR #26's main() defaults
        self.rows = o['rows']  # None: the full cycle (identity_options folds rows == width into None)
        self.cycles = width // (self.rows or width)
        self.Estimator = estimator_class(m)
        self.estimators, self.drift = [], None
        self.probe_row = heldout[:1].to(device)  # the drift check's sequence: held out, so no momentum holds it
        self.reset()

    def reset(self):
        for estimator in self.estimators + [self.drift]:
            if estimator is not None:
                estimator.close()
        m, o = self.m, self.o
        self.model = m.make_model(self.device, self.seed, self.config)
        self.optimizer = m.make_optimizer(self.model, self.recipe, self.steps)
        self.labels = rate_labels(self.optimizer, self.recipe.lr)
        self.named = list(self.model.named_parameters())
        self.estimators = [self.Estimator(self.model, self.settings, self.seed, rank, o['batch'])
                           for rank in range(o['batch'])]
        for estimator in self.estimators:
            estimator.rows = self.rows
        self.drift = (self.Estimator(self.model, self.settings, self.seed + DRIFT_SEED, 0, o['batch'])
                      if o['drift_every'] else None)
        for estimator in self.estimators + [self.drift]:
            if estimator is not None:
                estimator.close()  # measure() attaches the hooks of the one measuring

    def describe(self):
        return dict(name='pr26', file=refs.TRAINERS['pr26'], recipe=asdict(self.recipe),
                    settings=asdict(self.settings), model=asdict(self.config), virtual_gpus=self.o['batch'],
                    probe_rows=self.rows or PRESETS[self.o['preset']][1],
                    row_offset_per_step=self.o['row_offset_per_step'])

    def measure(self, estimator, x, y):
        """One sequence's flat float32 gradient (divided by its token count, as Boost.step) and its clean loss."""
        attach(estimator.capture, self.model)
        try:
            gradients = estimator.estimate(x, y)
        finally:
            estimator.close()
        valid = (y != -1).sum()
        flat = torch.cat([(gradients[name] / valid).float().reshape(-1) for name, _ in self.named])
        loss = (estimator.capture.clean_loss * (y != -1)).sum() / valid  # clean_loss[y != -1].mean(), with no sync
        for store in (estimator.capture.inputs, estimator.capture.outputs, estimator.capture.blocks):
            store.clear()  # this sequence's activations are no longer needed
        return flat, loss

    def update(self, step, batch):
        set_rate(self.optimizer, step, self.steps, self.recipe.warmdown)
        fields = current_rates(self.optimizer, self.labels)
        # E6: update s reads rows [(s mod K) * rows, ...) of the cycle s // K, the plan's indexing
        start = (step % self.cycles) * self.rows if self.o['row_offset_per_step'] else 0
        if self.rows is not None:
            fields['probe_rows'] = [start, start + self.rows]
        total, losses = None, []
        for rank, estimator in enumerate(self.estimators):
            estimator.start = start
            estimator.cycle = step // self.cycles if self.o['row_offset_per_step'] else None
            x, y = batch[rank:rank + 1, :-1].contiguous(), batch[rank:rank + 1, 1:].contiguous()
            flat, loss = self.measure(estimator, x, y)
            total = flat if total is None else total.add_(flat)
            losses.append(loss)
        total /= len(self.estimators)
        if not bool(torch.isfinite(total).all()):
            raise FloatingPointError('Non-finite estimate')
        self.optimizer.zero_grad(set_to_none=True)
        offset = 0
        for _, p in self.named:
            p.grad = total[offset:offset + p.numel()].view_as(p).to(p.dtype)
            offset += p.numel()
        self.optimizer.step()
        return torch.stack(losses).mean(), fields

    @torch.no_grad()
    def diagnose(self, step):
        """E6's drift check: a full-cycle estimate on a held-out sequence against the momentum ANVIL II steps along,
        weight * fast + (1 - weight) * slow, and against fast and slow alone, over all block matrices. One sequence's
        estimate is noisy, so read the cosines across K, not against an absolute threshold."""
        if self.drift is None or step % self.o['drift_every']:
            return {}
        x, y = self.probe_row[:, :-1].contiguous(), self.probe_row[:, 1:].contiguous()
        flat, _ = self.measure(self.drift, x, y)
        anvil = self.optimizer.optimizers[0]
        weight = anvil.fast_weight if anvil.t >= anvil.engage else 1.0
        full, fast, slow, offset = [], [], [], 0
        for _, p in self.named:
            if p in anvil.state:
                full.append(flat[offset:offset + p.numel()])
                fast.append(anvil.state[p]['fast'].reshape(-1))
                slow.append(anvil.state[p]['slow'].reshape(-1))
            offset += p.numel()
        full, fast, slow = torch.cat(full), torch.cat(fast), torch.cat(slow)
        return dict(drift=dict(blend=cosine(full, weight * fast + (1 - weight) * slow), fast=cosine(full, fast),
                               slow=cosine(full, slow)))


def run(spec, run_dir):
    o = identity_options(spec.get('args', {}))
    return train(spec, run_dir, o, lambda device, seed, steps, heldout: IdentityPath(o, device, seed, steps, heldout),
                 method='identity path (PR #26)')
