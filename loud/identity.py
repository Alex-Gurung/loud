"""H5: the identity-path forward-only estimator (Noise PR #26, track 2: 4.1772 in 46.96 s on 8xH100) on one GPU.

PR #26's train/speedrun/run.py @2604407 is imported unchanged: its model, init, optimizer, settings and FastRelay
estimator. That script puts one sequence on each of 8 GPUs; each GPU's estimator draws from generators seeded by its
rank, and Boost.step sums the per-sequence gradients (each divided by its token count) with an all-reduce, divides by
the number of sequences, casts to each parameter's dtype and steps the optimizer. Here one estimator per virtual GPU
(rank = the sequence's index in the batch) measures its sequence in turn, and their gradients are summed the same way
before one optimizer step. With the same seed and data, the draws are the 8-GPU run's.
Differences from PR #26, all deliberate:
  - data order: the records' fixed randperm(seed 1) for every seed (PR #26 shuffles with seed + 1; same at seed 0);
  - the shared loop (loud/train.py): held-out split, H4, untimed diagnostics, official scoring in a fresh process.
"""
from dataclasses import asdict

import torch

from . import refs
from .options import PRESETS, identity_options
from .train import current_rates, rate_labels, set_rate, train


def attach(capture, model):
    """Register a Capture's hooks again, as PR #26's Capture.__init__ does once. Each estimator's hooks are attached
    only while it measures: eight idle sets would add about 460 hook calls to every forward."""
    capture.hooks = [module.register_forward_hook(capture.output_hook(name)) for name, module in capture.modules.items()]
    capture.hooks += [b.register_forward_pre_hook(capture.block_hook(i)) for i, b in enumerate(model.transformer.h)]


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
        self.estimators = []
        self.reset()

    def reset(self):
        for estimator in self.estimators:
            estimator.close()
        m, o = self.m, self.o
        self.model = m.make_model(self.device, self.seed, self.config)
        self.optimizer = m.make_optimizer(self.model, self.recipe, self.steps)
        self.labels = rate_labels(self.optimizer, self.recipe.lr)
        self.named = list(self.model.named_parameters())
        self.estimators = [m.FastRelay(self.model, self.settings, self.seed, rank, o['batch'])
                           for rank in range(o['batch'])]
        for estimator in self.estimators:
            estimator.close()  # measure() attaches the hooks of the one measuring

    def describe(self):
        return dict(name='pr26', file=refs.TRAINERS['pr26'], recipe=asdict(self.recipe),
                    settings=asdict(self.settings), model=asdict(self.config), virtual_gpus=self.o['batch'])

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
        total, losses = None, []
        for rank, estimator in enumerate(self.estimators):
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

    def diagnose(self, step):
        return {}


def run(spec, run_dir):
    o = identity_options(spec.get('args', {}))
    return train(spec, run_dir, o, lambda device, seed, steps, heldout: IdentityPath(o, device, seed, steps, heldout),
                 method='identity path (PR #26)')
