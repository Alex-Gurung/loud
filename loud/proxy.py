"""H1: the single-GPU backprop proxy (development stand-in for the forward-only estimator; never a Noise entry).

backprop_preset.py (boost-cut @539386e) in one process: the reference trainer's model, init, optimizer (ANVIL II on
block matrices, Adam elsewhere), rates and warmdown, with the shared loop's data order and untimed warmup
(loud/train.py). The 8-GPU script gives each GPU one sequence, averages the float32 gradients across GPUs and casts
back to each parameter's dtype; here each micro-batch (default: one sequence) is one GPU's share, and the same float32
average is formed before the cast. The batch-mean loss is the gradient's objective, as there. Adds H2 (per-sequence
gradients), the one-change variants of E2, E4 and E10 (loud/variants.py), and the hooks through which E8 and E9
replace ANVIL II's orthogonalization (loud/anvil.py), each off by default.
"""
import contextlib
import copy
from dataclasses import asdict, is_dataclass

import torch

from . import checkpoints, refs
from .anvil import shaped_optimizer
from .curvature import CurvatureDamping
from .fisher import FisherFactor
from .options import PRESETS, proxy_options
from .pergrad import PerSequenceGrads
from .train import current_rates, rate_labels, set_rate, train
from .variants import Blur, StepControl, TokenShaping


class Reference:
    """The reference trainer's model, init, optimizer and schedule constants, used verbatim."""

    def __init__(self, key, lr_mult, preset):
        m = self.module = refs.trainer_module(key)
        n_layer, width = PRESETS[preset]
        if hasattr(m, 'WIDTH'):
            m.WIDTH, m.HIDDEN = width, 4 * width  # as the reference main() does before anything compiles
        self.config = m.GPTConfig(n_layer=n_layer, n_head=width // 64, n_kv_head=width // 64, n_embd=width)
        if key == 'boostcut':
            s = m.Recipe()
            s.lr, s.anvil_lr = s.lr * lr_mult, s.anvil_lr * lr_mult
            self.warmdown, self.final = s.warmdown, 0.0
            self.make_model = lambda device, seed: m.make_model(device, seed, self.config)
            self.extra = dict(polar_margin=m.POLAR_MARGIN)
        elif key in ('pr40', 'pr46'):
            s = m.Settings()
            s.lr, s.anvil_lr, s.model = s.lr * lr_mult, s.anvil_lr * lr_mult, self.config
            self.warmdown, self.final = s.warmdown, s.final
            self.make_model = lambda device, seed: m.make_model(device, seed, self.config, s.projection_std, s.head_std)
            self.extra = {}
        else:
            raise ValueError(f'unknown recipe {key!r}')
        self.settings, self.key = s, key
        self.make_optimizer = lambda model, steps: m.make_optimizer(model, s, steps)

    def describe(self):
        settings = asdict(self.settings) if is_dataclass(self.settings) else dict(vars(self.settings))
        if self.key in ('pr40', 'pr46'):  # only the optimizer/schedule/init fields act on backprop
            keep = ('lr', 'betas', 'warmdown', 'final', 'head_std', 'projection_std', 'anvil_lr', 'anvil_wd',
                    'fast_weight', 'polar_margin')
            settings = {k: settings[k] for k in keep}
        return dict(name=self.key, file=refs.TRAINERS[self.key], settings=settings, warmdown=self.warmdown,
                    final=self.final, model=asdict(self.config), **self.extra)


class Backprop:
    def __init__(self, o, reference, device, seed, steps):
        self.o, self.reference, self.device, self.seed, self.steps = o, reference, device, seed, steps
        self.warmdown = reference.warmdown if o['warmdown'] is None else o['warmdown']
        self.finish = self.end_merge if o['end_merge'] else None  # loud/train.py runs it after the last update
        self.reset()

    def reset(self):
        o = self.o
        self.model = self.reference.make_model(self.device, self.seed)
        self.shaper = shaper(self.o, self.reference, self.model, self.seed)  # E8 or E9; None leaves ANVIL II alone
        self.optimizer = (self.reference.make_optimizer(self.model, self.steps) if self.shaper is None else
                          shaped_optimizer(self.reference.module, self.model, self.reference.settings, self.steps,
                                           self.shaper))
        self.params = list(self.model.parameters())
        self.accumulators = [torch.zeros_like(p, dtype=torch.float32) for p in self.params]
        self.labels = rate_labels(self.optimizer, self.reference.settings.lr)
        self.pergrad = PerSequenceGrads(self.model.named_parameters()) if o['per_sequence_grads'] else None
        self.blur = Blur(self.model, o['smooth_tau'], self.seed, self.device) if o['smooth_tau'] else None
        self.blurred = None  # E2: the last update's mean loss at the displaced points
        self.control = StepControl(self.model, self.optimizer, o) if o['step_control'] != 'none' else None
        self.shaping = TokenShaping(self.model, self.accumulators, o) if o['token_shaping'] != 'none' else None

    def describe(self):
        described = self.reference.describe()
        if self.o['warmdown'] is not None:
            described.update(warmdown=self.warmdown, recipe_warmdown=self.reference.warmdown)
        return described

    def loss_of(self, micro):
        """A micro-batch's mean loss, from the bf16 autocast forward every update uses."""
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            return self.model(micro[:, :-1].contiguous(), micro[:, 1:].contiguous())

    def backward(self, micro):
        """One micro-batch's backward, its gradients added to the float32 accumulators. Returns the loss."""
        loss = self.loss_of(micro)
        if isinstance(self.shaper, FisherFactor):  # E9: a backward for sampled labels, before the training one
            self.shaper.observe(micro[:, 1:])
        loss.backward()
        if self.pergrad is not None:
            self.pergrad.capture(self.model)
        pairs = [(a, p.grad.float()) for a, p in zip(self.accumulators, self.params) if p.grad is not None]
        torch._foreach_add_([a for a, _ in pairs], [g for _, g in pairs])
        for p in self.params:
            p.grad = None
        return loss.detach().float()

    def gradient(self, batch):
        """Mean of the micro-batches' float32 gradients, cast to each parameter's dtype (backprop_preset's all-reduce
        and divide by world). Returns the mean training loss (a tensor). With E2's blur, each micro-batch gives two
        gradients, at +r and -r, and the loss returned is the clean one, from a no-grad forward."""
        torch._foreach_zero_(self.accumulators)
        losses, blurred = [], []
        self.model.requires_grad_(True)
        with self.shaping.batch() if self.shaping is not None else contextlib.nullcontext():
            for micro in batch.split(self.o['micro_batch']):
                if self.blur is None:
                    losses.append(self.backward(micro))
                    continue
                losses.append(self.blur.measure(lambda: self.loss_of(micro)))
                for sign in (1, -1):
                    with self.blur.shifted(sign):
                        blurred.append(self.backward(micro))
            if self.shaping is not None:
                self.shaping.finish()
        self.model.requires_grad_(False)
        passes = len(blurred) or len(losses)  # backward passes: two per micro-batch with the blur
        torch._foreach_div_(self.accumulators, passes)
        for a, p in zip(self.accumulators, self.params):
            p.grad = a.to(p.dtype)  # float32 parameters take the accumulator itself; it is zeroed only next update
        if blurred:
            self.blurred = torch.stack(blurred).mean()
        return torch.stack(losses).mean()

    @torch.no_grad()
    def batch_loss(self, batch):
        """The batch's mean loss at the current weights, over the same micro-batches as gradient()."""
        return torch.stack([self.loss_of(micro).float() for micro in batch.split(self.o['micro_batch'])]).mean()

    def update(self, step, batch):
        set_rate(self.optimizer, step, self.steps, self.warmdown, self.reference.final)
        if self.control is not None:
            self.control.scale(self.optimizer)
        fields = current_rates(self.optimizer, self.labels)
        loss = self.gradient(batch)
        if self.control is not None:
            self.control.save()
        if self.shaper is not None:
            self.shaper.before_step(step, self.pergrad)
        self.optimizer.step()
        if self.shaper is not None:
            self.shaper.after_step(self.model, batch, self.pergrad)
        self.optimizer.zero_grad(set_to_none=True)
        if self.control is not None:
            fields.update(self.control.judge(loss, self.batch_loss(batch)))
        return loss, fields

    def diagnose(self, step):
        record = self.shaper.report(step) if self.shaper is not None else {}
        if self.blurred is not None:
            record['blurred_loss'] = self.blurred.item()
        if self.pergrad is not None:
            every = self.o['grad_noise_every']
            if every and step % every == 0:
                record['grad_noise'] = self.pergrad.summary()
            self.pergrad.clear()
        return record

    def end_merge(self, means, heldout):
        """The record's end merge (PR #40's, PR #46's; loud.checkpoints.end_merge), timed with the training as the
        records run it, each candidate scored by H3's one-forward mean; returns its result fields."""
        return checkpoints.end_merge(self.reference.module, self.model, means, heldout, self.steps)

    def assign(self, other):
        """Take another trainer's weights and optimizer state (E3: branching a copy, replacing the worst one)."""
        self.model.load_state_dict(other.model.state_dict())
        for mine, theirs in zip(self.optimizer.optimizers, other.optimizer.optimizers):
            mine.load_state_dict(copy.deepcopy(theirs.state_dict()))  # load_state_dict would alias the tensors
        # load_state_dict replaces the groups: point the wrapper (and set_rate) at the new ones; ANVIL's clock too
        self.optimizer.param_groups = [g for o in self.optimizer.optimizers for g in o.param_groups]
        self.optimizer.optimizers[0].t = other.optimizer.optimizers[0].t


def shaper(o, reference, model, seed):
    """E8's curvature damping or E9's Fisher factor in place of ANVIL II's polar(), or None."""
    if o['curvature_damping']:
        return CurvatureDamping(o, model, reference.settings.anvil_lr, reference.module.polar)
    if o['fisher_damping']:
        return FisherFactor(o, model, seed, reference.module.polar)
    return None


def run(spec, run_dir):
    o = proxy_options(spec.get('args', {}))
    reference = Reference(o['recipe'], o['lr_mult'], o['preset'])
    return train(spec, run_dir, o, lambda device, seed, steps, heldout: Backprop(o, reference, device, seed, steps),
                 method='backprop')
