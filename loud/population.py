"""E3 (v2): population as local SGD. Does any rule over several weight copies beat one trajectory at equal steps and
equal data?

The job runs the batch-8 proxy (e1.bp_b8's trainer: the same code, seed and data, so the same trajectory) to the
branch, 20% of the updates (update 121 of 606 on the 10M set), and branches into K copies (weights and optimizer
state). From then on each update's B sequences go B / K to each copy (2 of 8 for K = 4, in batch order), so every copy
makes every remaining update, at local batch B / K with the baseline's rates and schedule (the same update number, so
ANVIL II's clock and the warmdown are the single run's). Forward and backward compute equal one run; optimizer compute
and state are K times. Each copy keeps its own optimizer state.

Rules (one per arm):
  average   every H updates (`every`), every copy continues from the mean of the copies' weights AND optimizer state
            (ANVIL II's rails and energy, Adam's moments); a last average after the final update. Final: the mean.
            H = 1 is a sanity anchor near the baseline, though not identical, since ANVIL's orthogonalization is
            nonlinear.
  select    no averaging; every H updates the copies are scored on the held-out set (H3) and the worst takes the
            best's weights and optimizer state. Final: the best copy by the held-out set after the last update.
  ensemble  no averaging until the end, then the mean of the copies' weights if H3 scores it below the best copy,
            else the best copy.
Every `midpoints_every` updates after the branch, before the rule acts at that update, population.jsonl records each
copy's held-out loss, each pairwise midpoint's and the barrier, midpoint minus the worse end (above 0: the pair has
left one connected basin), with the rule's action that follows. These are
untimed monitoring; the scores a rule decides on are timed. metrics.jsonl has one record per update, with each copy's
training loss after the branch. H4 (checkpoints): the trunk's fractions up to the branch, then each copy's
(ckpt_<f>.c<k>.pt); final.pt is the rule's choice.
"""
from contextlib import contextmanager
from itertools import combinations
import math
import time

import torch

from .checkpoints import FractionCheckpoints
from .common import JsonlLog
from .data import SEQ, benchmark_split
from .heldout import HeldoutScorer, mean_loss
from .options import branch_step, checkpoints_on, population_options
from .proxy import Backprop, Reference
from .runlog import Run
from .score import score_in_subprocess
from .train import device, shrink_scored


def state_tensors(trainer):
    """A copy's weights and every optimizer state tensor, in one order shared by all copies."""
    held = list(trainer.model.parameters())
    for o in trainer.optimizer.optimizers:
        for p in trainer.model.parameters():
            state = o.state.get(p, {})
            held += [state[k] for k in sorted(state) if torch.is_tensor(state[k])]
    return held


@torch.no_grad()
def average_into(groups):
    """groups[c]: copy c's tensors, in one order. Each tensor becomes the float32 mean over the copies."""
    mean = [t.float().clone() for t in groups[0]]
    for group in groups[1:]:
        torch._foreach_add_(mean, [t.float() for t in group])
    torch._foreach_div_(mean, len(groups))
    for group in groups:
        torch._foreach_copy_(group, mean)


@torch.no_grad()
def load_mean(model, sources):
    """Set the model's parameters to the mean of the sources' (taken in float32)."""
    for p, *qs in zip(model.parameters(), *(s.parameters() for s in sources)):
        p.copy_(torch.stack([q.float() for q in qs]).mean(0))


class Clock:
    """Training seconds from the first update, without the paused (untimed) work, as loud/train.py counts them."""

    def __init__(self, dev):
        self.dev, self.paused_total = dev, 0.0
        self.sync()
        self.start = time.perf_counter()

    def sync(self):
        if self.dev.type == 'cuda':
            torch.cuda.synchronize()

    def elapsed(self):
        return time.perf_counter() - self.start - self.paused_total

    @contextmanager
    def paused(self):
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.paused_total += time.perf_counter() - start


class Population:
    def __init__(self, o, trunk, reference, dev, seed, heldout, clock):
        self.o, self.heldout, self.clock = o, heldout, clock
        self.copies = [trunk]
        for _ in range(o['copies'] - 1):
            twin = Backprop(trunk.o, reference, dev, seed, trunk.steps)
            twin.assign(trunk)
            self.copies.append(twin)
        self.scratch = reference.make_model(dev, seed)  # holds means and midpoints

    def update(self, step, batch):
        """Every copy's update on its share of the batch. Returns (their losses, the first copy's rate fields)."""
        share = len(batch) // len(self.copies)
        out = [c.update(step, batch[k * share:(k + 1) * share]) for k, c in enumerate(self.copies)]
        return torch.stack([loss for loss, _ in out]), out[0][1]

    def average(self):
        average_into([state_tensors(c) for c in self.copies])

    def scores(self, step, reason):
        return [self.heldout.score(c.model, step, reason, copy=k) for k, c in enumerate(self.copies)]

    def select(self, step):
        """The worst copy takes the best's weights and optimizer state (timed: the rule decides on these scores)."""
        scores = self.scores(step, 'select')
        best, worst = scores.index(min(scores)), scores.index(max(scores))
        if scores[best] < scores[worst]:
            self.copies[worst].assign(self.copies[best])
        return dict(select=best, replace=worst, scores=scores)

    def midpoints(self, step):
        """Monitoring (untimed): each copy's held-out loss, each pairwise midpoint's and the barrier."""
        with self.clock.paused():
            copies = self.scores(step, 'monitor')
            midpoints, barriers = {}, {}
            for i, j in combinations(range(len(self.copies)), 2):
                load_mean(self.scratch, [self.copies[i].model, self.copies[j].model])
                pair = f'{i}-{j}'
                midpoints[pair] = self.heldout.score(self.scratch, step, 'monitor', midpoint=pair)
                barriers[pair] = midpoints[pair] - max(copies[i], copies[j])
        return dict(step=step, copies=copies, midpoints=midpoints, barriers=barriers)

    def final(self, step):
        """The run's final weights and how they were chosen, after the last update."""
        rule = self.o['rule']
        if rule == 'average':
            self.average()
            return self.copies[0].model, dict(final='mean')
        scores = self.scores(step, rule)
        best = scores.index(min(scores))
        choice = dict(final_copy_scores=scores)
        if rule == 'ensemble':
            load_mean(self.scratch, [c.model for c in self.copies])
            choice['final_mean_score'] = mean = self.heldout.score(self.scratch, step, 'ensemble', mean=True)
            if mean < scores[best]:
                return self.scratch, dict(choice, final='mean')
        return self.copies[best].model, dict(choice, final=f'copy {best}')


def run(spec, run_dir):
    o = population_options(spec.get('args', {}))
    seed, dev = spec['seed'], device()
    torch.set_float32_matmul_precision('high')
    split = benchmark_split(o['sequences'], o['heldout'])
    B, K = o['batch'], o['copies']
    branch, steps = branch_step(o)
    reference = Reference(o['recipe'], o['lr_mult'], o['preset'])
    trunk = Backprop(o, reference, dev, seed, steps)
    run = Run(run_dir, spec, dict(options=o, method=f'population ({o["rule"]})', trainer=trunk.describe(),
                                  data=split.describe(), steps=steps, branch_step=branch, local_batch=B // K))

    # untimed warmup on random tokens (compiles the update and the scorer's shapes), then a fresh trainer
    held = len(split.heldout)
    noise = torch.randint(0, trunk.model.config.vocab_size, (max(B, held), SEQ),
                          generator=torch.Generator().manual_seed(0)).to(dev)
    trunk.update(1, noise[:B])
    mean_loss(trunk.model, noise[:held], dev, held)
    trunk.reset()

    heldout = HeldoutScorer(split.heldout, run_dir / 'heldout.jsonl', dev)
    events = JsonlLog(run_dir / 'population.jsonl')
    checkpoints = FractionCheckpoints(run_dir, steps, dtype=o['checkpoint_dtype'], enabled=checkpoints_on(o, seed))
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    clock = Clock(dev)
    previous, population = 0.0, None

    def record(step, losses, fields, **extra):
        nonlocal previous
        values = losses.tolist()  # also waits for the update, so the step time is the GPU's
        loss = sum(values) / len(values)
        if not math.isfinite(loss):
            raise FloatingPointError(f'non-finite training loss at update {step}')
        elapsed = clock.elapsed()
        seqs = step * B
        run.metrics.write(dict(step=step, **extra, seqs=seqs, tokens=seqs * (SEQ - 1), train_loss=loss,
                               step_seconds=round(elapsed - previous, 4), train_seconds=round(elapsed, 2), **fields,
                               **(dict(copy_losses=values) if len(values) > 1 else {})))
        previous = elapsed
        if step % 50 == 0:
            print(dict(step=step, train_loss=loss, train_seconds=round(elapsed, 2), **extra), flush=True)
        return loss

    for step in range(0, steps + 1):
        if step:
            batch = split.train[(step - 1) * B:step * B].to(dev)
            if step <= branch:
                loss, fields = trunk.update(step, batch)
                last = record(step, loss.reshape(1), fields, phase='trunk')
            else:
                losses, fields = population.update(step, batch)
                last = record(step, losses, fields, phase='copies')
                after, action = step - branch, None
                # before the rule acts: an average would leave every barrier at 0
                monitor = population.midpoints(step) if after % o['midpoints_every'] == 0 else None
                if o['rule'] == 'average' and after % o['every'] == 0:
                    population.average()
                    action = 'average'
                elif o['rule'] == 'select' and after % o['every'] == 0:
                    action = population.select(step)
                if monitor is not None:
                    events.write(dict(monitor, **({} if action is None else dict(action=action))))
        if step == branch:
            population = Population(o, trunk, reference, dev, seed, heldout, clock)
        if checkpoints.due(step):
            with clock.paused():
                for k, member in enumerate(population.copies if step > branch else [trunk]):
                    checkpoints.save(member.model, step, copy=k if step > branch else None)
    model, choice = population.final(steps)
    clock.sync()
    elapsed = clock.elapsed()
    events.close()

    final = checkpoints.save_final(model)
    run.update_config(checkpoints=checkpoints.saved)
    result = dict(steps=steps, updates=steps, complete=True, train_seconds=round(elapsed, 2),
                  update_sequences=steps * B, train_loss_last=last, rule=o['rule'], every=o['every'], copies=K,
                  branch_step=branch, **choice)
    result['heldout_loss'] = heldout.score(model, steps, 'monitor', final=True)
    if dev.type == 'cuda':
        result['peak_cuda_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    if o['official_eval']:
        del population, trunk, model
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        result.update(score_in_subprocess(final, ['val', 'test'] if o['test'] else ['val'], run_dir / 'score.json'))
        shrink_scored(o, result, [final])
    return run.finish(result, spec.get('expect'))
