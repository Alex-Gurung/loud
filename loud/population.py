"""E3: a population of copies of the backprop proxy on split data, at the compute of one run.

The job first runs the batch-4 baseline (e1.bp_b4's trainer, the same code and seed, so the same trajectory) for its
first round(branch * steps) updates (branch 0.2: the plan's branch point; copies branched from the initialization do
not average well). It branches there into K copies (weights and optimizer state). Copy c receives every K-th of the
remaining sequences in the benchmark's order (rest[c::K]) and updates on batches of B from its share; the copies take
turns (round r: copy 0, 1, ..., K - 1), so together they use each sequence at most once (the up to B - 1 sequences
left in a share feed nothing) and the total compute is one run's.

Schedule: the copies' updates count as the single run's updates in turn order: copy c's r-th update is global update
prefix + (r - 1) K + c + 1, which sets its rate (the warmdown over the same horizon as the single run, so the copies'
warmdown ends where the single run's does) and ANVIL II's clock (its rail-beta ramp and engage step).

Rules (one per variant), applied after every N = `every` rounds and at the end:
  average   every copy continues from the mean of the copies' weights; each keeps its own optimizer state (momenta
            are not averaged: they are each copy's record of its own data). Final: the mean.
  select    the copies are scored on the held-out set (H3); the worst takes the best's weights and optimizer state.
            Final: the best copy.
  ensemble  the copies run independently; `every` only sets the logging. Final: the mean if H3 scores it below the
            best copy, else the best copy.
Every N rounds (before the rule acts) population.jsonl records each copy's H3 score and each pairwise midpoint's, and
the barrier of each pair, midpoint minus the worse end (above 0: the pair has left one basin). These scores are
untimed monitoring, except the ones a rule decides on. metrics.jsonl has one record per update, naming its copy.
H4 (checkpoints): the trunk's fractions before the branch (ckpt_00, _10, _20), then each copy's (ckpt_50.c<k>,
ckpt_100.c<k>) after the round in which the copies pass that fraction of the single run's updates (the copies' last
updates fall just short of the run's last, so 100% is their end); final.pt is the rule's choice. No trajectory means:
after the branch there is no single trajectory, and the rules are themselves E3's merges.
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
from .options import branch_rows, population_options
from .proxy import Backprop, Reference
from .runlog import Run
from .score import score_in_subprocess
from .train import device


def shares(count, copies, batch):
    """Indices of the sequences after the branch: copy c's rest[c::copies], cut to whole batches, and the unused."""
    index = torch.arange(count)
    per = [index[c::copies] for c in range(copies)]
    rounds = min(len(p) for p in per) // batch
    used = [p[:rounds * batch] for p in per]
    return used, torch.cat([p[rounds * batch:] for p in per]), rounds


def global_step(prefix, copies, r, c):
    """The single run's update number that copy c's r-th update counts as (one-based)."""
    return prefix + (r - 1) * copies + c + 1


@torch.no_grad()
def load_mean(model, sources):
    """Set the model's parameters to the mean of the sources' (taken in float32)."""
    for p, *qs in zip(model.parameters(), *(s.parameters() for s in sources)):
        p.copy_(torch.stack([q.float() for q in qs]).mean(0))


@torch.no_grad()
def load_weights(model, source):
    for p, q in zip(model.parameters(), source.parameters()):
        p.copy_(q)


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
    def __init__(self, o, trunk, reference, dev, seed, heldout, events, clock):
        self.o, self.heldout, self.events, self.clock = o, heldout, events, clock
        self.copies = [trunk]
        for _ in range(o['copies'] - 1):
            twin = Backprop(trunk.o, reference, dev, seed, trunk.steps)
            twin.assign(trunk)
            self.copies.append(twin)
        self.scratch = reference.make_model(dev, seed)  # holds means and midpoints

    def models(self):
        return [c.model for c in self.copies]

    def scores(self, step, reason):
        return [self.heldout.score(c.model, step, reason, copy=i) for i, c in enumerate(self.copies)]

    def checkpoint(self, r, step):
        """After every N rounds and at the end: log the copies and midpoints, then apply the rule."""
        rule = self.o['rule']
        if rule == 'select':
            copies = self.scores(step, 'select')
        else:
            with self.clock.paused():
                copies = self.scores(step, 'monitor')
        midpoints, barriers = {}, {}
        with self.clock.paused():
            for i, j in combinations(range(len(self.copies)), 2):
                load_mean(self.scratch, [self.copies[i].model, self.copies[j].model])
                pair = f'{i}-{j}'
                midpoints[pair] = self.heldout.score(self.scratch, step, 'monitor', midpoint=pair)
                barriers[pair] = midpoints[pair] - max(copies[i], copies[j])
        event = dict(round=r, step=step, copies=copies, midpoints=midpoints, barriers=barriers)
        if rule == 'average':
            load_mean(self.scratch, self.models())
            for c in self.copies:
                load_weights(c.model, self.scratch)
            event['action'] = 'average'
        elif rule == 'select':
            best, worst = copies.index(min(copies)), copies.index(max(copies))
            if copies[best] < copies[worst]:
                self.copies[worst].assign(self.copies[best])
            event['action'] = dict(select=best, replace=worst)
        self.events.write(event)
        return copies

    def final(self, step, copies):
        """The run's final weights and how they were chosen, after the last round's checkpoint()."""
        rule = self.o['rule']
        if rule == 'average':  # the copies were just averaged
            return self.copies[0].model, dict(final='mean')
        if rule == 'select':  # the scores taken before the last selection still name the best copy
            best = copies.index(min(copies))
            return self.copies[best].model, dict(final=f'copy {best}', final_copy_scores=copies)
        copies = self.scores(step, 'ensemble')
        load_mean(self.scratch, self.models())
        mean = self.heldout.score(self.scratch, step, 'ensemble', mean=True)
        best = copies.index(min(copies))
        choice = dict(final_copy_scores=copies, final_mean_score=mean)
        if mean < copies[best]:
            return self.scratch, dict(choice, final='mean')
        return self.copies[best].model, dict(choice, final=f'copy {best}')


def run(spec, run_dir):
    o = population_options(spec.get('args', {}))
    seed, dev = spec['seed'], device()
    torch.set_float32_matmul_precision('high')
    split = benchmark_split(o['sequences'], o['heldout'])
    B, K = o['batch'], o['copies']
    steps = len(split.train) // B  # the single run's updates: the schedule's horizon
    prefix, rest = branch_rows(o)
    assert rest == len(split.train) - prefix * B
    used, unused, rounds = shares(rest, K, B)
    rows = [split.train[prefix * B:][u] for u in used]
    reference = Reference(o['recipe'], o['lr_mult'], o['preset'])
    trunk = Backprop(o, reference, dev, seed, steps)
    run = Run(run_dir, spec, dict(options=o, method=f'population ({o["rule"]})', trainer=trunk.describe(),
                                  data=split.describe(), steps=steps, branch_step=prefix, rounds=rounds,
                                  updates=prefix + rounds * K, unused_sequences=len(unused)))

    # untimed warmup on random tokens (compiles the update and the scorer's shapes), then a fresh trainer
    held = len(split.heldout)
    noise = torch.randint(0, trunk.model.config.vocab_size, (max(B, held), SEQ),
                          generator=torch.Generator().manual_seed(0)).to(dev)
    trunk.update(1, noise[:B])
    mean_loss(trunk.model, noise[:held], dev, held)
    trunk.reset()

    heldout = HeldoutScorer(split.heldout, run_dir / 'heldout.jsonl', dev)
    events = JsonlLog(run_dir / 'population.jsonl')
    checkpoints = FractionCheckpoints(run_dir, steps, dtype=o['checkpoint_dtype'], enabled=o['checkpoints'])
    pending = sorted(s for s in checkpoints.at if s > prefix) if o['checkpoints'] else []
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    clock = Clock(dev)
    previous = 0.0

    def record(step, loss, fields, **extra):
        nonlocal previous
        loss = loss.item()  # also waits for the update, so the step time is the GPU's
        if not math.isfinite(loss):
            raise FloatingPointError(f'non-finite training loss at update {step}')
        elapsed = clock.elapsed()
        seqs = step * B
        run.metrics.write(dict(step=step, **extra, seqs=seqs, tokens=seqs * (SEQ - 1), train_loss=loss,
                               step_seconds=round(elapsed - previous, 4), train_seconds=round(elapsed, 2), **fields))
        previous = elapsed
        if step % 50 == 0:
            print(dict(step=step, train_loss=loss, train_seconds=round(elapsed, 2), **extra), flush=True)
        return loss

    for step in range(0, prefix + 1):
        if step:
            batch = split.train[(step - 1) * B:step * B].to(dev)
            record(step, *trunk.update(step, batch), phase='prefix')
        if checkpoints.due(step):
            with clock.paused():
                checkpoints.save(trunk.model, step)
    population = Population(o, trunk, reference, dev, seed, heldout, events, clock)
    for r in range(1, rounds + 1):
        losses = []
        for c, member in enumerate(population.copies):
            step = global_step(prefix, K, r, c)
            member.optimizer.optimizers[0].t = step - 1  # ANVIL II's clock follows the single run's
            batch = rows[c][(r - 1) * B:r * B].to(dev)
            losses.append(record(step, *member.update(step, batch), phase='population', copy=c, round=r))
        while pending and (pending[0] <= step or r == rounds):
            with clock.paused():
                for c, member in enumerate(population.copies):
                    checkpoints.save(member.model, pending[0], copy=c)
            pending.pop(0)
        if r % o['every'] == 0 or r == rounds:
            scores = population.checkpoint(r, step)
    model, choice = population.final(step, scores)
    clock.sync()
    elapsed = clock.elapsed()
    events.close()

    final = checkpoints.save_final(model)
    run.update_config(checkpoints=checkpoints.saved)
    result = dict(steps=steps, updates=prefix + rounds * K, complete=True, train_seconds=round(elapsed, 2),
                  update_sequences=(prefix + rounds * K) * B, train_loss_last=sum(losses) / len(losses),
                  rule=o['rule'], every=o['every'], copies=K, branch_step=prefix, **choice)
    result['heldout_loss'] = heldout.score(model, step, 'monitor', final=True)
    if dev.type == 'cuda':
        result['peak_cuda_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    if o['official_eval']:
        del population, trunk, model
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        result.update(score_in_subprocess(final, ['val', 'test'] if o['test'] else ['val'], run_dir / 'score.json'))
    return run.finish(result, spec.get('expect'))
