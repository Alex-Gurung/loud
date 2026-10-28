"""The training loop shared by every method (H1 backprop proxy, H5 forward-only ports), with H3 and H4 built in.

A method supplies a trainer with
    model                      the GPT being trained
    reset()                    a fresh model and optimizer from the seed
    update(step, batch)        one update; returns (mean train loss as a tensor, fields for the metrics record)
    diagnose(step)             untimed per-update diagnostics; returns fields for the record
    describe()                 what config.json records about the method
and optionally
    warmup(batch)              the untimed compilation on random tokens (default: update(1, batch))
    finish                     None, or finish(means, heldout): work after the last update of a complete run that
                               is timed with the training and may change the weights (the record's end merge, its
                               scores logged through H3); returns result fields. The weights before it are kept as
                               final_plain.pt and scored too.

The loop follows the record scripts: an untimed warmup on random tokens (compilation), then a fresh trainer;
the clock covers the updates only (diagnostics, held-out scoring and checkpoints are paused out); then final.pt and
the official validation (and test) score in a fresh process. Writes config.json, metrics.jsonl (per update: step,
sequences, tokens, train loss, step time, effective rates), heldout.jsonl, the H4 files and result.json.
"""
import time

import torch

from .checkpoints import MERGE_DECAYS, FractionCheckpoints, TrajectoryMeans
from .data import SEQ, benchmark_split, load_split
from .heldout import HeldoutScorer, mean_loss
from .runlog import Run
from .score import score_in_subprocess


def set_rate(optimizer, step, steps, warmdown, final=0.0):
    """The records' linear warmdown over the last `warmdown` of the updates, down to `final` of each rate (one-based
    steps). With final 0 this is backprop_preset.py's and PR #26's min(1, left / warmdown)."""
    left = (steps - step + 1) / steps
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])
        group['lr'] = group['initial_lr'] * (final + (1 - final) * min(1.0, left / warmdown if warmdown else 1.0))


def rate_labels(optimizer, base_lr):
    """One label per parameter group of the records' Optimizers(ANVIL II, Adam): 'anvil' or 'adam_x<rate multiple>'."""
    anvil, adam = optimizer.optimizers
    return ['anvil'] * len(anvil.param_groups) + [f'adam_x{round(g["lr"] / base_lr)}' for g in adam.param_groups]


def current_rates(optimizer, labels):
    """Metrics fields: the effective rate of each group label, and ANVIL II's current momentum (rail) beta."""
    rates = {}
    for label, group in zip(labels, optimizer.param_groups):
        rates.setdefault(label, group['lr'])
    return dict(lr=rates, anvil_beta=optimizer.optimizers[0].rail_beta())


def device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    dev = torch.device('cuda', 0)
    torch.cuda.set_device(dev)
    return dev


def train(spec, run_dir, o, build, method):
    """Run one training job. `build(device, seed, steps, heldout rows)` returns the trainer."""
    seed, dev = spec['seed'], device()
    torch.set_float32_matmul_precision('high')
    split = benchmark_split(o['sequences'], o['heldout'])
    steps = len(split.train) // o['batch']
    last = min(steps, o['stop_after'] or steps)
    trainer = build(dev, seed, steps, split.heldout)
    run = Run(run_dir, spec, dict(options=o, method=method, trainer=trainer.describe(), data=split.describe(),
                                  steps=steps, stop_after=last))

    # untimed warmup on random tokens (compiles the update and the scorer's shapes), then a fresh trainer
    held = len(split.heldout)
    noise = torch.randint(0, trainer.model.config.vocab_size, (max(o['batch'], held), SEQ),
                          generator=torch.Generator().manual_seed(0)).to(dev)
    if hasattr(trainer, 'warmup'):
        trainer.warmup(noise[:o['batch']])
    else:
        trainer.update(1, noise[:o['batch']])
    if held:
        mean_loss(trainer.model, noise[:held], dev, held)
    trainer.reset()

    heldout = HeldoutScorer(split.heldout, run_dir / 'heldout.jsonl', dev)
    checkpoints = FractionCheckpoints(run_dir, steps, dtype=o['checkpoint_dtype'], enabled=o['checkpoints'])
    means = TrajectoryMeans(len(split.train), enabled=o['means'])
    val_rows = load_split('val.pt') if o['eval_every'] else None

    def paused(step):
        """Untimed work after an update (or at the initialization, step 0)."""
        record = trainer.diagnose(step) if step else {}
        if checkpoints.due(step):
            optimizer = trainer.optimizer if o['checkpoint_optimizer'] else None
            record['checkpoint'] = checkpoints.save(trainer.model, step, optimizer).name
        due = checkpoints.due(step) or (o['heldout_every'] and step % o['heldout_every'] == 0)
        if heldout and due and step < last:  # the last update is scored once, below
            record['heldout_loss'] = heldout.score(trainer.model, step, 'monitor')
        if val_rows is not None and step and step % o['eval_every'] == 0 and step < steps:
            record['val_loss'] = mean_loss(trainer.model, val_rows, dev, 16)
        return record

    if dev.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    paused(0)
    start, pause_total = time.perf_counter(), 0.0  # the clock starts at the first update, as in the benchmark
    elapsed = previous = 0.0
    for step in range(1, last + 1):
        first = (step - 1) * o['batch']
        batch = split.train[first:first + o['batch']].to(dev)
        loss, fields = trainer.update(step, batch)
        loss = loss.item()  # also waits for the update, so the step time is the GPU's
        elapsed = time.perf_counter() - start - pause_total
        record = dict(step=step, seqs=first + o['batch'], tokens=(first + o['batch']) * (SEQ - 1), train_loss=loss,
                      step_seconds=round(elapsed - previous, 4), train_seconds=round(elapsed, 2), **fields)
        previous = elapsed
        pause = time.perf_counter()
        means.observe(trainer.model, step, first, last)
        record.update(paused(step))
        if dev.type == 'cuda':
            torch.cuda.synchronize()  # the paused work's GPU tail, or the next update's clock would absorb it
            record['peak_cuda_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        run.metrics.write(record)
        if step % 50 == 0 or step == last:
            print({k: record[k] for k in ('step', 'train_loss', 'train_seconds')}, flush=True)
        pause_total += time.perf_counter() - pause

    complete = last == steps
    finish, plain, finished = getattr(trainer, 'finish', None), None, {}
    if complete and finish is not None:
        plain = checkpoints.save_final(trainer.model, 'final_plain.pt')
        begin = time.perf_counter()
        finished = finish(means, heldout)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - begin
    final = checkpoints.save_final(trainer.model)
    result = dict(steps=steps, updates=last, complete=complete, train_seconds=round(elapsed, 2),
                  update_sequences=last * o['batch'], train_loss_last=loss, **finished)
    if heldout:
        result['heldout_loss'] = heldout.score(trainer.model, last, 'monitor', final=True)
    if means.means is not None:
        result['means_updates'] = len(means.updates)
        if o['save_means']:
            means.save(run_dir / 'means.pt', trainer.model, checkpoints.dtype)
        if heldout:  # final.pt is saved, so the model can hold each mean in turn
            result['means_heldout'] = {}
            buffers = {n for n, _ in trainer.model.named_buffers()}
            for decay, state in zip(MERGE_DECAYS, means.state_dicts(trainer.model)):
                loaded = trainer.model.load_state_dict(state, strict=False)
                if loaded.unexpected_keys or set(loaded.missing_keys) - buffers:
                    raise RuntimeError(f'a mean does not match the model: {loaded}')
                result['means_heldout'][str(decay)] = heldout.score(trainer.model, last, 'monitor', mean_decay=decay)
    if dev.type == 'cuda':
        result['peak_cuda_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    run.update_config(checkpoints=checkpoints.saved)
    if o['official_eval'] and complete:
        del trainer
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        splits = ['val', 'test'] if o['test'] else ['val']
        result.update(score_in_subprocess(final, splits, run_dir / 'score.json'))
        if plain is not None:
            scores = score_in_subprocess(plain, splits, run_dir / 'score_plain.json')
            result.update({f'{k}_plain': v for k, v in scores.items()})
    return run.finish(result, spec.get('expect'))
