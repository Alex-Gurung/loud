"""Task 'compare': check that two runs took the same steps, by their per-update training losses.

Each side is a job id (its finished run directory) or a published log pinned in external.json, [repo, path to a
metrics.jsonl or a printed log of JSON lines]. Used to verify a port against its source: the proxy with whole-batch forwards against the unmodified
script, a forward-only port against the published 8-GPU log of the same seed.

    args:
      a: verify.bp_b8_onefwd.s0
      b: verify.bp_b8_script.s0          # or [noise, archive/10M/train/val/log/metrics.jsonl]
      first: 100                         # the first N steps b logs (0: all of them), each of which a must log too;
                                         # the gap over every common step is reported as train_loss_max_diff_all
    expect: {train_loss_max_diff: [0, 0.002], val_loss_diff: [0, 0.01]}
"""
import json
from pathlib import Path

from .common import read_json
from . import paths, refs
from .runlog import Run


def side(ref):
    """(per-step train losses, result.json or {}) for a job id or a pinned [repo, metrics path]."""
    if isinstance(ref, str):
        experiment, variant, seed = ref.rsplit('.', 2)
        directory = paths.run_dir(experiment, variant, int(seed[1:]))
        result_path = directory / 'result.json'
        if not result_path.exists():
            raise FileNotFoundError(f'{ref} has not finished: no {result_path}')
        metrics, result = directory / 'metrics.jsonl', read_json(result_path)
    else:
        repo, relpath = ref
        metrics = refs.verified(repo, relpath)
        try:  # the published result, when it is pinned too
            result = read_json(refs.verified(repo, str(Path(relpath).with_name('result.json'))))
        except KeyError:
            result = {}
    losses = {}
    for r in records(metrics):
        if 'step' in r and 'train_loss' in r:
            if r['step'] in losses:
                raise ValueError(f'{ref}: step {r["step"]} is logged twice')
            losses[r['step']] = r['train_loss']
    return losses, result


def records(path):
    """The JSON objects in a metrics.jsonl or in a script's printed log (other lines skipped)."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.startswith('{')]


def run_task(spec, run_dir):
    args = spec['args']
    (a, result_a), (b, result_b) = side(args['a']), side(args['b'])
    # the first `first` steps b logs (all of them by default; b may be a sparse published log) must all be in a
    first = args.get('first') or len(b)
    if len(b) < first:
        raise ValueError(f'{args["b"]} logs only {len(b)} steps; first asks for {first}')
    head = sorted(b)[:first]
    missing = [s for s in head if s not in a]
    if missing:
        raise ValueError(f'{args["a"]} lacks steps that {args["b"]} logs: {missing[:10]}')
    common = sorted(set(a) & set(b))
    diffs = [abs(a[s] - b[s]) for s in head]
    result = dict(a=args['a'], b=args['b'], steps_a=len(a), steps_b=len(b), steps_compared=len(head),
                  train_loss_max_diff=max(diffs), train_loss_mean_diff=sum(diffs) / len(diffs),
                  max_diff_step=head[diffs.index(max(diffs))],
                  train_loss_max_diff_all=max(abs(a[s] - b[s]) for s in common), complete=True)
    if 'val_loss' in result_a and 'val_loss' in result_b:
        result['val_loss_diff'] = abs(result_a['val_loss'] - result_b['val_loss'])
    run = Run(run_dir, spec, dict(sides=[args['a'], args['b']]))
    return run.finish(result, spec.get('expect'))
