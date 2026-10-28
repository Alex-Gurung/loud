"""H7 runner: launch one run from a config (directly, on a GPU you already have) or from a queued job (the worker).

    python run.py --config configs/e1/bp_b8.yaml --seed 0          # direct
    python -m loud.run --job <attempt dir>/job.json                # what the worker runs

A direct run works in artifacts/runs/<experiment>/<variant>/s<seed>.direct<k>/. When it finishes, it becomes s<seed>/
(and enters the results table) unless that directory exists or the queue knows the job; then it stays where it is.

Config layout (YAML):
    experiment: e1              # results-table grouping
    variant: bp_b8              # one change per variant
    task: proxy                 # see loud/options.py
    seeds: [0, 1, 2]            # or seed: 0 (the default when neither is given)
    priority: 50                # lower runs first (default 100)
    after: [verify.bp_b8.s0]    # job ids that must be done (and pass their check) first; '{seed}' is substituted
    after_done: [...]           # job ids that must be done, passed or not (a comparison that explains a failed gate)
    gpu: H100                   # optional: only workers whose GPU name contains this run it (comparable wall-clocks)
    gpus: 8                     # optional (task reference only): only a worker with this many GPUs runs it (default 1)
    args: {...}                 # task options
    expect: {val_loss: [3.708, 0.01]}   # optional automatic check, recorded in result.json
    notes: ...                  # free text, kept with the job
"""
import argparse
import importlib
import json
import os
from pathlib import Path

from .common import digest, load_config, read_json
from .jobqueue import JobQueue, free_name, job_id
from .options import TASKS, VALIDATE, check_expect
from .paths import QUEUE, REPO, run_dir as default_run_dir

KEYS = {'experiment', 'variant', 'task', 'seed', 'seeds', 'priority', 'after', 'after_done', 'gpu', 'gpus', 'args',
        'expect', 'notes'}


def specs_from_config(path, seeds=None):
    """[(spec, meta)] for each seed: spec is what the task runs (and what the config hash covers), meta schedules it."""
    config = load_config(path)
    try:
        return _specs(config, path, seeds)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{path}: {exc}') from None


def _specs(config, path, seeds):
    if not isinstance(config, dict):
        raise ValueError('a config is a mapping of keys')
    unknown = set(config) - KEYS
    if unknown:
        raise ValueError(f'unknown keys {sorted(unknown)}; allowed: {sorted(KEYS)}')
    if 'seed' in config and 'seeds' in config:
        raise ValueError('give seed or seeds, not both')
    for key in ('experiment', 'variant', 'task'):
        if not config.get(key):
            raise ValueError(f'missing {key}')
    for name in ('experiment', 'variant'):
        if not all(c.isalnum() or c in '-_' for c in config[name]):
            raise ValueError(f'{name} may contain only letters, digits, - and _')
    task, args = config['task'], config.get('args') or {}
    if task not in TASKS:
        raise ValueError(f'unknown task {task!r}; have {sorted(TASKS)}')
    VALIDATE[task](args)
    check_expect(task, config.get('expect'))
    seeds = seeds if seeds is not None else config.get('seeds', [config.get('seed', 0)])
    if not isinstance(seeds, list) or not seeds or not all(isinstance(s, int) for s in seeds):
        raise ValueError('seeds must be a non-empty list of integers')
    if task == 'reference' and len(seeds) > 1 and not (args.get('script_seed')
                                                        or any('{seed}' in a for a in args.get('argv', []))):
        raise ValueError('a reference script gets the seed only through {seed} in argv or script_seed; without it, '
                         'seeds are copies')
    after, after_done = config.get('after') or [], config.get('after_done') or []
    for key, ids in (('after', after), ('after_done', after_done)):
        if not isinstance(ids, list) or not all(isinstance(a, str) for a in ids):
            raise ValueError(f'{key} must be a list of job ids')
    priority, gpu = config.get('priority', 100), config.get('gpu')
    if not isinstance(priority, int) or isinstance(priority, bool):
        raise ValueError('priority must be an integer')
    if gpu is not None and (not isinstance(gpu, str) or not gpu):
        raise ValueError('gpu must be part of a GPU name, e.g. H100')
    gpus = config.get('gpus', 1)
    if not isinstance(gpus, int) or isinstance(gpus, bool) or gpus not in (1, 2, 4, 8):
        raise ValueError('gpus must be 1, 2, 4 or 8')
    if gpus > 1 and task != 'reference':
        raise ValueError(f'gpus > 1 runs a reference script under torchrun; task {task} uses one GPU')
    try:
        rel = str(Path(path).resolve().relative_to(REPO.resolve()))
    except ValueError:
        rel = str(path)
    out = []
    for seed in seeds:
        spec = dict(experiment=config['experiment'], variant=config['variant'], seed=seed, task=task, args=args,
                    expect=config.get('expect'), gpus=gpus)
        meta = dict(config_path=rel, notes=config.get('notes'), priority=priority,
                    after=[a.format(seed=seed) for a in after], after_done=[a.format(seed=seed) for a in after_done],
                    gpu=gpu)
        out.append((spec, meta))
    return out


def config_hash(spec):
    """What identifies a job's config. gpus enters only above 1, so one-GPU jobs keep the hashes they were queued
    with before multi-GPU jobs existed."""
    keys = ('experiment', 'variant', 'seed', 'task', 'args', 'expect') + (('gpus',) if spec.get('gpus', 1) > 1 else ())
    return digest({k: spec[k] for k in keys})


def execute(spec, run_dir):
    module, name = TASKS[spec['task']]
    return getattr(importlib.import_module(module), name)(spec, Path(run_dir))


def direct(config, seed, out):
    specs = specs_from_config(config, None if seed is None else [seed])
    if len(specs) != 1:
        raise SystemExit(f'{config} lists seeds {[s["seed"] for s, _ in specs]}; pick one with --seed')
    spec, meta = specs[0]
    spec = dict(spec, config_path=meta['config_path'], notes=meta['notes'])
    if 'LOUD_CODE_DIR' not in os.environ:  # direct run from a checkout: freeze the code anyway
        from .snapshot import snapshot
        os.environ['LOUD_CODE_DIR'] = str(snapshot())
    if out:
        return execute(spec, out)
    final = default_run_dir(spec['experiment'], spec['variant'], spec['seed'])
    work = free_name(final, 'direct')
    result = execute(spec, work)
    jid = job_id(spec['experiment'], spec['variant'], spec['seed'])
    state, _ = JobQueue(QUEUE, create=False).find(jid)
    if final.exists() or state is not None:
        print(f'kept in {work}: {final} exists or the queue holds {jid} ({state})', flush=True)
    else:
        work.rename(final)
        print(f'saved as {final}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--out', type=Path, help='Run directory for a direct run (it stays there).')
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    if args.job:
        job = read_json(args.job)
        result = execute(job['spec'], job['attempt_dir'])
    elif args.config:
        result = direct(args.config, args.seed, args.out)
    else:
        raise SystemExit('give --config or --job')
    print(json.dumps(result, default=str), flush=True)


if __name__ == '__main__':
    main()
