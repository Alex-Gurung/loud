"""Queue runs for the GPU workers (CPU only; stdlib + PyYAML). Run from the repo root.

    python3 -m loud.submit configs/verify                 # every *.yaml under it (but followups/), every listed seed
    python3 -m loud.submit configs/e1/bp_b8.yaml --seeds 0
    python3 -m loud.submit configs/e1 --dry-run           # what each job would do, checked against the queue
    python3 -m loud.submit configs/e1/bp_b8.yaml --rerun  # supersede finished runs (their directories are kept)
    python3 -m loud.submit --cancel e1.bp_b8.s2 'e13.theta*'   # cancel queued jobs, by id or glob pattern

The code is frozen once per submission (artifacts/code/<digest>, from the checkout this module sits in). Resubmitting
an unchanged config updates a job that has not started (newer code, priority, `after`, `gpu`); finished and cancelled
jobs never change without --rerun; a failed job is queued again. Nothing is written unless every job can be submitted.
"""
import argparse
from collections import Counter
import fnmatch
from pathlib import Path

import yaml

from .jobqueue import STATES, JobQueue, job_id
from .paths import QUEUE, SOURCE, run_dir
from .run import config_hash, specs_from_config
from .snapshot import snapshot, verify


FOLLOWUPS = 'followups'  # configs the plan runs only after reading earlier results: submitted only when named


def config_files(paths):
    """The configs under each path. A directory leaves out its followups/ subdirectories; name one (or a file in
    it) to submit it."""
    out = []
    for p in map(Path, paths):
        if not p.is_dir():
            out.append(p)
            continue
        out += sorted(f for f in p.rglob('*') if f.suffix in ('.yaml', '.yml')
                      and FOLLOWUPS not in f.relative_to(p).parts[:-1])
    return out


def plan(paths, seeds, priority):
    jobs, seen = [], {}
    for path in config_files(paths):
        for spec, meta in specs_from_config(path, seeds):
            jid = job_id(spec['experiment'], spec['variant'], spec['seed'])
            if jid in seen:
                raise SystemExit(f'{jid} comes from both {seen[jid]} and {path}: give each config its own variant')
            seen[jid] = path
            if path.stem != spec['variant']:
                print(f'note: {path} defines variant {spec["variant"]!r}')
            jobs.append(dict(id=jid, spec=dict(spec, config_path=meta['config_path'], notes=meta['notes']),
                             priority=meta['priority'] if priority is None else priority, after=meta['after'],
                             after_done=meta['after_done'],
                             gpu=meta['gpu'], gpu_memory_gb=meta['gpu_memory_gb'], config_path=meta['config_path'],
                             config_hash=config_hash(spec),
                             run_dir=str(run_dir(spec['experiment'], spec['variant'], spec['seed']))))
    if not jobs:
        raise SystemExit('no configs found')
    return jobs


def check_dependencies(jobs, queue):
    """Every `after` id names a job in the queue or in this submission, and no job waits on itself (or a cycle)."""
    deps = lambda j: (j.get('after') or []) + (j.get('after_done') or [])
    after = {j['id']: deps(j) for state in STATES for j in queue.jobs(state)}
    after.update({j['id']: deps(j) for j in jobs})
    for job in jobs:
        missing = [a for a in deps(job) if a not in after]
        if missing:
            raise SystemExit(f'{job["id"]}: after names unknown jobs {missing}')
    for job in jobs:  # depth-first walk from each submitted job
        stack, seen = list(deps(job)), set()
        while stack:
            dep = stack.pop()
            if dep == job['id']:
                raise SystemExit(f'{job["id"]} waits on itself through its after list')
            if dep not in seen:
                seen.add(dep)
                stack.extend(after.get(dep, []))


def described(queue, jid, action):
    """The action, with the reason when the submission leaves the job as it is."""
    if action != 'unchanged':
        return action
    state, _ = queue.find(jid)
    return f'unchanged ({state}; resubmit with --rerun)' if state in ('cancelled', 'done') else f'unchanged ({state})'


def summary(actions):
    return f'{len(actions)} jobs: ' + ', '.join(f'{n} {a}' for a, n in sorted(Counter(actions).items()))


def cancel(queue, names, dry_run):
    """Cancel the queued jobs that each id or glob pattern (on job ids) names; False if one of them named none."""
    queued, seen, ok = sorted(j['id'] for j in queue.jobs('queued')), set(), True
    for name in names:
        matched = [jid for jid in queued if fnmatch.fnmatchcase(jid, name)]
        if not matched:
            ok, state = False, queue.find(name)[0]
            why = ('no queued job matches' if set(name) & set('*?[') else
                   f'{state}; only queued jobs can be cancelled' if state else 'not in the queue')
            print(f'{name}: {why}')
        for jid in (j for j in matched if j not in seen):
            seen.add(jid)
            if dry_run:
                print(f'would cancel {jid}')
                continue
            try:
                queue.cancel(jid)
                print(f'cancelled {jid}')
            except ValueError as exc:  # claimed since the listing
                ok = False
                print(exc)
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configs', nargs='*')
    parser.add_argument('--seeds', help='Comma-separated seeds, overriding each config\'s list.')
    parser.add_argument('--priority', type=int, help='Override each config\'s priority (lower runs first).')
    parser.add_argument('--rerun', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--cancel', nargs='+', metavar='JOB',
                        help="Cancel these queued jobs: ids or quoted glob patterns ('e13.theta*').")
    parser.add_argument('--queue', type=Path, default=QUEUE)
    args = parser.parse_args()
    queue = JobQueue(args.queue, create=not args.dry_run)
    if args.dry_run and not queue.root.exists():
        print(f'note: {queue.root} does not exist yet; read as an empty queue')
    if args.cancel:
        if not cancel(queue, args.cancel, args.dry_run):
            raise SystemExit(1)
        return
    if not args.configs:
        parser.error('give configs to submit, or --cancel')
    seeds = None if args.seeds is None else [int(s) for s in args.seeds.split(',')]
    try:
        jobs = plan(args.configs, seeds, args.priority)
    except (OSError, ValueError, yaml.YAMLError) as exc:  # each names its config file
        raise SystemExit(' '.join(str(exc).split()))
    check_dependencies(jobs, queue)
    actions = []
    if args.dry_run:  # decided against the queue without freezing code, so newer code alone is not shown
        for job in jobs:
            try:
                action = queue.submit(dict(job, code_dir=None, code_digest=None), args.rerun, dry_run=True)
            except ValueError as exc:
                action = f'CONFLICT: {exc}'
            shown = described(queue, job['id'], action)
            actions.append(shown.split(':')[0])
            deps = ''.join(f'  {key} {",".join(job[key])}' for key in ('after', 'after_done') if job[key])
            print(f'{job["id"]:40s} priority {job["priority"]:<4} {shown}{deps}')
        print(summary(actions))
        return
    for job in jobs:  # every conflict is raised before anything is written
        try:
            queue.submit(dict(job, code_dir=None, code_digest=None), args.rerun, dry_run=True)
        except ValueError as exc:
            raise SystemExit(f'{exc} (nothing was submitted)')
    code = snapshot()
    manifest = verify(code)
    for job in jobs:
        job.update(code_dir=str(code), code_digest=manifest['digest'], git=manifest.get('git'))
        actions.append(described(queue, job['id'], queue.submit(job, rerun=args.rerun)))
        print(f'{job["id"]:40s} {actions[-1]}')
    print(f'code {code} (from {SOURCE}, git {manifest["git"]})')
    print(summary(actions))


if __name__ == '__main__':
    main()
