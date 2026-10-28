"""Queue runs for the GPU workers (CPU only; stdlib + PyYAML). Run from the repo root.

    python3 -m loud.submit configs/verify                 # every *.yaml under it, every listed seed
    python3 -m loud.submit configs/e1/bp_b8.yaml --seeds 0
    python3 -m loud.submit configs/e1 --dry-run           # what each job would do, checked against the queue
    python3 -m loud.submit configs/e1/bp_b8.yaml --rerun  # supersede finished runs (their directories are kept)
    python3 -m loud.submit --cancel e1.bp_b8.s2           # cancel a queued job

The code is frozen once per submission (artifacts/code/<digest>, from the checkout this module sits in). Resubmitting
an unchanged config updates a job that has not started (newer code, priority, `after`, `gpu`); finished and cancelled
jobs never change without --rerun; a failed job is queued again. Nothing is written unless every job can be submitted.
"""
import argparse
from pathlib import Path

from .jobqueue import STATES, JobQueue, job_id
from .paths import QUEUE, SOURCE, run_dir
from .run import config_hash, specs_from_config
from .snapshot import snapshot, verify


def config_files(paths):
    out = []
    for p in map(Path, paths):
        out += sorted(f for f in p.rglob('*') if f.suffix in ('.yaml', '.yml')) if p.is_dir() else [p]
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
                             gpu=meta['gpu'], config_path=meta['config_path'], config_hash=config_hash(spec),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configs', nargs='*')
    parser.add_argument('--seeds', help='Comma-separated seeds, overriding each config\'s list.')
    parser.add_argument('--priority', type=int, help='Override each config\'s priority (lower runs first).')
    parser.add_argument('--rerun', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--cancel', nargs='+', metavar='JOB', help='Cancel these queued jobs.')
    parser.add_argument('--queue', type=Path, default=QUEUE)
    args = parser.parse_args()
    queue = JobQueue(args.queue, create=not args.dry_run)
    if args.cancel:
        for jid in args.cancel:
            queue.cancel(jid)
            print(f'cancelled {jid}')
        return
    if not args.configs:
        parser.error('give configs to submit, or --cancel')
    seeds = None if args.seeds is None else [int(s) for s in args.seeds.split(',')]
    jobs = plan(args.configs, seeds, args.priority)
    check_dependencies(jobs, queue)
    if args.dry_run:  # decided against the queue without freezing code, so newer code alone is not shown
        for job in jobs:
            try:
                action = queue.submit(dict(job, code_dir=None, code_digest=None), args.rerun, dry_run=True)
            except ValueError as exc:
                action = f'CONFLICT: {exc}'
            print(f'{job["id"]:36s} priority {job["priority"]:<4} after {",".join(job["after"]) or "-":40s} {action}')
        return
    for job in jobs:  # every conflict is raised before anything is written
        queue.submit(dict(job, code_dir=None, code_digest=None), args.rerun, dry_run=True)
    code = snapshot()
    manifest = verify(code)
    for job in jobs:
        job.update(code_dir=str(code), code_digest=manifest['digest'], git=manifest.get('git'))
        print(f'{queue.submit(job, rerun=args.rerun):9s} {job["id"]}')
    print(f'code {code} (from {SOURCE}, git {manifest["git"]})')


if __name__ == '__main__':
    main()
