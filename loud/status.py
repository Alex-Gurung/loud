"""Queue, worker and disk status (CPU only, read-only).

    python3 -m loud.status            # workers, running, queued, failed, cancelled, the last finished
    python3 -m loud.status --done 50
"""
import argparse
from pathlib import Path
import re
import shutil
import time

from .jobqueue import LEASE_SECONDS, JobQueue, queue_order
from .paths import QUEUE, REPO
from .worker import last_line

RECENT = 24 * 3600  # older worker records and startup failures are not shown


def age(seconds):
    seconds = int(seconds)
    return f'{seconds}s' if seconds < 120 else f'{seconds // 60}m' if seconds < 7200 else f'{seconds // 3600}h'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--queue', type=Path, default=QUEUE)
    parser.add_argument('--done', type=int, default=15)
    args = parser.parse_args()
    q = JobQueue(args.queue, create=False)
    now = time.time()
    disk = shutil.disk_usage(REPO)
    print(f'shared disk: {disk.free / 1e9:.1f} GB free of {disk.total / 1e9:.0f} GB (workers pause below '
          f'LOUD_MIN_FREE_GB, default 2)')

    workers = [w for w in q.jobs('workers') if now - w['seen'] < RECENT]
    print(f'\nworkers seen in the last day ({len(workers)}):')
    for w in sorted(workers, key=lambda w: -w['seen']):
        stale = now - w['seen']
        state = w['state'] if stale <= LEASE_SECONDS or w['state'].startswith('exited') else f'LOST ({w["state"]})'
        print(f'  {w["worker"]:44s} {state:36s} {w.get("job") or "":34s} seen {age(stale)} ago  '
              f'gpu {w.get("gpu")}  done {w.get("completed", 0)}')
    for log in sorted((q.root / 'bootstrap_logs').glob('*.log')):
        found = re.search(r'startup exit=(\d+) phase=(\S+)', last_line(log) or '')
        if found and found.group(1) != '0' and now - log.stat().st_mtime < RECENT:
            print(f'  {log.stem:44s} STARTUP FAILED in phase {found.group(2)} (exit {found.group(1)}): {log}')

    running = q.jobs('running')
    print(f'\nrunning ({len(running)}):')
    for j in running:
        p = j.get('progress') or {}
        beat = now - j.get('heartbeat', now)
        note = '  LEASE EXPIRED: requeued at the next claim' if beat > LEASE_SECONDS else ''
        print(f'  {j["id"]:40s} {j.get("worker", ""):44s} attempt {j["attempts"]}  step {p.get("step", "-")}  '
              f'loss {p.get("train_loss", "-")}  heartbeat {age(beat)} ago{note}')
        print(f'  {"":40s} {j.get("attempt_dir", "")}')

    finished = q.finished()
    done = finished[0]
    failed = {j['id']: j for j in q.jobs('failed')}
    cancelled = sorted(k for k, s in finished[1].items() if s == 'cancelled')
    queued = sorted(q.jobs('queued'), key=queue_order)
    print(f'\nqueued ({len(queued)}):')
    for j in queued:
        state, why = q.dependency_state(j, finished)
        gpu = f'  needs {j["gpu"]}' if j.get('gpu') else ''
        print(f'  {j["id"]:40s} priority {j.get("priority", 100):4d}  {state}{": " + why if why else ""}{gpu}')

    print(f'\nfailed ({len(failed)}):')
    for j in failed.values():
        last = (j.get('log_tail') or [''])[-1][:100]
        print(f'  {j["id"]:40s} exit {j.get("exit_code")}  {j.get("reason", "")}  {last}')
        if j.get('attempt_dir'):
            print(f'  {"":40s} log: {j["attempt_dir"]}/stdout.log')
    if cancelled:
        print(f'\ncancelled ({len(cancelled)}): {", ".join(cancelled)}')

    recent = sorted(done.values(), key=lambda j: j.get('finished_at', ''))[-args.done:]
    print(f'\ndone (last {len(recent)} of {len(done)}):')
    for j in recent:
        r = j.get('result') or {}
        val = r.get('val_loss')
        print(f'  {j["id"]:40s} val {"-" if val is None else f"{val:.5f}"}  check {r.get("check") or "-"}  '
              f'train {r.get("train_seconds")} s  {j.get("finished_at")}')


if __name__ == '__main__':
    main()
