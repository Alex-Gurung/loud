"""A dependency-aware job queue on the shared volume (the stylo request-queue pattern, one training run per job).

    artifacts/queue/{queued,running,done,failed,cancelled}/<job id>.json
    artifacts/queue/workers/<pod>.json        worker heartbeats
    artifacts/queue/history/                  superseded records (never deleted)
    artifacts/queue/bootstrap_logs/<pod>.log  each GPU pod's startup log

A job is one (experiment, variant, seed): id '<experiment>.<variant>.s<seed>', run directory
artifacts/runs/<experiment>/<variant>/s<seed>/. Each attempt writes to its own directory, s<seed>.attempt<k>/, which is
also the attempt's lease: heartbeats and completion must name it. A finished attempt is renamed to s<seed>/ under the
lock, so a stale attempt (its pod presumed dead) can only ever write into its own directory. A running job's lease
expires after LEASE_SECONDS without a heartbeat, and the job is queued again; the MAX_ATTEMPTS-th expiry fails it. A
worker that is stopping (SIGTERM) gives its job back without using up an attempt. A job runs only after every job in
its `after` list is done and did not fail its own check, and every job in `after_done` is done, passed or not. A
record's directory is its state.
"""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import time

from .common import read_json, utc_now, write_json
from .paths import QUEUE

STATES = ('queued', 'running', 'done', 'failed', 'cancelled')
LEASE_SECONDS = 300
MAX_ATTEMPTS = 3
# May change while a job waits (with an equal config hash, the spec differs only in its config_path and notes).
SCHEDULING = ('priority', 'after', 'after_done', 'gpu', 'config_path', 'spec')


class LostLease(RuntimeError):
    pass


def job_id(experiment, variant, seed):
    return f'{experiment}.{variant}.s{seed}'


def queue_order(job):
    """Claim order: priority (lower first), then seed, then submission time. Within a priority every arm's seed 0
    runs before any seed 1, so one GPU gives a first read of every arm early and fills in seeds after."""
    return job.get('priority', 100), job.get('spec', {}).get('seed', 0), job.get('submitted_ts', 0), job['id']


def free_name(path, suffix):
    """path.<suffix><k> for the first k >= 1 that does not exist yet."""
    k = 1
    while (target := path.with_name(f'{path.name}.{suffix}{k}')).exists():
        k += 1
    return target


class JobQueue:
    def __init__(self, root=QUEUE, lease_seconds=LEASE_SECONDS, max_attempts=MAX_ATTEMPTS, create=True):
        self.root = Path(root)
        self.lease_seconds, self.max_attempts = lease_seconds, max_attempts
        if create:
            for name in (*STATES, 'workers', 'history', 'bootstrap_logs'):
                (self.root / name).mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self):
        if not self.root.exists():  # a queue not made yet (a dry run): nothing to guard, and it reads as empty
            yield
            return
        with (self.root / 'lock').open('a+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def path(self, state, jid):
        return self.root / state / f'{jid}.json'

    def find(self, jid):
        for state in STATES:
            try:
                return state, read_json(self.path(state, jid))
            except FileNotFoundError:
                continue
        return None, None

    def jobs(self, state):
        out = []
        for p in sorted((self.root / state).glob('*.json')):
            try:
                out.append(read_json(p))
            except FileNotFoundError:
                continue
        return out

    def _move(self, job, old_state, new_state):
        """Rewrite the record in place, then rename it: a crash leaves it in exactly one state."""
        write_json(self.path(old_state, job['id']), job)
        if old_state != new_state:
            os.replace(self.path(old_state, job['id']), self.path(new_state, job['id']))

    def _archive(self, job, why):
        stamp = time.strftime('%Y%m%dT%H%M%S', time.gmtime())
        write_json(self.root / 'history' / f'{job["id"]}.{stamp}.json', dict(job, archived_because=why))

    # ------------------------------------------------------------------------------------------------ submission

    @staticmethod
    def _decide(state, old, job, rerun):
        """What submitting `job` does: 'queued', 'updated' or 'unchanged'; raises on a conflict."""
        if old is None:
            return 'queued'
        if state == 'running' and rerun:
            raise ValueError(f'{job["id"]} is running; wait for it before rerunning')
        if rerun:
            return 'queued'
        same = old['config_hash'] == job['config_hash']
        if same and state == 'queued':
            code = ('code_dir', 'code_digest') if job.get('code_dir') else ()  # a dry run carries no code
            changed = [k for k in (*code, *SCHEDULING) if old.get(k) != job.get(k)]
            return 'updated' if changed else 'unchanged'
        if state == 'cancelled':
            return 'unchanged'  # a cancelled job comes back only with --rerun, whatever its config
        if same and state in ('running', 'done'):
            return 'unchanged'  # a done job, even one that failed its check, changes only with --rerun
        if not same and state in ('queued', 'running', 'done'):
            raise ValueError(f'{job["id"]} is {state} with a different config; use a new variant name or --rerun')
        return 'queued'  # a failed (crashed) job is queued again

    def submit(self, job, rerun=False, dry_run=False):
        """Queue a job. Returns 'queued', 'updated' (same config; newer code or scheduling, not started), 'unchanged',
        or raises on a conflict. With dry_run, only decides."""
        jid = job['id']
        with self.locked():
            state, old = self.find(jid)
            action = self._decide(state, old, job, rerun)
            if dry_run or action == 'unchanged':
                return action
            if action == 'updated':
                old.update({k: job.get(k) for k in ('code_dir', 'code_digest', 'git', *SCHEDULING)},
                           updated_at=utc_now())
                write_json(self.path('queued', jid), old)
                return action
            if old is not None:
                self._archive(old, 'resubmitted')
                self.path(state, jid).unlink(missing_ok=True)
                run_dir = Path(job['run_dir'])
                if run_dir.exists():
                    job['superseded_run_dir'] = str(run_dir.rename(free_name(run_dir, 'superseded')))
            job.update(attempts=0, expiries=0, submitted_at=utc_now(), submitted_ts=time.time(), history=[])
            write_json(self.path('queued', jid), job)
            return action

    def cancel(self, jid):
        with self.locked():
            state, job = self.find(jid)
            if state != 'queued':
                raise ValueError(f'{jid} is {state}; only queued jobs can be cancelled')
            job['cancelled_at'] = utc_now()
            self._move(job, 'queued', 'cancelled')

    # ----------------------------------------------------------------------------------------------- dependencies

    def finished(self):
        """What dependency_state reads, in one pass: (done id -> record, failed or cancelled id -> that state, queued id
        -> record, a memo of the queued jobs' states)."""
        return ({j['id']: j for j in self.jobs('done')},
                {j['id']: s for s in ('failed', 'cancelled') for j in self.jobs(s)},
                {j['id']: j for j in self.jobs('queued')}, {})

    def dependency_state(self, job, finished=None):
        """('ready', ''), ('waiting', the first unfinished dependency), ('blocked', why) or ('skipped', why).
        Blocked: a dependency failed, was cancelled or failed its own check, or is queued and blocked itself; `why`
        names the chain ('a.s0 <- b.s0 failed its check': this job waits on a.s0, which waits on b.s0). Skipped: a
        `gate` job in `after` failed its check, so the plan skips this job (or a queued dependency is skipped). Pass
        one `finished` (from finished()) to a run of calls: the queue is read once and each job's state found once."""
        finished = finished or self.finished()
        done, stopped, queued, memo = finished
        deps = [(d, True) for d in job.get('after') or []] + [(d, False) for d in job.get('after_done') or []]
        first = {}
        for dep, must_pass in deps:
            if dep in stopped:
                found = 'blocked', f'{dep} {stopped[dep]}'
            elif dep in done:
                if not must_pass or (done[dep].get('result') or {}).get('check') != 'fail':
                    continue
                gate = (done[dep].get('spec') or {}).get('task') == 'gate'
                found = ('skipped', f'gate {dep} says skip') if gate else ('blocked', f'{dep} failed its check')
            elif dep in queued:
                if dep not in memo:
                    memo[dep] = 'blocked', 'a cycle of dependencies'  # submission rejects cycles; never recurse twice
                    memo[dep] = self.dependency_state(queued[dep], finished)
                state, why = memo[dep]
                found = (state, f'{dep} <- {why}') if state in ('blocked', 'skipped') else ('waiting', dep)
            else:
                found = 'waiting', dep  # running (or not in the queue)
            first.setdefault(found[0], found)
        return next((first[s] for s in ('skipped', 'blocked', 'waiting') if s in first), ('ready', ''))

    # ------------------------------------------------------------------------------------------------ workers

    def _requeue(self, job, event, reason):
        job['history'].append(dict(event=event, at=utc_now(), worker=job.pop('worker', None),
                                   attempt=job['attempts'], attempt_dir=job.pop('attempt_dir', None), reason=reason))
        if event == 'lease expired':
            job['expiries'] = job.get('expiries', 0) + 1
        if job.get('expiries', 0) >= self.max_attempts:
            job.update(failed_at=utc_now(), reason=f'lease expired {job["expiries"]} times')
            self._move(job, 'running', 'failed')
        else:
            self._move(job, 'running', 'queued')

    def claim(self, worker_id, gpu=None):
        """The first ready job (by priority, then submission) whose `gpu` requirement, if any, names this worker's GPU."""
        with self.locked():
            now = time.time()
            for job in self.jobs('running'):
                if now - job.get('heartbeat', 0) > self.lease_seconds:
                    self._requeue(job, 'lease expired', 'no heartbeat for the lease time')
            finished = self.finished()
            for job in sorted(finished[2].values(), key=queue_order):
                if self.dependency_state(job, finished)[0] != 'ready':
                    continue
                if job.get('gpu') and job['gpu'] not in (gpu or ''):
                    continue
                attempt_dir = free_name(Path(job['run_dir']), 'attempt')
                attempt_dir.mkdir(parents=True)
                job.update(worker=worker_id, claimed_at=utc_now(), heartbeat=now, attempts=job['attempts'] + 1,
                           attempt_dir=str(attempt_dir))
                job['history'].append(dict(event='claimed', at=job['claimed_at'], worker=worker_id,
                                           attempt=job['attempts'], attempt_dir=job['attempt_dir']))
                self._move(job, 'queued', 'running')
                return job
            return None

    def _own(self, jid, worker_id, attempt_dir):
        try:
            job = read_json(self.path('running', jid))
        except FileNotFoundError:
            raise LostLease(f'{jid} is no longer running') from None
        if job.get('worker') != worker_id or job.get('attempt_dir') != str(attempt_dir):
            raise LostLease(f'{jid} now belongs to {job.get("worker")} ({job.get("attempt_dir")})')
        return job

    def heartbeat(self, jid, worker_id, attempt_dir, progress=None):
        with self.locked():
            job = self._own(jid, worker_id, attempt_dir)
            job['heartbeat'] = time.time()
            if progress:
                job['progress'] = progress
            write_json(self.path('running', jid), job)

    def finish(self, jid, worker_id, attempt_dir, state, **info):
        """Record the attempt's outcome. A done attempt's directory becomes the run directory (an existing one, say
        from a direct run, is set aside as .superseded<k>); a failed attempt's directory stays where it is."""
        assert state in ('done', 'failed')
        with self.locked():
            job = self._own(jid, worker_id, attempt_dir)
            if state == 'done':
                run_dir = Path(job['run_dir'])
                if run_dir.exists():
                    job['superseded_run_dir'] = str(run_dir.rename(free_name(run_dir, 'superseded')))
                Path(attempt_dir).rename(run_dir)
                job.pop('attempt_dir')
            job.update(info, finished_at=utc_now())
            job['history'].append(dict(event=state, at=job['finished_at'], worker=worker_id, attempt=job['attempts']))
            self._move(job, 'running', state)

    def release(self, jid, worker_id, attempt_dir, reason):
        """Give a running job back (the worker is stopping; no attempt used up). Its partial attempt directory stays."""
        with self.locked():
            self._requeue(self._own(jid, worker_id, attempt_dir), 'released', reason)

    def worker_report(self, worker_id, **fields):
        write_json(self.root / 'workers' / f'{worker_id}.json', dict(fields, worker=worker_id, seen=time.time(),
                                                                       seen_at=utc_now()))
