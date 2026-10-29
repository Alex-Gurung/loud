"""The GPU worker: one job at a time on this pod's GPU, from the shared queue. Started by worker/launch_worker.sh inside
the Kubernetes Job (worker/gpu_worker.yaml); never started by the harness itself.

Each attempt runs as a child process from the code snapshot recorded at submission (`python -m loud.run --job ...`),
in its own directory (s<seed>.attempt<k>/, stdout to stdout.log there). While it runs, the worker renews the lease and
reports the last metrics line; when it succeeds, the queue renames the attempt directory to the run directory.
On SIGTERM (pod deletion or preemption) the child is stopped and the job is given back to the queue. With nothing to
run for --idle-exit seconds the worker exits cleanly: EIDF reclaims GPUs that sit idle for about 30 minutes anyway.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from .common import read_json, utc_now, write_json
from .drain import is_on
from .jobqueue import JobQueue, LostLease
from .paths import QUEUE, REPO
from .snapshot import verify

HEARTBEAT_SECONDS = 30


def last_line(path, limit=8192):
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - limit))
            lines = f.read().decode(errors='replace').strip().splitlines()
        return lines[-1] if lines else None
    except FileNotFoundError:
        return None


def tail(path, lines=40):
    try:
        return Path(path).read_text(errors='replace').splitlines()[-lines:]
    except FileNotFoundError:
        return []


def progress_of(run_dir):
    line = last_line(Path(run_dir) / 'metrics.jsonl')
    if not line:
        return None
    try:
        record = json.loads(line)
    except ValueError:
        return None
    return {k: record[k] for k in ('step', 'train_loss', 'train_seconds') if k in record}


def gpu_memory_gb():
    """The total memory of the GPU this worker is pinned to (nvidia-smi), or None."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')[0].strip()
    for line in out.stdout.splitlines():
        index, uuid, total = (x.strip() for x in line.split(',')[:3]) if line.count(',') >= 2 else ('', '', '')
        if total and visible in ('', index, uuid):
            return float(total) / 1024
    return None


def gpu_name():
    """The name of the GPU this worker's jobs run on: the first nvidia-smi lists, or the first of
    CUDA_VISIBLE_DEVICES (indices or UUIDs) when that pins the worker to some of the pod's GPUs (gpu_worker_2.yaml);
    None without nvidia-smi (a CPU check of the plumbing)."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    rows = [line.split(', ', 2) for line in out.stdout.splitlines() if line.count(', ') >= 2]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')[0].strip()
    for index, uuid, name in rows:
        if visible in ('', index, uuid):
            return name.strip()
    return None


class Worker:
    def __init__(self, queue, worker_id, idle_exit, poll, min_free_gb, python):
        self.queue, self.worker_id = queue, worker_id
        self.idle_exit, self.poll, self.min_free_gb, self.python = idle_exit, poll, min_free_gb, python
        self.stopping = None
        self.child = None
        self.gpu = gpu_name()
        # workers sharing one GPU (LOUD_JOBS_PER_GPU) name it alike and keep their jobs' memory within it
        self.slot, self.capacity_gb = os.environ.get('LOUD_GPU_SLOT'), gpu_memory_gb()
        self.completed = 0

    def report(self, state, **fields):
        self.queue.worker_report(self.worker_id, state=state, gpu=self.gpu, pid=os.getpid(),
                                 completed=self.completed, code=os.environ.get('LOUD_CODE_DIR'), **fields)

    def stop(self, number, frame):
        self.stopping = signal.Signals(number).name
        if self.child is not None and self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGTERM)

    def free_gb(self):
        return shutil.disk_usage(REPO).free / 1e9

    def loop(self):
        idle_since = time.time()
        while not self.stopping:
            if is_on(self.queue.root):
                self.report('exited: drained')
                print(f'{utc_now()} the queue is drained (python3 -m loud.drain off to resume); exiting', flush=True)
                return 0
            free = self.free_gb()
            disk_full = free < self.min_free_gb
            job = None if disk_full else self.queue.claim(self.worker_id, self.gpu, self.slot,
                                                         self.capacity_gb and 0.95 * self.capacity_gb)
            if job is None:
                why = f'shared disk nearly full ({free:.1f} GB free)' if disk_full else 'no ready job'
                idle = round(time.time() - idle_since)
                if idle > self.idle_exit:
                    self.report(f'exited: idle ({why})', idle_seconds=idle)
                    print(f'{utc_now()} idle for {idle}s ({why}); exiting', flush=True)
                    return 0
                self.report(f'waiting: {why}' if disk_full else 'idle', idle_seconds=idle)
                time.sleep(self.poll)
                continue
            self.execute(job)
            self.completed += 1
            idle_since = time.time()
        self.report(f'exited: {self.stopping or "done"}')
        return 0

    def execute(self, job):
        jid, run_dir, code = job['id'], Path(job['attempt_dir']), Path(job['code_dir'])
        print(f'{utc_now()} start {jid} (attempt {job["attempts"]}) in {run_dir.name}, code {code.name}', flush=True)
        self.report('running', job=jid)
        try:
            verify(code)
        except Exception as exc:  # a snapshot that no longer matches its manifest is never run
            self.queue.finish(jid, self.worker_id, run_dir, 'failed', reason=f'code snapshot check failed: {exc}')
            return
        write_json(run_dir / 'job.json', job)
        if self.stopping:  # the signal came during the claim
            self.release(jid, run_dir)
            return
        env = dict(os.environ, PYTHONPATH=str(code), LOUD_CODE_DIR=str(code), LOUD_JOB_ID=jid)
        with (run_dir / 'stdout.log').open('a') as log:
            log.write(f'{utc_now()} worker {self.worker_id} gpu {self.gpu} attempt {job["attempts"]}\n')
            log.flush()
            self.child = subprocess.Popen([self.python, '-u', '-m', 'loud.run', '--job', str(run_dir / 'job.json')],
                                          cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
        try:
            last_beat = 0.0
            while self.child.poll() is None:
                if self.stopping:
                    self.terminate()
                    self.release(jid, run_dir)
                    return
                if time.time() - last_beat >= HEARTBEAT_SECONDS:
                    try:
                        self.queue.heartbeat(jid, self.worker_id, run_dir, progress_of(run_dir))
                    except LostLease as exc:
                        print(f'{utc_now()} lost {jid}: {exc}; stopping it', flush=True)
                        self.terminate()
                        return
                    self.report('running', job=jid, progress=progress_of(run_dir))
                    last_beat = time.time()
                time.sleep(2)
            code_ = self.child.returncode
        finally:
            self.child = None
        result_path = run_dir / 'result.json'
        try:
            if self.stopping and code_ != 0:
                self.release(jid, run_dir)
            elif code_ == 0 and result_path.exists():
                result = read_json(result_path)
                summary = {k: result.get(k) for k in ('val_loss', 'test_loss', 'heldout_loss', 'train_seconds',
                                                      'complete', 'gpu')}
                summary['check'] = (result.get('check') or {}).get('status')
                self.queue.finish(jid, self.worker_id, run_dir, 'done', result=summary, exit_code=0)
                print(f'{utc_now()} done {jid}: {summary}', flush=True)
            else:
                self.queue.finish(jid, self.worker_id, run_dir, 'failed', exit_code=code_,
                                  log_tail=tail(run_dir / 'stdout.log'))
                print(f'{utc_now()} failed {jid}: exit {code_}', flush=True)
        except LostLease as exc:  # the lease expired while the child was finishing; the retry owns the job now
            print(f'{utc_now()} lost {jid} before recording it: {exc}', flush=True)

    def release(self, jid, run_dir):
        try:
            self.queue.release(jid, self.worker_id, run_dir, f'worker received {self.stopping}')
            print(f'{utc_now()} released {jid}: {self.stopping}', flush=True)
        except LostLease as exc:
            print(f'{utc_now()} could not release {jid}: {exc}', flush=True)

    def terminate(self, grace=60):
        if self.child is None or self.child.poll() is not None:
            return
        try:
            os.killpg(self.child.pid, signal.SIGTERM)
            self.child.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(self.child.pid, signal.SIGKILL)
            self.child.wait(timeout=30)
        except ProcessLookupError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--queue', type=Path, default=QUEUE)
    parser.add_argument('--worker-id', default=socket.gethostname())
    parser.add_argument('--idle-exit', type=float, default=float(os.environ.get('LOUD_IDLE_EXIT_SECONDS', 1200)))
    parser.add_argument('--poll', type=float, default=15)
    parser.add_argument('--min-free-gb', type=float, default=float(os.environ.get('LOUD_MIN_FREE_GB', 15)))
    parser.add_argument('--python', default=sys.executable, help='Interpreter for the jobs.')
    args = parser.parse_args()
    worker = Worker(JobQueue(args.queue), args.worker_id, args.idle_exit, args.poll, args.min_free_gb, args.python)
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    print(f'{utc_now()} worker {args.worker_id} gpu {worker.gpu} queue {args.queue} idle-exit {args.idle_exit}s',
          flush=True)
    sys.exit(worker.loop())


if __name__ == '__main__':
    main()
