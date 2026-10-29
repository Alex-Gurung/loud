"""End to end on CPU: submit noop configs, run the real worker from a code snapshot, stop it with SIGTERM."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def env_for(tmp_path):
    # LOUD_QUEUE_DIR too: inside a worker pod it names the real queue, which a test worker must never claim from.
    return dict(os.environ, LOUD_REPO=str(REPO), LOUD_ARTIFACTS=str(tmp_path / 'artifacts'),
                LOUD_QUEUE_DIR=str(tmp_path / 'artifacts' / 'queue'), PYTHONPATH=str(REPO))


def write_config(tmp_path, name, args, after=None, priority=100):
    path = tmp_path / 'configs' / f'{name}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'experiment: test', f'variant: {name}', 'task: noop', 'seeds: [0]', f'args: {json.dumps(args)}',
             f'priority: {priority}']
    if after:
        lines.append(f'after: {json.dumps(after)}')
    path.write_text('\n'.join(lines) + '\n')
    return path


def submit(tmp_path, *configs):
    subprocess.run([sys.executable, '-m', 'loud.submit', *map(str, configs)], cwd=REPO, env=env_for(tmp_path),
                   check=True, capture_output=True)


def queue_state(tmp_path, jid):
    for state in ('queued', 'running', 'done', 'failed'):
        p = tmp_path / 'artifacts' / 'queue' / state / f'{jid}.json'
        if p.exists():
            return state, json.loads(p.read_text())
    return None, None


def test_worker_runs_jobs_in_dependency_order(tmp_path):
    ok = write_config(tmp_path, 'ok', dict(seconds=1, val_loss=3.5))
    bad = write_config(tmp_path, 'bad', dict(fail=True))
    # the dependent job has the best priority, so only its dependency can make it wait
    after = write_config(tmp_path, 'after', dict(seconds=0, val_loss=3.4), after=['test.ok.s0'], priority=1)
    submit(tmp_path, ok, bad, after)
    worker = [sys.executable, '-m', 'loud.worker', '--idle-exit', '3', '--poll', '1']
    proc = subprocess.run(worker, cwd=REPO, env=env_for(tmp_path), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    ok_record, after_record = queue_state(tmp_path, 'test.ok.s0')[1], queue_state(tmp_path, 'test.after.s0')[1]
    assert queue_state(tmp_path, 'test.ok.s0')[0] == 'done' and queue_state(tmp_path, 'test.after.s0')[0] == 'done'
    assert after_record['claimed_at'] >= ok_record['finished_at']
    state, record = queue_state(tmp_path, 'test.bad.s0')
    assert state == 'failed' and any('noop asked to fail' in line for line in record['log_tail'])
    run = tmp_path / 'artifacts' / 'runs' / 'test' / 'ok' / 's0'
    result = json.loads((run / 'result.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    assert result['val_loss'] == 3.5 and config['code']['digest'] and (run / 'stdout.log').exists()
    assert Path(config['code']['snapshot']).parent == tmp_path / 'artifacts' / 'code'


def test_sigterm_gives_the_job_back(tmp_path):
    slow = write_config(tmp_path, 'slow', dict(seconds=60))
    submit(tmp_path, slow)
    worker = subprocess.Popen([sys.executable, '-m', 'loud.worker', '--idle-exit', '30', '--poll', '1'], cwd=REPO,
                              env=env_for(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 60
    while time.time() < deadline:
        run = tmp_path / 'artifacts' / 'runs' / 'test' / 'slow' / 's0.attempt1' / 'metrics.jsonl'
        if run.exists() and run.read_text().count('\n') >= 2:
            break
        time.sleep(0.5)
    worker.send_signal(signal.SIGTERM)
    worker.wait(timeout=90)
    state, record = queue_state(tmp_path, 'test.slow.s0')
    assert state == 'queued' and record['history'][-1]['event'] == 'released'
    assert (tmp_path / 'artifacts' / 'runs' / 'test' / 'slow' / 's0.attempt1' / 'metrics.jsonl').exists()


def test_gpu_name_follows_cuda_visible_devices(monkeypatch):
    """A worker pinned to one of the pod's GPUs (gpu_worker_2.yaml) reports that GPU's name."""
    from loud import worker
    listing = '0, GPU-a, NVIDIA H100 80GB HBM3\n1, GPU-b, NVIDIA H200\n'
    monkeypatch.setattr(worker.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, listing, ''))
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    assert worker.gpu_name() == 'NVIDIA H100 80GB HBM3'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    assert worker.gpu_name() == 'NVIDIA H200'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-b')
    assert worker.gpu_name() == 'NVIDIA H200'
