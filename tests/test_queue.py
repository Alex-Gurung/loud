"""Queue semantics: ordering, dependencies, leases, attempts, release, resubmission (CPU, no torch)."""
import os
from pathlib import Path
import time

import pytest

from loud.jobqueue import JobQueue, LostLease


def job(jid, run_root, priority=100, after=(), config_hash='h', code='c1'):
    return dict(id=jid, spec={}, priority=priority, after=list(after), config_hash=config_hash, code_dir=code,
                code_digest=code, git=None, run_dir=str(run_root / jid))


@pytest.fixture
def q(tmp_path):
    return JobQueue(tmp_path / 'queue', lease_seconds=2, max_attempts=2)


def done(q, jid, worker='w', **result):
    claimed = q.claim(worker)
    assert claimed['id'] == jid
    q.finish(jid, worker, claimed['attempt_dir'], 'done', result=result)


def test_priority_then_submission_order(q, tmp_path):
    q.submit(job('b', tmp_path, priority=50))
    q.submit(job('a', tmp_path, priority=50))
    q.submit(job('first', tmp_path, priority=1))
    assert [q.claim('w')['id'] for _ in range(3)] == ['first', 'b', 'a']
    assert q.claim('w') is None


def test_dependencies_wait_and_block_on_failed_check(q, tmp_path):
    q.submit(job('verify', tmp_path, priority=1))
    q.submit(job('e1', tmp_path, priority=1, after=['verify']))
    assert q.dependency_state(q.find('e1')[1]) == ('waiting', 'verify')
    done(q, 'verify', check='fail')
    assert q.dependency_state(q.find('e1')[1]) == ('blocked', 'verify failed its check') and q.claim('w') is None
    compare = dict(job('compare', tmp_path, priority=1), after_done=['verify'])  # explains a failed gate
    q.submit(compare)
    assert q.claim('w')['id'] == 'compare'
    q.submit(job('verify', tmp_path, priority=1), rerun=True)
    done(q, 'verify', check='pass')
    assert q.claim('w')['id'] == 'e1'


def test_each_attempt_has_its_own_directory_and_a_done_one_becomes_the_run(q, tmp_path):
    q.submit(job('x', tmp_path))
    first = q.claim('w')
    assert first['attempt_dir'] == str(tmp_path / 'x.attempt1')
    (tmp_path / 'x.attempt1' / 'metrics.jsonl').write_text('{}\n')
    time.sleep(2.2)
    second = q.claim('w')  # the same worker id reclaims the expired job
    assert second['attempts'] == 2 and second['attempt_dir'] == str(tmp_path / 'x.attempt2')
    with pytest.raises(LostLease):  # the stale attempt can neither renew nor finish, even under the same worker id
        q.heartbeat('x', 'w', first['attempt_dir'])
    with pytest.raises(LostLease):
        q.finish('x', 'w', first['attempt_dir'], 'done', result={})
    (tmp_path / 'x.attempt2' / 'result.json').write_text('{}')
    q.finish('x', 'w', second['attempt_dir'], 'done', result={})
    assert (tmp_path / 'x' / 'result.json').exists() and not (tmp_path / 'x.attempt2').exists()
    assert (tmp_path / 'x.attempt1' / 'metrics.jsonl').exists()  # the stale attempt's files stay where they were


def test_lease_expiry_on_the_last_attempt_fails_the_job(q, tmp_path):
    q.submit(job('x', tmp_path))
    q.claim('dead-pod')
    time.sleep(2.2)
    assert q.claim('live-pod')['attempts'] == 2
    time.sleep(2.2)
    assert q.claim('third') is None
    state, record = q.find('x')
    assert state == 'failed' and 'lease expired' in record['reason']


def test_interrupted_transition_leaves_one_record(q, tmp_path, monkeypatch):
    """A process killed in the middle of moving a record (its rename, or an unlink of a record file) leaves it in
    exactly one state; the temp-file writes of write_json still work."""
    q.submit(job('x', tmp_path))
    replace, unlink = os.replace, Path.unlink

    def killed_replace(src, dst):
        if '.writing-' not in str(src):
            raise OSError('killed mid-move')
        return replace(src, dst)

    def killed_unlink(path, *args, **kwargs):
        if str(path).endswith('.json'):
            raise OSError('killed mid-move')
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, 'replace', killed_replace)
    monkeypatch.setattr(Path, 'unlink', killed_unlink)
    with pytest.raises(OSError):
        q.claim('w')
    monkeypatch.undo()
    assert [s for s in ('queued', 'running') if q.path(s, 'x').exists()] == ['queued']
    assert q.claim('w')['id'] == 'x'


def test_gpu_requirement(q, tmp_path):
    q.submit(dict(job('x', tmp_path), gpu='H100'))
    assert q.claim('w', 'NVIDIA H200') is None
    assert q.claim('w', 'NVIDIA H100 80GB HBM3')['id'] == 'x'


def test_gpu_count_must_match_the_worker(q, tmp_path):
    q.submit(dict(job('big', tmp_path, priority=1), spec=dict(gpus=8)))
    q.submit(job('small', tmp_path, priority=2))
    assert q.claim('w1', 'H100')['id'] == 'small'  # a one-GPU worker skips the 8-GPU job
    assert q.claim('w8', 'H100', gpus=8)['id'] == 'big'
    q.submit(job('small2', tmp_path))
    assert q.claim('w8', 'H100', gpus=8) is None  # and an 8-GPU worker never takes a one-GPU job


def test_heartbeat_keeps_lease(q, tmp_path):
    q.submit(job('x', tmp_path))
    claimed = q.claim('w')
    for _ in range(3):
        time.sleep(1)
        q.heartbeat('x', 'w', claimed['attempt_dir'], dict(step=1))
    assert q.claim('other') is None and q.find('x')[0] == 'running'


def test_release_requeues_keeps_the_attempt_and_uses_up_nothing(q, tmp_path):
    q.submit(job('x', tmp_path))
    for k in range(1, 4):  # more releases than max_attempts: a stopping worker is never the job's fault
        claimed = q.claim('w')
        assert claimed['attempt_dir'] == str(tmp_path / f'x.attempt{k}')
        q.release('x', 'w', claimed['attempt_dir'], 'SIGTERM')
    state, record = q.find('x')
    assert state == 'queued' and (tmp_path / 'x.attempt1').exists() and 'attempt_dir' not in record
    assert record['history'][-1]['event'] == 'released'


def test_resubmission(q, tmp_path):
    assert q.submit(job('x', tmp_path)) == 'queued'
    assert q.submit(job('x', tmp_path)) == 'unchanged'
    assert q.submit(job('x', tmp_path, code='c2')) == 'updated'
    assert q.submit(job('x', tmp_path, code='c2', priority=5, after=['y'])) == 'updated'
    record = q.find('x')[1]
    assert record['code_dir'] == 'c2' and record['priority'] == 5 and record['after'] == ['y']
    with pytest.raises(ValueError):
        q.submit(job('x', tmp_path, config_hash='changed'))
    q.submit(job('x', tmp_path, code='c2'))  # drop the dependency again
    done(q, 'x')
    assert q.submit(job('x', tmp_path)) == 'unchanged'
    assert q.submit(job('x', tmp_path), rerun=True) == 'queued'
    assert (tmp_path / 'x.superseded1').exists() and list((q.root / 'history').glob('x.*.json'))


def test_cancel(q, tmp_path):
    q.submit(job('x', tmp_path))
    q.cancel('x')
    assert q.find('x')[0] == 'cancelled' and q.claim('w') is None
    assert q.submit(job('x', tmp_path)) == 'unchanged'
    assert q.submit(job('x', tmp_path, config_hash='changed')) == 'unchanged'  # only --rerun brings it back
    assert q.submit(job('x', tmp_path), rerun=True) == 'queued'
