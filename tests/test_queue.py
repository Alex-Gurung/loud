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


def test_priority_then_seed_then_submission_order(q, tmp_path):
    seeded = lambda jid, seed, priority: dict(job(jid, tmp_path, priority=priority), spec=dict(seed=seed))
    q.submit(seeded('b.s0', 0, 50))
    q.submit(seeded('b.s1', 1, 50))
    q.submit(seeded('a.s0', 0, 50))
    q.submit(seeded('a.s1', 1, 50))
    q.submit(seeded('first', 2, 1))
    assert [q.claim('w')['id'] for _ in range(5)] == ['first', 'b.s0', 'a.s0', 'b.s1', 'a.s1']
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
    from loud.submit import described
    q.submit(job('x', tmp_path))
    q.cancel('x')
    assert q.find('x')[0] == 'cancelled' and q.claim('w') is None
    assert q.submit(job('x', tmp_path)) == 'unchanged'
    assert q.submit(job('x', tmp_path, config_hash='changed')) == 'unchanged'  # only --rerun brings it back
    assert described(q, 'x', 'unchanged') == 'unchanged (cancelled; resubmit with --rerun)'
    assert q.submit(job('x', tmp_path), rerun=True) == 'queued'
    assert described(q, 'x', q.submit(job('x', tmp_path))) == 'unchanged (queued)'


def test_cancel_takes_ids_and_patterns_and_honours_dry_run(q, tmp_path, capsys):
    from loud.submit import cancel
    q.submit(job('d', tmp_path))
    done(q, 'd')
    for jid in ('e13.theta2.s0', 'e13.theta3.s0', 'e6.time_k4.s0'):
        q.submit(job(jid, tmp_path))
    assert not cancel(q, ['e13.theta*', 'nope.s0', 'd', 'e9.*'], dry_run=True)
    assert capsys.readouterr().out.splitlines() == [
        'would cancel e13.theta2.s0', 'would cancel e13.theta3.s0', 'nope.s0: not in the queue',
        'd: done; only queued jobs can be cancelled', 'e9.*: no queued job matches']
    assert not q.jobs('cancelled')
    assert cancel(q, ['e13.theta*', 'e13.theta2.s0'], dry_run=False)
    assert capsys.readouterr().out.splitlines() == ['cancelled e13.theta2.s0', 'cancelled e13.theta3.s0']
    assert [q.find(j)[0] for j in ('e13.theta2.s0', 'e13.theta3.s0', 'e6.time_k4.s0')] == ['cancelled'] * 2 + ['queued']


def test_blocking_is_transitive_and_a_failed_gate_skips(q, tmp_path):
    q.submit(job('verify', tmp_path, priority=1))
    q.submit(dict(job('e7.precondition', tmp_path, priority=1), spec=dict(task='gate')))
    for jid, after in (('e1', ['verify']), ('h6', ['e1']), ('e6', ['h6']), ('e7.arm', ['e7.precondition']),
                       ('e7.next', ['e7.arm']), ('both', ['h6', 'e7.precondition'])):
        q.submit(job(jid, tmp_path, priority=2, after=after))
    assert q.dependency_state(q.find('e6')[1]) == ('waiting', 'h6')
    done(q, 'verify', check='fail')
    done(q, 'e7.precondition', check='fail')
    finished = q.finished()  # one read of the queue for every call below
    state = lambda jid: q.dependency_state(finished[2][jid], finished)
    assert state('e1') == ('blocked', 'verify failed its check')
    assert state('e6') == ('blocked', 'h6 <- e1 <- verify failed its check')
    assert state('e7.arm') == ('skipped', 'gate e7.precondition says skip')
    assert state('e7.next') == ('skipped', 'e7.arm <- gate e7.precondition says skip')
    assert state('both') == ('skipped', 'gate e7.precondition says skip')  # a skip is final; a block may be fixed
    assert q.claim('w') is None
    q.submit(job('x', tmp_path, after=['y']))
    q.submit(job('y', tmp_path, after=['x']))  # loud.submit rejects a cycle; the queue still must not hang on one
    assert q.dependency_state(q.find('x')[1]) == ('blocked', 'y <- x <- y <- a cycle of dependencies')


def test_dry_run_against_a_queue_not_made_yet(tmp_path):
    from loud.submit import check_dependencies
    q = JobQueue(tmp_path / 'none', create=False)
    check_dependencies([job('x', tmp_path)], q)
    assert q.submit(job('x', tmp_path), dry_run=True) == 'queued' and not (tmp_path / 'none').exists()


def test_status_names_its_queue_and_each_state(q, tmp_path, capsys, monkeypatch):
    import sys
    from loud import status
    q.submit(job('verify', tmp_path, priority=1))
    q.submit(dict(job('e7.precondition', tmp_path, priority=1), spec=dict(task='gate')))
    for jid, after in (('e1', ['verify']), ('h6', ['e1']), ('e7.arm', ['e7.precondition']), ('later', ['e1'])):
        q.submit(job(jid, tmp_path, priority=2, after=after))
    done(q, 'verify', check='fail')
    done(q, 'e7.precondition', check='fail')
    for k in range(12):
        q.submit(job(f'c{k:02d}', tmp_path))
        q.cancel(f'c{k:02d}')
    monkeypatch.setattr(sys, 'argv', ['status', '--queue', str(q.root)])
    status.main()
    out = capsys.readouterr().out
    assert out.startswith(f'queue: {q.root}\n')
    assert 'queued (4: 0 ready, 0 waiting, 3 blocked, 1 skipped):' in out
    assert 'blocked: e1 <- verify failed its check' in out and 'skipped (gate e7.precondition says skip)' in out
    listed = ', '.join(f'c{k:02d}' for k in range(10))
    assert f'cancelled (12): {listed}, and 2 more (ls {q.root}/cancelled)' in out
    monkeypatch.setattr(sys, 'argv', ['status', '--queue', str(tmp_path / 'typo')])
    status.main()
    assert 'typo  (MISSING' in capsys.readouterr().out
