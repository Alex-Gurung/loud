"""Data order, held-out split, checkpoint schedule, trajectory means, config checks, results table, compare (CPU)."""
import json

import pytest
import torch

from loud import collect, compare, paths
from loud.checkpoints import MERGE_DECAYS, FractionCheckpoints, TrajectoryMeans
from loud.data import benchmark_split
from loud.paths import DATA
from loud.run import specs_from_config


@pytest.mark.skipif(not (DATA / 'train.pt').exists(), reason='needs data/train.pt')
def test_order_is_the_records():
    split = benchmark_split('10M', heldout=32)
    # randperm(4880, seed 1), the order of backprop_preset.py, PR #26 at seed 0 and PR #40 (regression constant)
    assert split.order[:8].tolist() == [4085, 2118, 1064, 3720, 3667, 318, 1787, 4166]
    assert split.order[-3:].tolist() == [1890, 2427, 3838]
    assert len(split.train) == 4848 and len(split.heldout) == 32
    rows = torch.load(DATA / 'train.pt', map_location='cpu', weights_only=True)[:4880].long()
    assert torch.equal(split.train[0], rows[4085]) and torch.equal(split.heldout[-1], rows[3838])
    full = benchmark_split('10M', heldout=0)
    assert torch.equal(full.train[:4848], split.train)  # holding out only drops the order's tail


def test_checkpoint_steps():
    c = FractionCheckpoints(None, 606)
    assert {s: f for s, fs in c.at.items() for f in fs} == {0: 0.0, 61: 0.1, 121: 0.2, 303: 0.5, 606: 1.0}


def test_means_follow_the_record():
    model = torch.nn.Linear(2, 1, bias=False)
    means = TrajectoryMeans(update_sequences=100)
    with torch.no_grad():
        model.weight.fill_(1.0)
    means.observe(model, step=5, first_sequence=55, last_step=20)  # before 60%: ignored
    assert means.means is None
    means.observe(model, step=6, first_sequence=60, last_step=20)  # not a multiple of 5: ignored
    assert means.means is None
    means.observe(model, step=10, first_sequence=60, last_step=20)  # first: a copy
    with torch.no_grad():
        model.weight.fill_(3.0)
    means.observe(model, step=15, first_sequence=80, last_step=20)
    for decay, mean in zip(MERGE_DECAYS, means.means):
        assert torch.allclose(mean[0], torch.full((1, 2), 1.0 + 2.0 * decay))
    assert means.updates == [10, 15]


@pytest.mark.parametrize('text, error', [
    ('seedz: [1, 2]', 'unknown keys'),
    ('after: verify.bp_b8.s0', 'after must be a list'),
    ('seed: [0, 1]', 'seeds must be'),
    ('args: {bacth: 4}', 'unknown proxy options'),
    ('args: {sequences: 10m}', 'sequences'),
    ('args: {stop_after: wrong}', 'stop_after must be an integer'),
    ('priority: high', 'priority must be an integer'),
    ('gpu: [H100, H200]', 'gpu must be'),
    ('args: {checkpoints: 1}', 'checkpoints must be true or false'),
    ('expect: {val_loss: 3.5}', r'needs \[target, tolerance'),
    ('expect: {val_los: [3.5, 0.1]}', "no 'val_los'"),
])
def test_bad_configs_fail_at_submission(tmp_path, text, error):
    path = tmp_path / 'c.yaml'
    path.write_text(f'experiment: e\nvariant: v\ntask: proxy\n{text}\n')
    with pytest.raises(ValueError, match=error) as caught:
        specs_from_config(path)
    assert str(path) in str(caught.value)


def row(variant, seed, val, check='pass', gpu='H100', code='c1', complete=True):
    return dict(experiment='e', variant=variant, seed=seed, val_loss=val, test_loss=val, heldout_loss=val,
                heldout_min=val, train_seconds=1.0, complete=complete, check=check, gpu=gpu, code=code, commit='x')


def test_table_leaves_out_failed_checks_and_flags_differences():
    rows = [row('a', 0, 3.7), row('a', 1, 9.0, check='fail'), row('a', 2, 3.8, complete=False),
            row('b', 0, 3.6, gpu='H200'), row('b', 1, 3.6, code='c2')]
    cells = {c['variant']: c for c in collect.table(rows)}
    assert cells['a']['n'] == 1 and cells['a']['val_mean'] == 3.7
    assert cells['b']['flags'] == 'mixed GPUs, mixed code, GPU differs, code differs'
    assert cells['a']['flags'] == 'GPU differs, code differs'


def test_compare_matches_two_runs_by_step(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'RUNS', tmp_path / 'runs')
    for variant, losses, val in (('a', [5.0, 4.0, 3.0], 3.70), ('b', [5.0, 4.001, 3.01], 3.705)):
        run = tmp_path / 'runs' / 'e' / variant / 's0'
        run.mkdir(parents=True)
        (run / 'metrics.jsonl').write_text(''.join(json.dumps(dict(step=i + 1, train_loss=v)) + '\n'
                                                   for i, v in enumerate(losses)))
        (run / 'result.json').write_text(json.dumps(dict(val_loss=val)))
    spec = dict(experiment='verify', variant='m', seed=0, task='compare', args=dict(a='e.a.s0', b='e.b.s0', first=2),
                expect=dict(train_loss_max_diff=[0, 0.002], val_loss_diff=[0, 0.01]))
    result = compare.run_task(spec, tmp_path / 'out')
    assert result['steps_compared'] == 2 and result['max_diff_step'] == 2
    assert result['train_loss_max_diff_all'] == pytest.approx(0.01) and result['check']['status'] == 'pass'
    with pytest.raises(ValueError, match='first asks for 5'):
        compare.run_task(dict(spec, args=dict(spec['args'], first=5)), tmp_path / 'out2')
    (tmp_path / 'runs' / 'e' / 'a' / 's0' / 'metrics.jsonl').write_text('{"step": 1, "train_loss": 5.0}\n')
    with pytest.raises(ValueError, match='lacks steps'):  # a's log stops early: b's steps 2 and 3 are not in it
        compare.run_task(dict(spec, args=dict(spec['args'], first=0)), tmp_path / 'out3')


def test_submission_rejects_unknown_and_circular_dependencies(tmp_path):
    from loud.jobqueue import JobQueue
    from loud.submit import check_dependencies
    queue = JobQueue(tmp_path / 'queue')
    a = dict(id='e.a.s0', after=['e.b.s0'])
    with pytest.raises(SystemExit, match='unknown jobs'):
        check_dependencies([a], queue)
    with pytest.raises(SystemExit, match='waits on itself'):
        check_dependencies([a, dict(id='e.b.s0', after=['e.a.s0'])], queue)
    check_dependencies([a, dict(id='e.b.s0', after=[])], queue)
