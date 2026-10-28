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
    ('gpus: 3', 'gpus must be 1, 2, 4 or 8'),
    ('gpus: 8', 'task proxy uses one GPU'),
    ('args: {checkpoints: 1}', 'checkpoints must be true or false'),
    ('expect: {val_loss: 3.5}', r'needs \[target, tolerance'),
    ('expect: {val_los: [3.5, 0.1]}', "no 'val_los'"),
    ('args: {smooth_tau: 1, token_shaping: cap}', 'one change per variant'),
    ('args: {step_control: reject, accept_down: 0.9}', 'accept_down acts only with step_control: adapt'),
    ('args: {token_shaping: agree, agree_threshold: 0.5}', r'agree_threshold must be in \(0.5, 1\]'),
    ('args: {per_sequence_grads: true, token_shaping: cap}', 'per_sequence_grads'),
    ('args: {warmdown: 1.5}', 'warmdown must be a fraction'),
    ('args: {step_control: undo}', 'step_control must be one of'),
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


def test_launch_replaces_settings_and_rows_but_not_the_script(tmp_path):
    """loud/launch.py runs a record script with some Settings fields replaced and the epoch cut to the first rows."""
    import subprocess
    import sys
    script = tmp_path / 'record.py'
    script.write_text(
        'from dataclasses import asdict, dataclass\nimport json, sys\n'
        'SEED = 0\n@dataclass\nclass Settings:\n    lr: float = 1.0\n    switch: float = 0.2\n'
        'def load_rows(path, rows):\n    return list(range(rows))\n'
        'def main():\n'
        '    print(json.dumps(dict(seed=SEED, settings=asdict(Settings()), rows=len(load_rows("x/train.pt", 10)),'
        ' val=len(load_rows("x/val.pt", 4)), argv=sys.argv[1:])))\n'
        "if __name__ == '__main__':\n    main()\n")
    launch = paths.REPO / 'loud' / 'launch.py'
    run = lambda *a: subprocess.run([sys.executable, str(launch), str(script), *a], capture_output=True, text=True)
    out = json.loads(run('{"lr": 0.5}', '6', '2', '--output', 'o').stdout)
    assert out == dict(seed=2, settings=dict(lr=0.5, switch=0.2), rows=6, val=4, argv=['--output', 'o'])
    assert json.loads(run('{}', '0', '-').stdout) == dict(seed=0, settings=dict(lr=1.0, switch=0.2), rows=10, val=4,
                                                         argv=[])
    bad = run('{"lrr": 0.5}', '0', '-')
    assert bad.returncode and 'Settings has no' in bad.stderr


def fake_run(root, experiment, variant, seed, val, extra=None, summary=None):
    run = root / 'runs' / experiment / variant / f's{seed}'
    run.mkdir(parents=True)
    if summary is not None:
        (run / 'tool.json').write_text(json.dumps(summary))
        (run / 'result.json').write_text(json.dumps(dict(complete=True, summary_file='tool.json')))
        return
    lines = [dict(step=s, seqs=8 * s, train_loss=8 - s / 50 + 0.01 * seed, **(extra(s) if extra else {}))
             for s in range(1, 101)]
    (run / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in lines))
    (run / 'heldout.jsonl').write_text(''.join(json.dumps(dict(step=s, heldout_loss=7 - s / 60)) + '\n'
                                               for s in (0, 50, 100)))
    (run / 'result.json').write_text(json.dumps(dict(val_loss=val, complete=True, train_seconds=1.0)))


def test_plots_read_only_the_run_files(tmp_path, monkeypatch):
    pytest.importorskip('matplotlib')
    from loud import plots
    monkeypatch.setattr(plots, 'RUNS', tmp_path / 'runs')
    monkeypatch.setattr(plots, 'ARTIFACTS', tmp_path)
    drift = lambda s: dict(drift=dict(blend=0.5, fast=0.4, slow=0.3)) if s % 10 == 0 else {}
    for seed in (0, 1, 2):
        fake_run(tmp_path, 'e6', 'identity_k1', seed, 4.17 + 0.01 * seed, drift)
        fake_run(tmp_path, 'e6', 'identity_k4', seed, 4.19 + 0.01 * seed, drift)
    site = dict(p10=0.8, median=0.9, p90=0.95)
    recovery = {s: {'guess_vs_direct': site, 'bf16 condition': site, 'bf16 ridge 0.0': site, 'bf16 ridge 0.1': site}
                for s in plots.SITES}
    fake_run(tmp_path, 'e5', 'recovery_c20', 0, None, summary=dict(recovery, settings=dict(ridges=[0.0, 0.1])))
    fake_run(tmp_path, 'h6', 'credit_dump_c20', 0, None, summary={s: site for s in plots.SITES})
    stats = {'transformer.h.0.attn.c_q.weight': [dict(h2_over_c=2.0), dict(h2_over_c=0.5)]}
    fake_run(tmp_path, 'h6', 'direction_stats_c20', 0, None, summary=stats)
    for seed in (0, 1):
        fake_run(tmp_path, 'e1', 'bp_b8', seed, 3.71, lambda s: dict(lr=dict(anvil=0.02 * (1 - s / 120))))
        fake_run(tmp_path, 'e4', 'adapt', seed, 3.70, lambda s: dict(accepted=s % 3 > 0, lr=dict(anvil=0.02)))
        fake_run(tmp_path, 'e2', 'smooth_t3', seed, 3.69, lambda s: dict(blurred_loss=8.1 - s / 50))
        fake_run(tmp_path, 'e8', 'lambda_3e1', seed, 3.70, lambda s: dict(damping={'mlp.c_fc': dict(
            h=1.0, c=2.0, h2_over_c=0.5, factor=0.9, overlap=0.8)}) if s % 50 == 1 else {})
        fake_run(tmp_path, 'e3', 'avg10', seed, 3.72)
        (tmp_path / 'runs' / 'e3' / 'avg10' / f's{seed}' / 'population.jsonl').write_text(''.join(
            json.dumps(dict(round=r, step=242 + 4 * r, barriers={'0-1': 0.01 * r, '0-2': -0.01})) + '\n'
            for r in (10, 20)))
    for experiment in ('e2', 'e3', 'e4', 'e8'):
        plots.plot(experiment)
    made = {p.relative_to(tmp_path / 'plots').as_posix() for p in (tmp_path / 'plots').rglob('*.*')}
    assert {'e4/step_control.png', 'e3/barriers.png', 'e2/gap.md', 'e8/damping.md', 'e2/final.png'} <= made
    assert '| baseline e1.bp_b8 | 2 | 3.7100 |' in (tmp_path / 'plots' / 'e2' / 'gap.md').read_text()
    assert '| lambda_3e1 | mlp.c_fc | 0.5 | 0.9 | 0.8 |' in (tmp_path / 'plots' / 'e8' / 'damping.md').read_text()
    timed = lambda s: dict(train_seconds=s * 6.0, val_loss=5 - s / 40) if s % 25 == 0 else {}
    fake_run(tmp_path, 'e12', 'record', 0, 3.34, timed)
    for experiment in ('e6', 'e5', 'e7', 'e8', 'e12'):
        plots.plot(experiment)
    assert (tmp_path / 'plots' / 'e12' / 'val_time.png').exists()
    assert {p.name for p in (tmp_path / 'plots' / 'e6').iterdir()} == {'final.png', 'curves.png', 'drift.png'}
    assert '| mid7 | 0.9 | 0.9 | 0.9 | 0.9 |' in (tmp_path / 'plots' / 'e5' / 'recovery.md').read_text()
    assert '| emb | - | - | 0.9 | - | - |' in (tmp_path / 'plots' / 'e7' / 'credit_dump.md').read_text()
    assert (tmp_path / 'plots' / 'e8' / 'h2_over_c.png').exists()


def test_one_gpu_jobs_keep_their_config_hash(tmp_path):
    """gpus enters a job's identity only above 1, so queued one-GPU jobs stay unchanged configs."""
    from loud.common import digest
    from loud.run import config_hash
    path = tmp_path / 'c.yaml'
    path.write_text('experiment: e\nvariant: v\ntask: proxy\n')
    spec, _ = specs_from_config(path)[0]
    assert config_hash(spec) == digest({k: spec[k] for k in ('experiment', 'variant', 'seed', 'task', 'args', 'expect')})
    assert config_hash(dict(spec, task='reference', gpus=8)) != config_hash(dict(spec, task='reference'))
