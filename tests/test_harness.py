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
    ('gpus: 8', 'unknown keys'),
    ('args: {checkpoints: 1}', 'checkpoints must be true, false or a list of seeds'),
    ('args: {checkpoints: [true]}', 'checkpoints must be true, false or a list of seeds'),
    ('expect: {val_loss: 3.5}', r'needs \[target, tolerance'),
    ('expect: {val_los: [3.5, 0.1]}', "no 'val_los'"),
    ('args: {blur: 0.1, fisher: true}', 'one change per variant'),
    ('args: {step_control: reject, accept_down: 0.9}', 'accept_down acts only with step_control: adapt'),
    ('args: {per_sequence_grads: true, blur: 0.1}', 'per_sequence_grads'),
    ('args: {warmdown: 1.5}', 'warmdown must be a fraction'),
    ('args: {step_control: undo}', 'step_control must be one of'),
])
def test_bad_configs_fail_at_submission(tmp_path, text, error):
    path = tmp_path / 'c.yaml'
    path.write_text(f'experiment: e\nvariant: v\ntask: proxy\n{text}\n')
    with pytest.raises(ValueError, match=error) as caught:
        specs_from_config(path)
    assert str(path) in str(caught.value)


@pytest.mark.parametrize('args, error', [
    ('{batch: 3}', 'does not divide'),
    ('{rows: 96}', 'power of two'),
    ('{rows: 1, virtual_gpus: 8, batch: 4}', 'power of two from 2'),
    ('{heldout: 0}', 'end merge needs'),
    ('{rows: 128, row_assign: random}', 'row_assign must be one of'),
    ('{prior: identity}', 'needs rows below the full cycle'),
    ('{rows: 512, prior: identity}', 'needs rows below the full cycle'),
    ('{rows: 128, row_assign: time, row_budget: ngram}', 'needs rows and token windows'),
    ('{row_budget: ngram}', 'needs rows and token windows'),
    ('{rows_start: 64}', 'unknown cut options'),
])
def test_bad_cut_options_fail_at_submission(tmp_path, args, error):
    path = tmp_path / 'c.yaml'
    path.write_text(f'experiment: e\nvariant: v\ntask: cut\nargs: {args}\n')
    with pytest.raises(ValueError, match=error):
        specs_from_config(path)


def test_turns_sum_the_parts_as_an_all_reduce():
    from loud.cut import Turns
    turns = Turns(2)
    parts = [torch.tensor([1.0, 2.0]), torch.tensor([10.0, 20.0])]
    out = turns.run([lambda p=p: turns.group_sum(p, parts[p]).clone() for p in range(2)])
    assert all(torch.equal(t, torch.tensor([11.0, 22.0])) for t in out + parts)


@pytest.mark.parametrize('extra', [0, 1])
def test_turns_fail_when_one_part_has_an_extra_collective(extra):
    """In either direction: the part with the extra group_sum would otherwise wait forever (part 0) or get its own
    tensor back unsummed (part 1)."""
    from loud.cut import Turns
    turns = Turns(2)

    def job(part):
        total = turns.group_sum(part, torch.ones(1))
        if part == extra:
            turns.group_sum(part, torch.ones(1))
        return total
    with pytest.raises(RuntimeError, match='different collectives'):
        turns.run([lambda p=p: job(p) for p in range(2)])


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
    synth = lambda s: dict(cosine_to_identity=0.8, stable_rank={'attn.c_q': 20.0, 'mlp.c_fc': 30.0}) if s % 10 == 0 \
        else {}
    for seed in (0, 1, 2):
        fake_run(tmp_path, 'e1', 'h8_synth', seed, 4.17 + 0.01 * seed)
        fake_run(tmp_path, 'e6', 'token_k4', seed, 4.19 + 0.01 * seed, synth)
    site = dict(p10=0.8, median=0.9, p90=0.95)
    readout = {s: {'cosine ridge 0.0': site, 'cosine ridge 0.001': site, 'condition': site} for s in plots.SITES[2:]}
    fake_run(tmp_path, 'e5', 'readout_c20', 0, None, summary=dict(readout, settings=dict(ridge=1e-3)))
    fake_run(tmp_path, 'h6', 'credit_dump_c20', 0, None, summary={s: dict(cosine=site, variance_ratio=0.3)
                                                                  for s in plots.SITES})
    fake_run(tmp_path, 'h6', 'snr_spectrum_c20', 0, None, summary=dict(energy_below_1={
        'backprop': {'attn.c_q': 0.25}, 'time_k4': {'attn.c_q': 0.6}}))
    fake_run(tmp_path, 'e13', 'tiers_c20', 0, None, summary=dict(theta2=dict(energy=0.05, tokens=0.45)))
    for seed in (0, 1):
        fake_run(tmp_path, 'e1', 'bp_b8', seed, 3.71, lambda s: dict(lr=dict(anvil=0.02 * (1 - s / 120))))
        fake_run(tmp_path, 'e4', 'adapt', seed, 3.70, lambda s: dict(accepted=s % 3 > 0, lr=dict(anvil=0.02)))
        fake_run(tmp_path, 'e2', 'blur_s01', seed, 3.69, lambda s: dict(blurred_loss=8.1 - s / 50))
        fake_run(tmp_path, 'e8', 'snr', seed, 3.70, lambda s: dict(snr_ratios={'mlp.c_fc': dict(
            r_mean=0.5, frobenius=0.6)}))
        fake_run(tmp_path, 'e3', 'avg10', seed, 3.72)
        (tmp_path / 'runs' / 'e3' / 'avg10' / f's{seed}' / 'population.jsonl').write_text(''.join(
            json.dumps(dict(round=r, step=242 + 4 * r, barriers={'0-1': 0.01 * r, '0-2': -0.01})) + '\n'
            for r in (10, 20)))
    for seed in (0, 1):
        fake_run(tmp_path, 'e1', 'bp_b4', seed, 3.70)
        fake_run(tmp_path, 'e1', 'h8_cut', seed, 3.80)
        fake_run(tmp_path, 'e14', 'lambda_1', seed, 3.75)
        fake_run(tmp_path, 'e13', 'theta2', seed, 4.20, lambda s: dict(low_tier=0.4))
        fake_run(tmp_path, 'e16', 'bp_fixed', seed, 3.60)
        fake_run(tmp_path, 'e16', 'bp_track3', seed, 3.40)
        fake_run(tmp_path, 'e16', 'h8_track3', seed, 3.85)
    for experiment in ('e2', 'e3', 'e4', 'e8', 'e14'):
        plots.plot(experiment)
    assert '| lambda_1 | 2 | 3.7500 | +0.50 |' in (tmp_path / 'plots' / 'e14' / 'gap.md').read_text()
    made = {p.relative_to(tmp_path / 'plots').as_posix() for p in (tmp_path / 'plots').rglob('*.*')}
    assert {'e4/step_control.png', 'e3/barriers.png', 'e2/gap.md', 'e8/snr.md', 'e2/final.png'} <= made
    assert '| baseline e1.bp_b8 | 2 | 3.7100 |' in (tmp_path / 'plots' / 'e2' / 'gap.md').read_text()
    snr = (tmp_path / 'plots' / 'e8' / 'snr.md').read_text()
    assert '| snr | mlp.c_fc | 0.5000 | 0.6000 |' in snr and '| attn.c_q | 0.250 | 0.600 |' in snr
    timed = lambda s: dict(train_seconds=s * 6.0, val_loss=5 - s / 40) if s % 25 == 0 else {}
    fake_run(tmp_path, 'e12', 'bp_pr46', 0, 3.34, timed)
    fake_run(tmp_path, 'e1', 'h8_cut', 2, 3.10)  # E16 reads seeds 0-1 of e1.h8_cut only
    for experiment in ('e6', 'e5', 'e7', 'e12', 'e13', 'e16'):
        plots.plot(experiment)
    assert (tmp_path / 'plots' / 'e12' / 'val_time.png').exists()
    assert '| bp_pr46 | 1 | 3.3400 | - | 6.0 min (n=1) |' in (tmp_path / 'plots' / 'e12' / 'summary.md').read_text()
    assert {p.name for p in (tmp_path / 'plots' / 'e6').iterdir()} == {'final.png', 'curves.png', 'delta.md'}
    e6 = (tmp_path / 'plots' / 'e6' / 'delta.md').read_text()
    assert '| token_k4 | 3 | 4.2000 ± 0.0100 | +0.0200 | 0.8000 | 25.0 |' in e6, e6
    assert '| reference e1.h8_synth | 3 | 4.1800 ± 0.0100 | 0 | - | - |' in e6, e6
    assert '| S4 | 0.900 | 0.900 | 0.900 |' in (tmp_path / 'plots' / 'e5' / 'readout.md').read_text()
    assert '| S0 | - | - | 0.300 | - | - |' in (tmp_path / 'plots' / 'e7' / 'credit_dump.md').read_text()
    e13 = (tmp_path / 'plots' / 'e13').iterdir()
    assert {'tiers.md', 'delta.md'} <= {p.name for p in e13}
    assert '| theta2 | 0.050 | 0.450 |' in (tmp_path / 'plots' / 'e13' / 'tiers.md').read_text()
    assert '| theta2 | 2 | 4.2000 ± 0.0000 | +0.0200 | 0.4000 |' in (tmp_path / 'plots' / 'e13' / 'delta.md').read_text()
    ranking = (tmp_path / 'plots' / 'e16' / 'ranking.md').read_text()
    assert '| H8 | fixed | 2 | 3.8000 |' in ranking and 'backprop +0.2000, H8 -0.0500' in ranking, ranking
    assert 'reproduces the order: no' in ranking


def test_gate_reduces_a_diagnostic_and_checks_min_max(tmp_path, monkeypatch):
    from loud import gate
    from loud.runlog import check
    monkeypatch.setattr(paths, 'RUNS', tmp_path / 'runs')
    for fraction, medians in (('c20', [0.95, 0.92, 0.5]), ('c100', [0.6, 0.65, 0.9])):
        run = tmp_path / 'runs' / 'h6' / f'credit_dump_{fraction}' / 's0'
        run.mkdir(parents=True)
        (run / 'credit_dump.json').write_text(json.dumps(dict(
            {f'mid{i}': dict(p10=0.0, median=m, p90=1.0) for i, m in enumerate(medians)},
            emb=dict(p10=0.0, median=-1.0, p90=0.0))))  # left out by the 'mid' prefix
    spec = dict(experiment='e7', variant='precondition', seed=0, task='gate', expect=dict(
        early=dict(min=0.9), late=dict(max=0.7)), args=dict(values=dict(
            early=['h6.credit_dump_c20.s0', 'credit_dump.json', 'median', 'mid'],
            late=['h6.credit_dump_c100.s0', 'credit_dump.json', 'median', 'mid'])))
    result = gate.run_task(spec, tmp_path / 'gate')
    assert result['early'] == 0.92 and result['late'] == 0.65 and result['check']['status'] == 'pass'
    assert check(dict(x=0.5), dict(x=dict(min=0.6)))['status'] == 'fail'
    assert check(dict(x=0.5), dict(x=[0.4, 0.2]))['status'] == 'pass'
    path = tmp_path / 'g.yaml'
    path.write_text('experiment: e\nvariant: v\ntask: gate\nargs: {values: {a: [j.v.s0, f.json, median]}}\n'
                    'expect: {b: {min: 1}}\n')
    with pytest.raises(ValueError, match="no 'b'"):
        specs_from_config(path)


def test_followups_are_submitted_only_when_named(tmp_path):
    from loud.submit import config_files
    for rel in ('e1/a.yaml', 'followups/e7/b.yaml', 'e2/followups/c.yaml'):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text('')
    assert [f.name for f in config_files([tmp_path])] == ['a.yaml']
    assert [f.name for f in config_files([tmp_path / 'followups'])] == ['b.yaml']
    assert [f.name for f in config_files([tmp_path / 'e2' / 'followups' / 'c.yaml'])] == ['c.yaml']


def test_turns_parts_see_the_callers_dynamo_config():
    """torch >= 2.13 keeps Dynamo's config per context: each part runs in a copy of the caller's, so the Cut port's
    raised recompile limit holds in every part (with torch 2.10 the config is global and this holds anyway)."""
    import torch._dynamo
    from loud.cut import Turns
    previous = torch._dynamo.config.recompile_limit
    torch._dynamo.config.recompile_limit = 123
    try:
        assert Turns(2).run([lambda: torch._dynamo.config.recompile_limit] * 2) == [123, 123]
    finally:
        torch._dynamo.config.recompile_limit = previous


def test_checkpoints_for_some_seeds_only(tmp_path):
    """`checkpoints: [0]` keeps H4's fraction checkpoints for seed 0 only (final.pt is always kept)."""
    from loud.options import checkpoints_on
    path = tmp_path / 'c.yaml'
    path.write_text('experiment: e\nvariant: v\ntask: proxy\nseeds: [0, 1]\nargs: {checkpoints: [0]}\n')
    keeps = [checkpoints_on(spec['args'], spec['seed']) for spec, _ in specs_from_config(path)]
    assert keeps == [True, False]
    assert checkpoints_on(dict(checkpoints=True), 5) and not checkpoints_on(dict(checkpoints=False), 0)
