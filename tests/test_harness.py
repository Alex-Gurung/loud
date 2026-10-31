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


CHECKPOINT = 'args: {tool: credit_dump, checkpoint: "e1.h8_synth.s0:ckpt_20.pt"}'


@pytest.mark.parametrize('text, error', [
    (f'task: diagnose\n{CHECKPOINT}', 'e1.h8_synth.s0.s files, so it must be done first: list it in after$'),
    (f'task: diagnose\nafter_done: [e1.h8_synth.s0]\n{CHECKPOINT}', 'e1.h8_synth.s0.s files.*in after$'),
    ('task: compare\nafter: [verify.a.s0]\nargs: {a: verify.a.s0, b: [noise, log.jsonl]}', 'in after_done$'),
    ('task: gate\nafter_done: [h6.x.s0]\nargs: {values: {v: [h6.y.s0, f.json, median]}}\nexpect: {v: {min: 0}}',
     'h6.y.s0.s files.*in after or after_done$'),
    (f'task: diagnose\nafter: [e1.h8_synth.s0]\n{CHECKPOINT}', None),
    ('task: diagnose\nargs: {tool: credit_dump, checkpoint: artifacts/reference/x.pt}', None),
    ('task: compare\nafter_done: [verify.a.s0]\nargs: {a: verify.a.s0, b: [noise, log.jsonl]}', None),
    ('task: gate\nafter: [h6.y.s0]\nargs: {values: {v: [h6.y.s0, f.json, median]}}\nexpect: {v: {min: 0}}', None),
])
def test_job_ids_in_args_must_be_dependencies(tmp_path, text, error):
    path = tmp_path / 'c.yaml'
    path.write_text(f'experiment: e\nvariant: v\n{text}\n')
    if error is None:
        specs_from_config(path)
        return
    with pytest.raises(ValueError, match=error) as caught:
        specs_from_config(path)
    assert str(caught.value).startswith(f'{path}: args read ')


def test_submit_prints_a_config_error_in_one_line(tmp_path, monkeypatch):
    import sys
    from loud import submit
    path = tmp_path / 'c.yaml'
    path.write_text('experiment: e\nvariant: v\ntask: proxy\nargs: {bacth: 4}\n')
    monkeypatch.setattr(sys, 'argv', ['submit', str(path), '--dry-run', '--queue', str(tmp_path / 'queue')])
    with pytest.raises(SystemExit) as caught:
        submit.main()
    assert caught.value.code == f"{path}: unknown proxy options: ['bacth']"


def test_plots_reject_unknown_experiments_and_name_the_python_with_matplotlib(tmp_path, monkeypatch, capsys):
    import sys
    from loud import plots
    monkeypatch.setattr(plots, 'RUNS', tmp_path / 'runs')
    monkeypatch.setattr(plots, 'ARTIFACTS', tmp_path)
    monkeypatch.setattr(plots, 'REPO', paths.SOURCE)
    names = plots.known(set())
    assert names[:3] == ['verify', 'h6', 'e1'] and {'e8', 'e12', 'e16'} <= set(names) and 'followups' not in names
    monkeypatch.setattr(sys, 'argv', ['plots', 'E4'])
    with pytest.raises(SystemExit):
        plots.main()
    assert 'invalid choice' in capsys.readouterr().err and not (tmp_path / 'plots').exists()

    def no_matplotlib():
        raise ModuleNotFoundError("No module named 'matplotlib'", name='matplotlib')
    monkeypatch.setattr(plots, 'pyplot', no_matplotlib)
    fake_run(tmp_path, 'e4', 'adapt', 0, 3.70)
    monkeypatch.setattr(sys, 'argv', ['plots', 'e4'])
    with pytest.raises(SystemExit, match='plots need matplotlib; run /opt/loud-cpu-venv/bin/python -m loud.plots'):
        plots.main()


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
    synth = lambda s: dict(phase='cut', cosine_to_identity=0.8, stable_rank={'attn.c_q': 20.0, 'mlp.c_fc': 30.0}) \
        if s % 10 == 0 else {}
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
        for arm, val in (('h8_exact_control', 4.10), ('h8_control', 4.20), ('h8_snr', 4.14), ('h8_rate', 4.18)):
            fake_run(tmp_path, 'e8', arm, seed, val)
        (tmp_path / 'runs' / 'e3' / 'avg10' / f's{seed}' / 'population.jsonl').write_text(''.join(
            json.dumps(dict(round=r, step=242 + 4 * r, barriers={'0-1': 0.01 * r, '0-2': -0.01})) + '\n'
            for r in (10, 20)))
    for seed in (0, 1):
        fake_run(tmp_path, 'e1', 'bp_b4', seed, 3.70)
        fake_run(tmp_path, 'e1', 'h8_cut', seed, 3.80)
        fake_run(tmp_path, 'e14', 'lambda_1', seed, 3.75)
        fake_run(tmp_path, 'e13', 'theta2', seed, 4.20, lambda s: dict(low_tier=0.4))
        fake_run(tmp_path, 'e1', 'bp_b4_pr40', seed, 3.60)
        fake_run(tmp_path, 'e16', 'bp_track3', seed, 3.40)
        fake_run(tmp_path, 'e16', 'h8_track3', seed, 3.85)
    for experiment in ('e2', 'e3', 'e4', 'e8', 'e14'):
        plots.plot(experiment)
    assert '| lambda_1 | 2 | 3.7500 | +0.25 |' in (tmp_path / 'plots' / 'e14' / 'gap.md').read_text()
    made = {p.relative_to(tmp_path / 'plots').as_posix() for p in (tmp_path / 'plots').rglob('*.*')}
    assert {'e4/step_control.png', 'e3/barriers.png', 'e2/gap.md', 'e8/snr.md', 'e2/final.png'} <= made
    assert '| baseline e1.bp_b8 | 2 | 3.7100 |' in (tmp_path / 'plots' / 'e2' / 'gap.md').read_text()
    snr = (tmp_path / 'plots' / 'e8' / 'snr.md').read_text()
    assert '| snr | mlp.c_fc | 0.5000 | 0.6000 |' in snr and '| attn.c_q | 0.250 | 0.600 |' in snr
    assert 'partial rows cost +0.1000; SNR scaling recovers 60% of it' in snr and 'beats it: yes' in snr, snr
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
    e13 = (tmp_path / 'plots' / 'e13' / 'delta.md').read_text()
    assert '| theta2 | 2 | 4.2000 ± 0.0000 | +0.0200 | 0.4000 |' in e13, e13
    fake_run(tmp_path, 'h6', 'h8_validation', 0, None, summary=dict(cosines={'a': 0.999, 'b': 0.995}, below_099=[]))
    fake_run(tmp_path, 'h6', 'cos_to_autograd_c20', 0, None, summary={'all': 0.97, 'mlp.c_fc': 0.95})
    plots.plot('h6')
    h6 = (tmp_path / 'plots' / 'h6' / 'h6.md').read_text()
    assert 'min cosine 0.9950; below 0.99: none' in h6 and '| mlp.c_fc | 0.9500 | - | - | - | - | - |' in h6, h6
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
            {f'S{i + 1}': dict(p10=0.0, median=m, p90=1.0) for i, m in enumerate(medians)},
            S0=dict(p10=0.0, median=-1.0, p90=0.0))))  # left out by the pattern
    spec = dict(experiment='e7', variant='precondition', seed=0, task='gate', expect=dict(
        early=dict(min=0.9), late=dict(max=0.7)), args=dict(values=dict(
            early=['h6.credit_dump_c20.s0', 'credit_dump.json', 'median', 'S[1-8]'],
            late=['h6.credit_dump_c100.s0', 'credit_dump.json', 'median', 'S[1-8]'])))
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


def test_scored_finals_shrink_to_bf16_unless_kept_exact(tmp_path):
    torch = pytest.importorskip('torch')
    from loud.checkpoints import shrink_final
    from loud.options import proxy_options
    from loud.train import shrink_scored
    state = dict(w=torch.randn(64, 64), lam=torch.randn(8), emb=torch.randn(16, 8).bfloat16(), n=torch.arange(3))
    for name in ('a.pt', 'b.pt'):
        torch.save(state, tmp_path / name)
    result = {}
    shrink_scored(proxy_options({}), result, [tmp_path / 'a.pt'])
    shrunk = torch.load(tmp_path / 'a.pt', weights_only=True)
    assert result == dict(final_dtype='bfloat16') and shrunk['w'].dtype == torch.bfloat16
    assert torch.equal(shrunk['w'], state['w'].bfloat16()) and torch.equal(shrunk['lam'], state['lam'])
    assert torch.equal(shrunk['emb'], state['emb']) and torch.equal(shrunk['n'], state['n'])
    exact = {}
    shrink_scored(proxy_options(dict(final_dtype='float32')), exact, [tmp_path / 'b.pt'])
    assert exact == {} and torch.load(tmp_path / 'b.pt', weights_only=True)['w'].dtype == torch.float32
    assert shrink_final(tmp_path / 'b.pt', torch.bfloat16) > 0
    with pytest.raises(ValueError, match='final_dtype'):
        proxy_options(dict(final_dtype='float16'))


def test_board_tables_from_configs_queue_and_runs(tmp_path):
    from loud import board
    from loud.jobqueue import JobQueue
    configs = tmp_path / 'configs'
    for rel, text in (('e1/h8_synth.yaml', 'experiment: e1\nvariant: h8_synth\ntask: h8\nseeds: [0, 1]\npriority: 50\n'
                                           'args: {sequences: 1M}\n'),
                      ('e6/token_k4.yaml', 'experiment: e6\nvariant: token_k4\ntask: h8\nseeds: [0, 1]\npriority: 55\n'
                                           'after: [e1.h8_synth.s0]\nargs: {sequences: 1M, rows: 128}\n'),
                      ('followups/e8/snr.yaml', 'experiment: e8\nvariant: snr\ntask: proxy\nseeds: [0]\npriority: 62\n'
                                                'args: {sequences: 1M, per_sequence_grads: true, snr_scaling: snr}\n')):
        (configs / rel).parent.mkdir(parents=True, exist_ok=True)
        (configs / rel).write_text(text)
    queue = JobQueue(tmp_path / 'queue')
    jobs = board.plan([configs], None, None)
    for job in jobs:
        queue.submit(dict(job, code_dir=None, code_digest=None))
    for seed, val in ((0, 4.10), (1, 4.12)):
        fake_run(tmp_path, 'e1', 'h8_synth', seed, val)
    fake_run(tmp_path, 'e6', 'token_k4', 0, 4.15)
    counts = board.build(configs, tmp_path / 'queue', tmp_path / 'runs', tmp_path / 'board')
    assert counts['jobs'] == 5 and counts['runs'] == 3
    read = lambda name: json.loads((tmp_path / 'board' / f'{name}.json').read_text())
    arms = {a['key']: a for a in read('arms')}
    assert arms['e1.h8_synth']['counted'] == 2 and abs(arms['e1.h8_synth']['val_mean'] - 4.11) < 1e-9
    assert arms['e6.token_k4']['change'] == 'rows=128' and abs(arms['e6.token_k4']['delta'] - 0.04) < 1e-9
    assert arms['e8.snr']['followup'] and arms['e8.snr']['val_mean'] is None
    jobs = {j['id']: j for j in read('jobs')}
    assert jobs['e6.token_k4.s1']['state'] == 'waiting' and jobs['e8.snr.s0']['state'] == 'follow-up'
    experiments = {e['id']: e for e in read('experiments')}
    assert experiments['e6']['verdict'] == 'waiting' and experiments['e6']['reference_val'] == arms['e1.h8_synth']['val_mean']
    assert experiments['e8']['followup'] and not experiments['e6']['followup']
    assert {c['arm'] for c in read('curves')} == {'e1.h8_synth', 'e6.token_k4'}


def test_prune_keeps_what_is_read(tmp_path):
    """loud.prune: a finished run's weights go (E1's too); the verifications, files a config reads, unfinished runs and
    diagnostic tensors stay."""
    from loud import prune
    runs, configs = tmp_path / 'runs', tmp_path / 'configs'
    configs.mkdir()
    (configs / 'a.yaml').write_text('args:\n  checkpoint: e7.arm.s1:ckpt_20.pt\n')
    (configs / 'b.yaml').write_text("args: {tool: seed_line, partner: 'e7.arm.s1:ckpt_50.pt'}\n")  # quoted, partner

    def run(path, files, complete=True):
        d = runs / path
        d.mkdir(parents=True)
        for f in files:
            (d / f).write_bytes(b'x')
        if complete is not None:
            (d / 'result.json').write_text(json.dumps({'complete': complete}))
        return d
    run('e1/h8_cut/s0', ['final.pt'])
    run('e1/h8_cut/s1', ['final.pt'])
    run('verify/bp_b8/s0', ['final.pt'])
    run('e7/arm/s1', ['final.pt', 'ckpt_20.pt', 'ckpt_20.optim.pt', 'ckpt_50.pt', 'snr_spectrum.pt'])
    run('e7/arm/s2', ['final.pt'], complete=None)  # still running
    early = run('e7/smoke/s0', ['final.pt'], complete=False)  # stopped early on purpose
    (early / 'result.json').write_text(json.dumps({'complete': False, 'finished_at': 'then'}))
    original = prune.checkpoints_read
    prune.checkpoints_read = lambda: original(configs)
    try:
        gone = sorted(str(f.relative_to(runs)) for f in prune.candidates(runs))
    finally:
        prune.checkpoints_read = original
    assert gone == ['e1/h8_cut/s0/final.pt', 'e1/h8_cut/s1/final.pt', 'e7/arm/s1/final.pt', 'e7/smoke/s0/final.pt'], gone
