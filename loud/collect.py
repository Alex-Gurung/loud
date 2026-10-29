"""H7 results table: every run's result.json in one table, mean +- standard deviation per variant.
Never report a number that is not in this table.

    python3 collect.py                  # all experiments; also writes artifacts/results/table.md and runs.json
    python3 collect.py --experiment e1  # print one experiment only (writes nothing)
    python3 collect.py --runs           # one row per run as well

Only complete runs (full epoch, scored, and not failing their own check) enter the means; the rest are listed apart,
with run directories that hold no result.json. Flags: 'mixed GPUs' / 'mixed code' when one variant's seeds differ;
'GPU differs' / 'code differs' when a variant differs from another variant of the same experiment, so wall-clocks
(or, for code, numbers) need care before they are compared.
"""
import argparse
import json
from pathlib import Path
import re
import statistics

from .common import read_json, read_jsonl, write_json, utc_now
from .jobqueue import JobQueue
from .paths import QUEUE, RESULTS, RUNS

SEED_DIR = re.compile(r'^s(\d+)$')
ATTEMPT_DIR = re.compile(r'^(s\d+)\.attempt\d+$')


# heldout.jsonl fields that mark a score of something other than the run's weights: a trajectory mean, an E3 copy,
# midpoint or mean of copies
OTHER_SCORES = ('mean_decay', 'copy', 'midpoint', 'mean')


def own_score(h):
    """Whether a heldout.jsonl entry scores the run's own weights (not an end-merge candidate either)."""
    return h.get('heldout_loss') is not None and h.get('reason') != 'merge' and not any(k in h for k in OTHER_SCORES)


def runs(root=RUNS, experiment=None, active=()):
    """(rows with a result.json, directories without a finished run: seed directories with no result.json, and
    attempts of seeds that never finished, marked running when their path is in `active`)."""
    rows, missing = [], []
    for run_dir in sorted(Path(root).glob('*/*/s*')):
        exp, variant = run_dir.parent.parent.name, run_dir.parent.name
        if experiment and exp != experiment:
            continue
        attempt = ATTEMPT_DIR.match(run_dir.name)
        if attempt and not (run_dir.parent / attempt.group(1)).exists():
            missing.append(f'{run_dir} ({"running" if str(run_dir) in active else "crashed or abandoned"})')
        if not SEED_DIR.match(run_dir.name):
            continue  # attempts, direct runs and superseded runs
        if not (run_dir / 'result.json').exists():
            missing.append(f'{run_dir} (no result.json)')
            continue
        result = read_json(run_dir / 'result.json')
        config = read_json(run_dir / 'config.json') if (run_dir / 'config.json').exists() else {}
        code = config.get('code') or {}
        git = code.get('git') or {}
        heldout = read_jsonl(run_dir / 'heldout.jsonl')
        rows.append(dict(
            experiment=exp, variant=variant, seed=int(SEED_DIR.match(run_dir.name).group(1)),
            val_loss=result.get('val_loss'), test_loss=result.get('test_loss'),
            heldout_loss=result.get('heldout_loss'),
            heldout_min=min((h['heldout_loss'] for h in heldout if own_score(h)), default=None),
            train_seconds=result.get('train_seconds'), wall_seconds=result.get('wall_seconds'),
            complete=result.get('complete', True), check=(result.get('check') or {}).get('status'),
            gpu=result.get('gpu'), torch=(config.get('environment') or {}).get('torch'),
            code=code.get('digest'), commit=(git.get('commit') or '')[:9] + ('+dirty' if git.get('dirty') else ''),
            run_dir=str(run_dir)))
    return rows, missing


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None, 0
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else None), len(values)


def fmt(mean, sd, digits=4):
    if mean is None:
        return '-'
    return f'{mean:.{digits}f}' + (f' ± {sd:.{digits}f}' if sd is not None else '')


def counts(r):
    return r['complete'] and r['val_loss'] is not None and r['check'] != 'fail'


def table(rows):
    cells = {}
    for r in filter(counts, rows):
        cells.setdefault((r['experiment'], r['variant']), []).append(r)
    out = []
    for (exp, variant), rs in sorted(cells.items()):
        rs = sorted(rs, key=lambda r: r['seed'])
        val, test, held, low, secs = (mean_sd([r[k] for r in rs]) for k in
                                      ('val_loss', 'test_loss', 'heldout_loss', 'heldout_min', 'train_seconds'))
        checks = [r['check'] for r in rs if r['check']]
        out.append(dict(experiment=exp, variant=variant, n=len(rs), seeds=[r['seed'] for r in rs],
                        val=fmt(val[0], val[1]), val_mean=val[0], val_sd=val[1], test=fmt(test[0], test[1]),
                        heldout=fmt(held[0], held[1]), heldout_min=fmt(low[0], low[1]),
                        train_seconds=fmt(secs[0], secs[1], 1), check=f'pass {len(checks)}' if checks else '-',
                        gpus={r['gpu'] for r in rs}, codes={r['code'] for r in rs},
                        gpu=', '.join(sorted(str(r['gpu']) for r in rs)),
                        code=', '.join(sorted({str(r['code'])[:8] for r in rs})),
                        commit=', '.join(sorted({r['commit'] for r in rs}))))
    for c in out:
        flags = []
        if len(c['gpus']) > 1:
            flags.append('mixed GPUs')
        if len(c['codes']) > 1:
            flags.append('mixed code')
        others = [d for d in out if d['experiment'] == c['experiment'] and d is not c]
        if any(d['gpus'] != c['gpus'] for d in others):
            flags.append('GPU differs')
        if any(d['codes'] != c['codes'] for d in others):
            flags.append('code differs')
        c['flags'] = ', '.join(flags)
    for c in out:
        del c['gpus'], c['codes']
    return out


HEAD = ['experiment', 'variant', 'n', 'seeds', 'val', 'test', 'heldout', 'heldout_min', 'train_seconds', 'check', 'gpu',
        'code', 'commit', 'flags']


def markdown(cells):
    lines = ['| ' + ' | '.join(HEAD) + ' |', '|' + '---|' * len(HEAD)]
    for c in cells:
        lines.append('| ' + ' | '.join(str(c[h]) for h in HEAD) + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment')
    parser.add_argument('--runs', action='store_true', help='Also print one row per run.')
    parser.add_argument('--out', type=Path, default=RESULTS)
    args = parser.parse_args()
    active = {j.get('attempt_dir') for j in JobQueue(QUEUE, create=False).jobs('running')}
    rows, missing = runs(experiment=args.experiment, active=active)
    text = markdown(table(rows))
    print(text)
    why = lambda r: 'failed its check' if r['check'] == 'fail' else 'incomplete' if not r['complete'] else 'unscored'
    other = ['{}.{}.s{} ({})'.format(r['experiment'], r['variant'], r['seed'], why(r)) for r in rows if not counts(r)]
    if other:
        print(f'\nnot in the table: {", ".join(other)}')
    if missing:
        print('\nno finished run:\n  ' + '\n  '.join(missing))
    if args.runs:
        print()
        for r in rows:
            print(json.dumps({k: r[k] for k in ('experiment', 'variant', 'seed', 'val_loss', 'test_loss', 'heldout_loss',
                                                 'train_seconds', 'complete', 'check', 'gpu', 'code', 'commit')}))
    if not args.experiment:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'table.md').write_text(f'Generated {utc_now()} by collect.py\n\n{text}\n')
        write_json(args.out / 'runs.json', rows)


if __name__ == '__main__':
    main()
