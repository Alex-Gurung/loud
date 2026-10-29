"""Plots per experiment, read from the run files only (metrics.jsonl, heldout.jsonl, result.json, a diagnostic's JSON).

    /opt/loud-cpu-venv/bin/python -m loud.plots e1   # artifacts/plots/e1/*.png (and *.md tables); needs matplotlib
    /opt/loud-cpu-venv/bin/python -m loud.plots all

Every training experiment gets final.png (each variant's final validation loss: a dot per seed, the mean +- sd) and
curves.png (train loss and held-out loss against sequences seen, the mean over seeds), from the runs the results
table counts (complete, scored, not failing their check). Experiments add their own series where the plan names one.
Diagnostics (e5, h6) are tables. Colors: the categorical slots in a fixed order by variant name, so a variant keeps its
color across an experiment's charts.
"""
import argparse
from functools import partial
from pathlib import Path
import statistics

from .collect import counts, own_score, runs
from .common import read_json, read_jsonl
from .paths import ARTIFACTS, REPO, RUNS

FULL_CYCLE = ('e1', 'h8_synth')  # E6's and E7's reference: H8 synthesis with all 512 rows (K = 1)
SLOTS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
INK, INK_2, MUTED, SURFACE, GRID = '#0b0b0b', '#52514e', '#8a8984', '#fcfcfb', '#e6e5e1'
SITES = [f'S{k}' for k in range(9)]  # H8's: S0 the embedding, S(i+1) block i's mid stream


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': MUTED, 'axes.labelcolor': INK_2, 'text.color': INK, 'xtick.color': INK_2,
        'ytick.color': INK_2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.5, 'font.size': 9,
        'legend.frameon': False, 'figure.dpi': 130})
    return plt


def colors(variants):
    """Variant -> slot, in sorted order; past eight, the rest share the muted ink (fold them before plotting)."""
    return {v: SLOTS[i] if i < len(SLOTS) else MUTED for i, v in enumerate(sorted(variants))}


def rolling(values, window):
    out, total = [], 0.0
    for i, v in enumerate(values):
        total += v - (values[i - window] if i >= window else 0.0)
        out.append(total / min(i + 1, window))
    return out


def seed_mean(series):
    """[{x: y}] per seed -> (xs, mean ys) over the x every seed has."""
    xs = sorted(set.intersection(*(set(s) for s in series))) if series else []
    return xs, [statistics.fmean(s[x] for s in series) for x in xs]


def counted(experiment):
    rows, _ = runs(RUNS, experiment)
    by_variant = {}
    for r in filter(counts, rows):
        by_variant.setdefault(r['variant'], []).append(r)
    return by_variant


def metrics(run):
    return read_jsonl(Path(run['run_dir']) / 'metrics.jsonl')


def final_plot(plt, by_variant, palette, out, title, baseline=None, name='final.png'):
    """baseline: (label, its runs), the plan's reference cell from another experiment, drawn in muted ink with a line
    at its mean - 0.02 (the plan's success margin)."""
    variants = sorted(by_variant, key=lambda v: statistics.fmean(r['val_loss'] for r in by_variant[v]))
    rows = [(v, by_variant[v], palette[v]) for v in variants]
    if baseline:
        rows.append((baseline[0], baseline[1], MUTED))
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(rows) + 1.2))
    for y, (v, runs_, color) in enumerate(rows):
        vals = [r['val_loss'] for r in runs_]
        mean = statistics.fmean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        ax.errorbar(mean, y, xerr=sd, fmt='o', color=color, markersize=7, capsize=0, elinewidth=1.5, zorder=3)
        ax.scatter(vals, [y] * len(vals), s=18, facecolor=SURFACE, edgecolor=color, linewidth=1.2, zorder=4)
        ax.annotate(f'{mean:.4f}' + (f' ± {sd:.4f}' if len(vals) > 1 else '') + f'  (n={len(vals)})',
                    (mean, y), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=7.5, color=INK_2)
    if baseline:
        line = statistics.fmean(r['val_loss'] for r in baseline[1]) - 0.02
        ax.axvline(line, color=MUTED, linewidth=1, linestyle='--')
        ax.annotate('baseline - 0.02', (line, len(rows) - 0.5), xytext=(3, 0), textcoords='offset points',
                    fontsize=7.5, color=INK_2, va='top')
    variants = [v for v, _, _ in rows]
    ax.set_yticks(range(len(variants)), variants)
    ax.set_ylim(-0.7, len(variants) - 0.3)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('final validation loss (official scorer), lower is better')
    ax.set_title(title, loc='left', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / name)
    plt.close(fig)


def curves_plot(plt, by_variant, palette, out, title, window=25):
    fig, (train_ax, held_ax) = plt.subplots(1, 2, figsize=(11, 4))
    for v in sorted(by_variant):
        train, held = [], []
        for r in by_variant[v]:
            records = metrics(r)
            seqs = {m['step']: m['seqs'] for m in records}
            smooth = rolling([m['train_loss'] for m in records], window)
            train.append({m['seqs']: y for m, y in zip(records, smooth)})
            held.append({seqs.get(h['step'], 0): h['heldout_loss']
                         for h in read_jsonl(Path(r['run_dir']) / 'heldout.jsonl')
                         if own_score(h)})
        for ax, series in ((train_ax, train), (held_ax, held)):
            xs, ys = seed_mean([s for s in series if s])
            if xs:
                ax.plot(xs, ys, color=palette[v], label=f'{v} (n={len(series)})')
    train_ax.set_title(f'train loss, rolling mean over {window} updates', loc='left', fontsize=9)
    held_ax.set_title('held-out loss (32 training sequences that feed no update)', loc='left', fontsize=9)
    for ax in (train_ax, held_ax):
        ax.set_xlabel('sequences seen')
        ax.set_ylabel('loss')
    train_ax.legend(fontsize=7.5)
    fig.suptitle(title, x=0.01, ha='left', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'curves.png')
    plt.close(fig)


def val_time_plot(plt, by_variant, palette, out, target=3.5):
    """E12: the untimed validation curve against training time (mean over seeds at the logged updates), with the 3.5
    speedrun target; the ring at each curve's end is the final score, after the end merge where there is one."""
    fig, ax = plt.subplots(figsize=(6.4, 4))
    drawn = False
    for v in sorted(by_variant):
        series = [{m['step']: (m['train_seconds'], m['val_loss']) for m in metrics(r) if 'val_loss' in m}
                  for r in by_variant[v]]
        series = [s for s in series if s]
        if not series:
            continue
        steps = sorted(set.intersection(*(set(s) for s in series)))
        xs = [statistics.fmean(s[k][0] for s in series) / 60 for k in steps]
        ys = [statistics.fmean(s[k][1] for s in series) for k in steps]
        ax.plot(xs, ys, color=palette[v], marker='o', markersize=3, label=f'{v} (n={len(series)})')
        finals = by_variant[v]
        ax.scatter([statistics.fmean(r['train_seconds'] for r in finals) / 60],
                   [statistics.fmean(r['val_loss'] for r in finals)], s=50, facecolor=SURFACE,
                   edgecolor=palette[v], linewidth=1.5, zorder=4)
        drawn = True
    if not drawn:
        plt.close(fig)
        return
    ax.axhline(target, color=MUTED, linewidth=1, linestyle='--')
    ax.annotate(f'speedrun target {target}', (ax.get_xlim()[1], target), xytext=(-4, 4), textcoords='offset points',
                ha='right', fontsize=7.5, color=INK_2)
    ax.set_xlabel('training time (minutes, timed region)')
    ax.set_ylabel('validation loss')
    ax.set_title('validation loss over training time (ring: final score)', loc='left', fontsize=9)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / 'val_time.png')
    plt.close(fig)


def windows(series, width):
    """{step: value} -> (window starts, mean value per window of `width` steps)."""
    groups = {}
    for step, value in series.items():
        groups.setdefault((step - 1) // width, []).append(value)
    keys = sorted(groups)
    return [k * width + 1 for k in keys], [statistics.fmean(groups[k]) for k in keys]


def e4_plots(plt, by_variant, palette, out, width=50):
    """E4: the acceptance rate per 50 updates, and the effective ANVIL II rate against the hand schedule (e1.bp_b8's
    logged rate), means over seeds."""
    hand = counted('e1').get('bp_b8', [])
    fig, (accept_ax, rate_ax) = plt.subplots(1, 2, figsize=(11, 4))
    for v in sorted(by_variant):
        accepted = [{m['step']: float(m['accepted']) for m in metrics(r) if 'accepted' in m} for r in by_variant[v]]
        xs, ys = seed_mean([dict(zip(*windows(a, width))) for a in accepted if a])
        if xs:
            accept_ax.plot(xs, ys, color=palette[v], marker='o', markersize=3, label=v)
        xs, ys = seed_mean([{m['step']: m['lr']['anvil'] for m in metrics(r)} for r in by_variant[v]])
        rate_ax.plot(xs, ys, color=palette[v], label=v)
    if hand:
        xs, ys = seed_mean([{m['step']: m['lr']['anvil'] for m in metrics(r)} for r in hand])
        rate_ax.plot(xs, ys, color=MUTED, linestyle='--', label='hand schedule (e1.bp_b8)')
    accept_ax.axhline(0.6, color=MUTED, linewidth=1, linestyle=':')
    accept_ax.set_title(f'acceptance rate per {width} updates (dotted: the plan\'s 0.6)', loc='left', fontsize=9)
    rate_ax.set_title('effective ANVIL II rate', loc='left', fontsize=9)
    for ax in (accept_ax, rate_ax):
        ax.set_xlabel('update')
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / 'step_control.png')
    plt.close(fig)


def e3_plot(plt, by_variant, palette, out):
    """E3: the largest pairwise barrier (a midpoint's held-out loss minus the worse of its two copies; above 0, the
    pair has left one basin) at each logged round, mean over seeds."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for v in sorted(by_variant):
        series = [{e['step']: max(e['barriers'].values()) for e in read_jsonl(Path(r['run_dir']) / 'population.jsonl')}
                  for r in by_variant[v]]
        xs, ys = seed_mean([s for s in series if s])
        if xs:
            ax.plot(xs, ys, color=palette[v], marker='o', markersize=3, label=v)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xlabel('update (the single run\'s count)')
    ax.set_ylabel('largest barrier (nats)')
    ax.set_title('E3: midpoints between copies (above 0: the copies left one basin)', loc='left', fontsize=9)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / 'barriers.png')
    plt.close(fig)


def e2_table(out):
    """E2: final validation loss, the clean training loss over the last 50 updates, the gap between them, and the blur's
    cost (the displaced passes' loss minus the clean one), means over seeds; e1.bp_b8 as the baseline."""
    cells = dict(counted('e2'))
    cells['baseline e1.bp_b8'] = counted('e1').get('bp_b8', [])
    rows = []
    for v, runs_ in sorted(cells.items()):
        if not runs_:
            continue
        tails = [metrics(r)[-50:] for r in runs_]
        val = statistics.fmean(r['val_loss'] for r in runs_)
        train = statistics.fmean(statistics.fmean(m['train_loss'] for m in t) for t in tails)
        blur = [m['blurred_loss'] - m['train_loss'] for t in tails for m in t if 'blurred_loss' in m]
        rows.append([v, len(runs_), f'{val:.4f}', f'{train:.4f}', f'{val - train:+.4f}',
                     f'{statistics.fmean(blur):+.4f}' if blur else '-'])
    if not rows:
        return None
    text = markdown(['variant', 'n', 'val', 'train (last 50)', 'val - train', 'blurred - clean'], rows)
    (out / 'gap.md').write_text(text)
    return text


def e8_table(out):
    """E8 (v2): H6's snr_spectrum, the share of the momentum's energy in directions below SNR 1 per matrix type and
    estimator at 20% and 100% (the build condition: partial rows push a large share below 1), and per SNR arm and
    matrix type the mean r and Frobenius ratio over its updates and seeds."""
    texts = []
    found = {c: diagnostic('h6', f'snr_spectrum_c{c}') for c in (20, 100)}
    found = {c: x['energy_below_1'] for c, x in found.items() if x}
    estimators = sorted({e for x in found.values() for e in x})
    kinds = sorted({k for x in found.values() for e in x.values() for k in e})
    if kinds:
        header = ['matrix'] + [f'{c}%: {e}' for c in found for e in estimators]
        rows = [[k] + [f'{found[c][e][k]:.3f}' if k in found[c].get(e, {}) else '-' for c in found for e in estimators]
                for k in kinds]
        texts.append('Share of the momentum\'s energy below SNR 1 (at batch 8)\n\n' + markdown(header, rows))
    rows = []
    for v, runs_ in sorted(counted('e8').items()):
        ratios = [m['snr_ratios'] for r in runs_ for m in metrics(r) if 'snr_ratios' in m]
        for kind in sorted(ratios[0]) if ratios else []:
            rows.append([v, kind] + [f'{statistics.fmean(x[kind][key] for x in ratios):.4f}'
                                     for key in ('r_mean', 'frobenius')])
    if rows:
        texts.append(markdown(['variant', 'matrix', 'mean r', 'Frobenius ratio'], rows))
    cells = counted('e8')
    means = {v: val_stats(cells[v])[0] for v in ('h8_exact_control', 'h8_control', 'h8_snr', 'h8_rate') if v in cells}
    if {'h8_exact_control', 'h8_control', 'h8_snr'} <= set(means) and means['h8_control'] != means['h8_exact_control']:
        cost = means['h8_control'] - means['h8_exact_control']
        recovered = (means['h8_control'] - means['h8_snr']) / cost
        beats = f', rate control {means["h8_rate"]:.4f} (SNR beats it: ' + \
            f'{"yes" if means["h8_snr"] < means["h8_rate"] else "no"})' if 'h8_rate' in means else ''
        texts.append(f'H8 setting: partial rows cost {cost:+.4f}; SNR scaling recovers {recovered:.0%} of it '
                     f'(success: at least half, and beating the rate control){beats}.\n')
    if not texts:
        return None
    text = '\n'.join(texts)
    (out / 'snr.md').write_text(text)
    return text


def gap_table(experiment, out, low=('e1', 'bp_b4_pr40'), high=('e1', 'h8_cut'), also=(), extra=()):
    """E14 and E15: each variant's final validation loss and the fraction of the gap between H8's cut-mode
    Boost -> Cut (e1.h8_cut) and backprop on the same recipe and batch (e1.bp_b4_pr40) it closes, means over seeds,
    with the reference cells (and `also`, cells from elsewhere the plan compares against) as rows; `extra` adds the
    experiment's own columns (header, function of a cell's runs)."""
    means = {}
    for name, (source, variant) in (('low', low), ('high', high)):
        found = counted(source).get(variant)
        means[name] = statistics.fmean(r['val_loss'] for r in found) if found else None
    cells = [(v, runs_) for v, runs_ in sorted(counted(experiment).items())]
    if not cells:
        return None
    cells += [(f'reference {source}.{variant}', counted(source).get(variant, []))
              for source, variant in (high, *also, low)]
    rows = []
    for v, runs_ in cells:
        if not runs_:
            rows.append([v, 0, '-', '-'] + ['-' for _ in extra])
            continue
        val = statistics.fmean(r['val_loss'] for r in runs_)
        closed = ('-' if None in means.values() or means['high'] == means['low']
                  else f'{(means["high"] - val) / (means["high"] - means["low"]):+.2f}')
        rows.append([v, len(runs_), f'{val:.4f}', closed] + [f(runs_) for _, f in extra])
    text = markdown(['variant', 'n', 'val', f'gap closed ({high[0]}.{high[1]} -> {low[0]}.{low[1]})']
                    + [h for h, _ in extra], rows)
    (out / 'gap.md').write_text(text)
    return text


def e12_table(out, target=3.5):
    """E12: per variant, the final validation loss (the records' after their end merge), the plain weights' where a
    record kept them, and when the untimed validation curve first reached the speedrun target (linear between the
    logged points), means over seeds."""
    rows = []
    for v, runs_ in sorted(counted('e12').items()):
        results = [read_json(Path(r['run_dir']) / 'result.json') for r in runs_]
        plain = [x['val_loss_plain'] for x in results if 'val_loss_plain' in x]
        crossings = []
        for r in runs_:
            curve = [(m['train_seconds'], m['val_loss']) for m in metrics(r) if 'val_loss' in m]
            crossings += [t0 + (t1 - t0) * (y0 - target) / (y0 - y1) for (t0, y0), (t1, y1) in zip(curve, curve[1:])
                          if y0 > target >= y1][:1]
        rows.append([v, len(runs_), f'{statistics.fmean(r["val_loss"] for r in runs_):.4f}',
                     f'{statistics.fmean(plain):.4f}' if plain else '-',
                     f'{statistics.fmean(crossings) / 60:.1f} min (n={len(crossings)})' if crossings else '-'])
    if not rows:
        return None
    text = markdown(['variant', 'n', 'val', 'val, plain weights', f'reaches {target}'], rows)
    text += ('\nRuns with the record\'s end merge report val after it: compare val with val and the plain weights '
             'with the plain weights.\n')
    (out / 'summary.md').write_text(text)
    return text


def val_stats(runs_):
    vals = [r['val_loss'] for r in runs_]
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def delta_table(experiment, out, baseline, extra=None, also=()):
    """Each variant's final validation loss (mean +- sd over seeds) and its difference from the plan's reference cell
    (negative: better). extra(runs) adds the experiment's own columns (header, function); `also` adds cells from
    other experiments the plan compares against (E7: E6's matching arms without the prior)."""
    source, variant = baseline
    reference = counted(source).get(variant, [])
    base = val_stats(reference)[0] if reference else None
    header = ['variant', 'n', 'val', f'minus {source}.{variant}'] + ([h for h, _ in extra] if extra else [])
    rows = []
    cells = sorted(counted(experiment).items())
    if not cells:
        return None
    cells += [(f'{e}.{v}', counted(e).get(v, [])) for e, v in also]
    for v, runs_ in cells:
        if not runs_:
            rows.append([v, 0, '-', '-'] + ['-' for _ in extra or ()])
            continue
        mean, sd = val_stats(runs_)
        rows.append([v, len(runs_), f'{mean:.4f} ± {sd:.4f}', '-' if base is None else f'{mean - base:+.4f}']
                    + [f(runs_) for _, f in extra or ()])
    if not rows:
        return None
    if reference:
        mean, sd = val_stats(reference)
        rows.append([f'reference {source}.{variant}', len(reference), f'{mean:.4f} ± {sd:.4f}', '0']
                    + [f(reference) for _, f in extra or ()])
    text = markdown(header, rows)
    (out / 'delta.md').write_text(text)
    return text


def logged_mean(key, sub=None, phase=None):
    """A column: the mean over a cell's runs and logged updates (of `phase` only, if given) of metrics field `key`
    (its `sub` entry)."""
    def column(runs_):
        values = [m[key] if sub is None else m[key][sub] for r in runs_ for m in metrics(r)
                  if key in m and (sub is None or sub in m[key]) and (phase is None or m.get('phase') == phase)]
        return f'{statistics.fmean(values):.4f}' if values else '-'
    return column


def stable_rank_mean(runs_):
    """E6's stable rank of the orthogonalization's input, the median over matrix types of the mean over the Cut
    phase's logged updates (Boost's are the same in every arm)."""
    ranks = [m['stable_rank'] for r in runs_ for m in metrics(r) if 'stable_rank' in m and m.get('phase') == 'cut']
    if not ranks:
        return '-'
    kinds = sorted(ranks[0])
    return f'{statistics.median(statistics.fmean(x[k] for x in ranks if k in x) for k in kinds):.1f}'


E6_COLUMNS = [('cosine to K = 1', logged_mean('cosine_to_identity', phase='cut')), ('stable rank', stable_rank_mean)]
E7_ALSO = (('e6', 'token_k8'), ('e6', 'token_k4'))  # the same token rows (m = 64, 128) without the prior


def e7_tables(out):
    """E7: its precondition (H6's credit_dump) and its criterion (loss against the full cycle)."""
    return '\n'.join(t for t in (e7_table(out), delta_table('e7', out, FULL_CYCLE, E6_COLUMNS, E7_ALSO)) if t) or None


def e13_tables(out):
    """E13: the diagnostic (H6's tiers on e1.h8_synth's 20% and 100% checkpoints) and the arms against the full
    cycle."""
    return '\n'.join(t for t in (tiers_table(out), delta_table('e13', out, FULL_CYCLE,
                                                                [('low tier share', logged_mean('low_tier'))])) if t) \
        or None


def tiers_table(out):
    """E13's build condition: per theta, the low tier's share of the credit energy (under 10%) and of the tokens (over
    40%) at 20% and 100%."""
    found = {c: diagnostic('e13', f'tiers_c{c}') for c in (20, 100)}
    found = {c: x for c, x in found.items() if x}
    thetas = sorted({k for x in found.values() for k in x if k.startswith('theta')})
    if not thetas:
        return None
    header = ['theta'] + [f'{c}%: {what}' for c in found for what in ('energy share', 'token share')]
    rows = [[t] + [f'{found[c][t][w]:.3f}' if t in found[c] else '-' for c in found for w in ('energy', 'tokens')]
            for t in thetas]
    text = markdown(header, rows) + '\nBuild only if the low tier holds under 10% of the energy with over 40% of ' \
                                     'the tokens.\n'
    (out / 'tiers.md').write_text(text)
    return text


def e16_table(out):
    """E16: the four arms (fixed and track-3 models, each under backprop and H8 cut-mode Boost -> Cut, PR #40's recipe
    at batch 4, seeds 0-1), whether H8 orders the two models as backprop does, and the gaps."""
    cells = {('backprop', 'fixed'): [r for r in counted('e1').get('bp_b4_pr40', []) if r['seed'] in (0, 1)],
             ('backprop', 'track3'): counted('e16').get('bp_track3', []),
             ('H8', 'fixed'): [r for r in counted('e1').get('h8_cut', []) if r['seed'] in (0, 1)],
             ('H8', 'track3'): counted('e16').get('h8_track3', [])}
    means = {k: val_stats(v)[0] for k, v in cells.items() if v}
    if not means:
        return None
    rows = [[trainer, model, len(cells[(trainer, model)]), f'{means[(trainer, model)]:.4f}' if (trainer, model) in means
             else '-'] for trainer, model in cells]
    text = markdown(['trainer', 'model', 'n', 'val'], rows)
    if len(means) == 4:
        gap = {t: means[(t, 'fixed')] - means[(t, 'track3')] for t in ('backprop', 'H8')}
        text += (f'\nTrack-3 gain (fixed minus track-3): backprop {gap["backprop"]:+.4f}, H8 {gap["H8"]:+.4f} '
                 f'(leaderboard on 30M: {3.338 - 3.211:+.3f}). H8 reproduces the order: '
                 f'{"yes" if (gap["backprop"] > 0) == (gap["H8"] > 0) else "no"}.\n')
    (out / 'ranking.md').write_text(text)
    return text


def diagnostic(experiment, variant):
    """The summary JSON of a finished diagnose run (seed 0), or None."""
    run = RUNS / experiment / variant / 's0'
    if not (run / 'result.json').exists():
        return None
    return read_json(run / read_json(run / 'result.json')['summary_file'])


def markdown(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join('---' for _ in header) + ' |']
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(lines) + '\n'


def e5_table(out, ridge=1e-3):
    """E5's diagnostic table (H6 deep_readout_check on e1.h8_synth's 20% and 100% checkpoints): per site S2..S8, the
    median cosine of the credit solved from one 512-probe cycle launched at S1 to the exact credit (plain, and with
    the ridge toward the identity-path guess the implementation would use), and the median condition number. bf16
    forward differences, as the chain runs them."""
    found = {f'{c}%{" fp32" if fp32 else ""}': diagnostic('e5', f'readout{"_fp32" if fp32 else ""}_c{c}')
             for fp32 in (False, True) for c in (20, 100)}
    found = {k: v for k, v in found.items() if v}
    if not found:
        return None
    header = ['site'] + [f'{c}: {what}' for c in found for what in ('cos (ridge 0)', f'cos (ridge {ridge:g})',
                                                                    'condition')]
    pick = lambda s, key: f'{s[key]["median"]:.3f}' if s and key in s else '-'
    rows = []
    for site in SITES[2:]:
        row = [site]
        for summary in found.values():
            s = (summary or {}).get(site)
            row += [pick(s, 'cosine ridge 0.0'), pick(s, f'cosine ridge {ridge}'), pick(s, 'condition')]
        rows.append(row)
    text = markdown(header, rows)
    text += ('\nDecision (plan E5): all sites at 0.95 or above: ask the maintainers, then implement; failures only at '
             'the top two or three sites: a hybrid with direct cycles there; most sites failing at 100%: drop.\n')
    (out / 'readout.md').write_text(text)
    return text


def h6_tables(out):
    """H6's diagnostics as tables: H8's validation against PR #40 (the gate of every H8 experiment), the identity path's
    cosine to autograd per parameter group, how linear each site's loss response is at the chain's probe size, and H8
    synthesis against the real chain's partial-row estimate per site. credit_dump is E7's table, snr_spectrum E8's."""
    texts = []
    valid = diagnostic('h6', 'h8_validation')
    if valid:
        cos = valid['cosines']
        texts.append(f'H8 validation (e1.bp_b8.s0, 50%): {len(cos)} matrices, min cosine {min(cos.values()):.4f}; '
                     f'below 0.99: {", ".join(valid["below_099"]) or "none"}.\n')
    found = {f'{name} {c}%': diagnostic('h6', f'cos_to_autograd{suffix}_c{c}')
             for name, suffix in (('identity', ''), ('H8', '_h8'), ('Cut', '_cut')) for c in (20, 100)}
    groups = [g for g in ('all', 'block_matrices', 'attn.c_q', 'attn.c_k', 'attn.c_v', 'attn.c_proj', 'mlp.c_fc',
                          'mlp.c_proj', 'wte', 'value_embeds', 'lm_head') if any(x and g in x for x in found.values())]
    if groups:
        rows = [[g] + [f'{x[g]:.4f}' if x and g in x else '-' for x in found.values()] for g in groups]
        texts.append('Estimators against autograd (identity path, H8 synthesis, PR #40\'s Cut), cosine of the weight '
                     'gradient\n\n' + markdown(['group'] + list(found), rows))
    sites = {c: (diagnostic('h6', f'cos_to_autograd_h8_c{c}') or {}).get('sites') for c in (20, 100)}
    if any(sites.values()):
        rows = [[site] + [f'{x[site]["median"]:.3f}' if x and site in x else '-' for x in sites.values()]
                for site in SITES]
        texts.append('Cut graph against the full graph: median per-token cosine of the credit per site\n\n'
                     + markdown(['site'] + [f'{c}%' for c in sites], rows))
    found = {c: diagnostic('h6', f'linearity_check_c{c}') for c in (20, 100)}
    if any(found.values()):
        keys = [f'{p} 0.01' for p in ('float32', 'bf16')]
        rows = [[site] + [f'{x[site][k]["nonlinearity"]["median"]:.3g}' if x and site in x and k in x[site] else '-'
                          for x in found.values() for k in keys] for site in SITES]
        texts.append('Nonlinearity |L+ + L- - 2 L0| / |L+ - L-| at the chain\'s probe size (0.01), median over '
                     'tokens\n\n' + markdown(['site'] + [f'{c}%: {k}' for c in found for k in keys], rows))
    found = {c: diagnostic('h6', f'synth_vs_real_c{c}') for c in (20, 100)}
    variants = sorted({k for x in found.values() if x for k in x if k != 'settings'})
    if variants:
        rows = [[site] + [f'{x[v][site]["real_vs_synth"]["median"]:.3f}' if x and v in x and site in x[v] else '-'
                          for x in found.values() for v in variants] for site in SITES]
        texts.append('H8 synthesis against the real chain, same rows and signs: median per-token cosine\n\n'
                     + markdown(['site'] + [f'{c}%: {v}' for c in found for v in variants], rows))
    if not texts:
        return None
    text = '\n'.join(texts)
    (out / 'h6.md').write_text(text)
    return text


def e7_table(out):
    """E7's diagnostic (H6 credit_dump on e1.h8_synth's checkpoints): per site and checkpoint, the variance ratio
    |g - p|^2 / |g|^2 of the exact credit against the identity-path guess (the control variate's variance relative to
    plain probing). Skip E7's runs if it is above 0.5 from 50% on (e7.precondition gates them)."""
    fractions = (0, 10, 20, 50, 100)
    found = {f: diagnostic('h6', f'credit_dump_c{f:02d}') for f in fractions}
    if not any(found.values()):
        return None
    cell = lambda f, site: (f'{found[f][site]["variance_ratio"]:.3f}'
                            if found[f] and site in found[f] else '-')
    rows = [[site] + [cell(f, site) for f in fractions] for site in SITES]
    text = 'Variance ratio |g - p|^2 / |g|^2 (lower: the guess explains more)\n\n' + \
        markdown(['site'] + [f'{f}%' for f in fractions], rows)
    (out / 'credit_dump.md').write_text(text)
    return text


EXTRA = {'e3': [e3_plot], 'e4': [e4_plots], 'e12': [val_time_plot]}
TABLES = {'e2': e2_table, 'e5': e5_table, 'e6': partial(delta_table, 'e6', baseline=FULL_CYCLE, extra=E6_COLUMNS),
          'e7': e7_tables, 'e8': e8_table, 'e10': partial(delta_table, 'e10', baseline=FULL_CYCLE, extra=[
              ('entries kept', logged_mean('agree_kept'))]),
          'e11': partial(delta_table, 'e11', baseline=FULL_CYCLE), 'e12': e12_table, 'e13': e13_tables,
          'e14': partial(gap_table, 'e14', extra=[('top cosine to the token loss',
                                                    logged_mean('nextlat', 'top_cosine'))]),
          'e15': partial(gap_table, 'e15', also=[('e1', 'h8_cut_noboost')],
                         extra=[('handoff cosine', logged_mean('handoff_cut_cosine'))]),
          'e16': e16_table, 'h6': lambda out: h6_tables(out)}
# The plan's reference cell for experiments whose baseline lives elsewhere (drawn in final.png).
BASELINES = {'e2': ('e1', 'bp_b8'), 'e3': ('e1', 'bp_b8'), 'e4': ('e1', 'bp_b8'), 'e8': ('e1', 'bp_b8'),
             'e9': ('e1', 'bp_b8'), 'e6': FULL_CYCLE, 'e7': FULL_CYCLE, 'e10': FULL_CYCLE, 'e11': FULL_CYCLE,
             'e13': FULL_CYCLE, 'e14': ('e1', 'h8_cut'), 'e15': ('e1', 'h8_cut'), 'e16': ('e1', 'h8_cut')}
FIGURES = {}
# Experiments with a second setting drawn apart against its own reference: variant prefix, reference, file.
SPLIT = {'e8': ('h8_', ('e8', 'h8_exact_control'), 'final_h8.png')}


def plot(experiment):
    out = ARTIFACTS / 'plots' / experiment
    out.mkdir(parents=True, exist_ok=True)
    made = []
    plt = None
    by_variant = counted(experiment)
    if by_variant:
        plt = pyplot()
        palette = colors(by_variant)
        title = f'{experiment}: {sum(len(v) for v in by_variant.values())} complete runs'
        baseline = None
        if experiment in BASELINES:
            source, variant = BASELINES[experiment]
            found = counted(source).get(variant)
            baseline = (f'baseline {source}.{variant}', found) if found else None
        prefix, (source, variant), name = SPLIT.get(experiment, (None, (None, None), None))
        main = {v: r for v, r in by_variant.items() if not (prefix and v.startswith(prefix))}
        second = {v: r for v, r in by_variant.items()
                  if prefix and v.startswith(prefix) and (experiment, v) != (source, variant)}
        if main:
            final_plot(plt, main, palette, out, title, baseline)
        if second:
            found = counted(source).get(variant)
            final_plot(plt, second, palette, out, f'{title}: the {prefix.rstrip("_")} setting',
                       (f'baseline {source}.{variant}', found) if found else None, name)
            made.append(name)
        curves_plot(plt, by_variant, palette, out, title)
        for extra in EXTRA.get(experiment, []):
            extra(plt, by_variant, palette, out)
        made += ['final.png', 'curves.png']
    if experiment in TABLES:
        text = TABLES[experiment](out)
        if text:
            print(text)
            made.append('tables')
    if experiment in FIGURES and FIGURES[experiment](plt or pyplot(), out):
        made.append('diagnostic figure')
    print(f'{experiment}: {", ".join(made) if made else "nothing to plot yet"} -> {out}')


def known(found):
    """The experiments with configs (follow-ups included), runs or a table, in the plan's order."""
    configs = REPO / 'configs'
    named = {p.name for d in (configs, configs / 'followups') if d.exists() for p in d.iterdir() if p.is_dir()}
    first = ('verify', 'h6')
    order = lambda e: (0, first.index(e)) if e in first else (1, int(e[1:])) if e[1:].isdigit() else (2, e)
    return sorted((named - {'followups'}) | found | set(TABLES) | set(FIGURES), key=order)


def main():
    found = {p.name for p in RUNS.iterdir()} if RUNS.exists() else set()
    names = known(found)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('experiment', choices=[*names, 'all'], metavar='experiment',
                        help=f"one of {', '.join(names)}, or 'all' (those with runs or a table)")
    args = parser.parse_args()
    experiments = sorted(found | set(TABLES) | set(FIGURES)) if args.experiment == 'all' else [args.experiment]
    try:
        for experiment in experiments:
            plot(experiment)
    except ModuleNotFoundError as exc:  # plain python3 on the CPU pod has no matplotlib
        raise SystemExit(f'{exc}: plots need matplotlib; run /opt/loud-cpu-venv/bin/python -m loud.plots '
                         f'(scripts/cpu_env.sh builds it)')


if __name__ == '__main__':
    main()
