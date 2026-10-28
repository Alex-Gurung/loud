"""Plots per experiment, read from the run files only (metrics.jsonl, heldout.jsonl, result.json, a diagnostic's JSON).

    python3 -m loud.plots e1            # artifacts/plots/e1/*.png (and *.md tables)
    python3 -m loud.plots all

Every training experiment gets final.png (each variant's final validation loss: a dot per seed, the mean +- sd) and
curves.png (train loss and held-out loss against sequences seen, the mean over seeds), from the runs the results
table counts (complete, scored, not failing their check). Experiments add their own series where the plan names one.
Diagnostics (e5, h6) are tables. Colors: the categorical slots in a fixed order by variant name, so a variant keeps its
color across an experiment's charts.
"""
import argparse
from pathlib import Path
import statistics

from .collect import counts, runs
from .common import read_json, read_jsonl
from .paths import ARTIFACTS, RUNS

SLOTS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
INK, INK_2, MUTED, SURFACE, GRID = '#0b0b0b', '#52514e', '#8a8984', '#fcfcfb', '#e6e5e1'
SITES = ['emb'] + [f'mid{i}' for i in range(8)]


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


def final_plot(plt, by_variant, palette, out, title, baseline=None):
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
    fig.savefig(out / 'final.png')
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
                         if 'mean_decay' not in h and h.get('heldout_loss') is not None})
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


def drift_plot(plt, by_variant, palette, out):
    """E6: the cosine of a full-cycle estimate on a held-out sequence with the momentum ANVIL II steps along."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    drawn = False
    for v in sorted(by_variant):
        series = [{m['step']: m['drift']['blend'] for m in metrics(r) if 'drift' in m} for r in by_variant[v]]
        xs, ys = seed_mean([s for s in series if s])
        if xs:
            ax.plot(xs, ys, color=palette[v], marker='o', markersize=3, label=v)
            drawn = True
    if not drawn:
        plt.close(fig)
        return
    ax.set_xlabel('update')
    ax.set_ylabel('cosine')
    ax.set_title('drift check: full-cycle estimate vs the momentum (mean over seeds)', loc='left', fontsize=9)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / 'drift.png')
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
    """E8: at each run's last refresh, per matrix type: the median curvature-to-noise ratio, damping factor and overlap
    of the tracked directions with the previous refresh's, means over seeds."""
    rows = []
    for v, runs_ in sorted(counted('e8').items()):
        last = [next((m['damping'] for m in reversed(metrics(r)) if 'damping' in m), None) for r in runs_]
        last = [d for d in last if d]
        for kind in sorted(last[0]) if last else []:
            values = [[d[kind].get(key) for d in last] for key in ('h2_over_c', 'factor', 'overlap')]
            rows.append([v, kind] + [f'{statistics.fmean(x):.4g}' if None not in x else '-' for x in values])
    if not rows:
        return None
    text = markdown(['variant', 'matrix', 'h2/c', 'factor', 'overlap'], rows)
    (out / 'damping.md').write_text(text)
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


def e5_table(out):
    """The plan's 9 x 2 table: per site, the median cosine of the recovered credit with a direct cycle's and the
    median condition number, on the 20% and 100% checkpoints (bf16 probes, as a kernel runs them)."""
    found = {c: diagnostic('e5', f'recovery_c{c}') for c in (20, 100)}
    if not any(found.values()):
        return None
    ridges = next(s for s in found.values() if s)['settings']['ridges']
    header = ['site'] + [f'{c}%: {what}' for c in found for what in
                         [f'cos ridge {r}' for r in ridges] + ['condition', 'guess cos']]
    rows = []
    for site in SITES:
        row = [site]
        for c, summary in found.items():
            s = (summary or {}).get(site)
            row += ([s[f'bf16 ridge {r}']['median'] for r in ridges] + [s['bf16 condition']['median'],
                    s['guess_vs_direct']['median']]) if s else ['-'] * (len(ridges) + 2)
        rows.append(row)
    text = markdown(header, rows)
    text += '\nDecision (plan E5): cosine >= 0.95 at all sites: implement; failing only at the top two or three ' \
            'sites: the hybrid; failing at most sites at 100%: drop.\n'
    (out / 'recovery.md').write_text(text)
    return text


def e7_table(out):
    """E7's diagnostic (H6 credit_dump): median per-token cosine of the identity-path guess with the true credit, per
    site and checkpoint. A saving to take if the median is above 0.9 at 20% and below 0.7 at 100%."""
    fractions = (0, 10, 20, 50, 100)
    found = {f: diagnostic('h6', f'credit_dump_c{f:02d}') for f in fractions}
    if not any(found.values()):
        return None
    rows = [[site] + [(found[f] or {}).get(site, {}).get('median', '-') for f in fractions] for site in SITES]
    text = markdown(['site'] + [f'{f}%' for f in fractions], rows)
    (out / 'credit_dump.md').write_text(text)
    return text


def e8_plot(plt, out):
    """E8's diagnostic (H6 direction_stats): the distribution of h^2 / c per matrix type, per checkpoint."""
    found = {c: diagnostic('h6', f'direction_stats_c{c}') for c in (20, 100)}
    found = {c: s for c, s in found.items() if s}
    if not found:
        return None
    types = ('attn.c_q', 'attn.c_k', 'attn.c_v', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj')
    fig, axes = plt.subplots(1, len(found), figsize=(5.5 * len(found), 3.6), squeeze=False)
    for ax, (c, summary) in zip(axes[0], found.items()):
        for y, kind in enumerate(types):
            values = [d['h2_over_c'] for name, ds in summary.items() if name.endswith(f'{kind}.weight')
                      for d in ds if d.get('h2_over_c')]
            ax.scatter(values, [y] * len(values), s=22, color=SLOTS[0], alpha=0.5, linewidth=0)
            if values:
                ax.scatter([statistics.median(values)], [y], s=60, marker='|', color=INK, linewidth=2)
        ax.set_xscale('log')
        ax.set_yticks(range(len(types)), types)
        ax.set_ylim(-0.6, len(types) - 0.4)
        ax.grid(axis='y', visible=False)
        ax.set_xlabel('h² / c along the top directions of the pending update (tick: median)')
        ax.set_title(f'{c}% checkpoint', loc='left', fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'h2_over_c.png')
    plt.close(fig)
    return True


EXTRA = {'e3': [e3_plot], 'e4': [e4_plots], 'e6': [drift_plot], 'e12': [val_time_plot]}
TABLES = {'e2': e2_table, 'e5': e5_table, 'e7': e7_table, 'e8': e8_table}
# The plan's reference cell for experiments whose baseline lives in E1.
BASELINES = {'e2': ('e1', 'bp_b8'), 'e4': ('e1', 'bp_b8'), 'e10': ('e1', 'bp_b8'), 'e3': ('e1', 'bp_b4'),
             'e8': ('e1', 'bp_b4'), 'e9': ('e1', 'bp_b4')}
FIGURES = {'e8': e8_plot}


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
        final_plot(plt, by_variant, palette, out, title, baseline)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('experiment', help="e1 ... e12, or 'all'")
    args = parser.parse_args()
    found = {p.name for p in RUNS.iterdir()} if RUNS.exists() else set()
    experiments = sorted(found | set(TABLES) | set(FIGURES)) if args.experiment == 'all' else [args.experiment]
    for experiment in experiments:
        plot(experiment)


if __name__ == '__main__':
    main()
