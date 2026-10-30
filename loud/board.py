"""The experiment board's data: JSON tables that the hosted board (a claude.ai dashboard; its page is board.html
beside this file) reads, built from the configs, the queue and the run files only. Each call rewrites them whole;
nothing here decides anything.

    python3 -m loud.board          # artifacts/board/{experiments,arms,runs,curves,jobs}.json

experiments  one row per experiment, in run order: its question, setup, reference cell, success test, gate, job
             counts by state and a verdict computed from the arm means (waiting until every arm has a result)
arms         one row per (experiment, variant): what it changes, seeds planned and counted, mean +- sd of the final
             validation loss and its difference from the reference cell
runs         one row per finished run (the results table's rows), with a short note for diagnostics and gates
curves       the held-out loss against sequences seen, the mean over an arm's counted seeds
jobs         one row per planned job (submitted or not, follow-ups included) with its queue state and why it waits
"""
import argparse
import statistics
from pathlib import Path

from .collect import counts, own_score, runs
from .common import read_json, read_jsonl, write_json
from .jobqueue import JobQueue
from .paths import ARTIFACTS, QUEUE, REPO, RUNS
from .submit import plan

OUT = ARTIFACTS / 'board'
SYNTH, CUT, BP8, BP4 = 'e1.h8_synth', 'e1.h8_cut', 'e1.bp_b8', 'e1.bp_b4_pr40'  # BP4: H8's recipe and batch
COMMON = {'sequences', 'heldout', 'heldout_every', 'final_dtype', 'checkpoints', 'checkpoint_optimizer', 'recipe',
          'compare_every', 'rank_every'}  # run options every arm shares, or logs only

# The plan's experiments (v2, sections 4 and 5), in the order the queue runs them.
EXPERIMENTS = [
    dict(id='verify', name='Smoke runs and verifications', priority='P0',
         question='Do the data, the scorer and each port reproduce their published numbers?',
         setup='Short complete runs of every method; the record rescored; backprop, the identity path and the Cut '
               'chain against their published logs.',
         success='Every check within its tolerance (rescoring 1e-4; backprop and identity 0.02; Cut prefix 0.003 '
                 'over 20 updates).', gate='Every experiment waits for these.'),
    dict(id='h6', name='H6 diagnostics', priority='P1',
         question='What do the trained checkpoints say before the experiments run?',
         setup='H8 validation against PR #40 (gates every H8 experiment); credit dumps, cosines, linearity, '
               'synthesis against the real chain, and E8\'s SNR spectrum, on E1\'s checkpoints.',
         success='h8_validation: every matrix at cosine 0.99 or more.', gate='After E1\'s seed-0 runs.'),
    dict(id='e1', name='E1 baselines', priority='P1',
         question='Where do backprop, H8 and the identity path land on the 10M set with 32 sequences held out?',
         setup='Backprop at batch 8 and 4; H8 synthesis (identity), cut mode, cut mode without Boost; the identity '
               'path.', success='bp_b8 about 3.71; h8_synth between 3.777 and 3.896; identity about 4.18.',
         gate='After the verifications.'),
    dict(id='e6', name='E6 where to put the missing rows', priority='P1', reference=SYNTH,
         question='With 512/K rows per token per site, is it better that tokens share rows (time) or cover the '
                  'basis between them (token)?',
         setup='H8 synthesis; time and token assignment at K = 4, 8, 16.',
         success='An arm at K = 4 within 0.02 of K = 1, and at K = 16 within 0.05.', gate='H8 validation.'),
    dict(id='e7', name='E7 identity-path control variate', priority='P1', reference=SYNTH,
         question='Does measuring only what the identity-path guess gets wrong cut the rows needed?',
         setup='H8 synthesis, token rows with the prior at m = 64 and 128.',
         success='The prior at m = 64 within 0.02 of the full cycle.',
         gate='Variance ratio at most 0.5 at 50% and 100% (e7.precondition).'),
    dict(id='e14', name='E14 NextLat latent consistency', priority='P1', reference=CUT,
         question='Does a next-latent prediction loss beside the token loss close the gap to backprop?',
         setup='H8 cut mode, EMA target, onset at the handoff; lambda 0.3, 1, 3.',
         success='Closes at least half the gap from e1.h8_cut to e1.bp_b4_pr40, or beats e1.h8_cut by 0.02.',
         gate='H8 validation.'),
    dict(id='e15', name='E15 Boost with a latent target', priority='P1', reference=CUT,
         question='Is Boost\'s logit-lens objective a crutch?',
         setup='H8 cut mode; Boost\'s sublayers predict the teacher\'s latent beside the lens or instead of it.',
         success='An arm beats e1.h8_cut by 0.02, or raises the handoff cosine at equal final loss.',
         gate='H8 validation.'),
    dict(id='e2', name='E2 smoothing with antithetic pairs', priority='P1', reference=BP8,
         question='Does the gradient of a blurred loss train better than the gradient of the loss?',
         setup='Backprop at batch 8; blur s = 0.03, 0.1, 0.3 at the nine sites.',
         success='Some s beats e1.bp_b8 by 0.02.', gate='Its smoke run.'),
    dict(id='e4', name='E4 measured step control', priority='P1', reference=BP8,
         question='Can accept-or-reject on measured held-out loss replace the hand-tuned warmdown?',
         setup='Backprop at batch 8; reject, adapt (x1.1 / x0.87), adapt without the warmdown.',
         success='adapt_nowd matches e1.bp_b8 within noise (0.01).', gate='Its smoke run.'),
    dict(id='e3', name='E3 population as local SGD', priority='P1', reference=BP8,
         question='Does any rule over several weight copies beat one trajectory at equal steps and data?',
         setup='Backprop; 4 copies from 20% at local batch 2; average every 1, 10, 50; select; ensemble.',
         success='Any arm beats e1.bp_b8 by 0.02.', gate='Its smoke run.'),
    dict(id='e5', name='E5 deep readout (diagnostic)', priority='P2',
         question='Can one probe cycle launched at S1 recover the credit at S2 to S8?',
         setup='H6 deep_readout_check on e1.h8_synth\'s 20% and 100% checkpoints.',
         success='All sites at cosine 0.95 or more: build it; failing at the top two or three: a hybrid; most '
                 'failing at 100%: drop.', gate='e1.h8_synth seed 0.'),
    dict(id='e10', name='E10 per-token credit shaping', priority='P2', reference=SYNTH,
         question='Do clipping credit norms or keeping only sign-agreeing entries help?',
         setup='H8 synthesis; clip at 3x the batch median; keep entries where 70% of tokens agree.',
         success='Our reading: an arm beats e1.h8_synth by 0.02 (v2 states none).', gate='H8 validation.'),
    dict(id='e11', name='E11 per-token runway weights', priority='P2', reference=SYNTH,
         question='Does weighting tokens by how far their loss keeps falling along their credit help?',
         setup='H8 synthesis with reforward; moves of 0.01 to 0.08 RMS per coordinate.',
         success='Beats e1.h8_synth by 0.02.', gate='H8 validation.'),
    dict(id='e13', name='E13 n-gram probe budgets', priority='P2', reference=SYNTH,
         question='Can tokens an n-gram model and the network both find easy get 64 rows instead of 512?',
         setup='H8 synthesis; theta = 2, 3, 4 nats; each gated on the tiers diagnostic.',
         success='Our reading: within 0.02 of e1.h8_synth at fewer rows (v2 states none).',
         gate='Low tier under 10% of the credit energy with over 40% of the tokens.'),
    dict(id='e16', name='E16 does H8 rank architectures', priority='P2', reference=CUT,
         question='Does H8 order the fixed model and track-3 record #5\'s design as backprop does?',
         setup='Both models under backprop and under H8 cut mode, PR #40\'s recipe at batch 4, two seeds.',
         success='H8 reproduces backprop\'s order (leaderboard on 30M: 3.211 against 3.338).',
         gate='The track-3 smoke run and H8 validation.'),
    dict(id='e17', name='E17 where the cut loses credit (added)', priority='P1', reference=CUT,
         question='Which cross-token path carries the credit H8 loses to backprop: other tokens\' keys or values?',
         setup='H8 cut mode with the values, the keys or both left live (both: the full graph), PR #40\'s recipe.',
         success='Our reading: the share of the e1.h8_cut -> e1.bp_b4_pr40 gap (0.16) each path recovers.',
         gate='H8 validation.'),
    dict(id='e18', name='E18 the 30M length check (added)', priority='P1', reference='e18.h8_synth',
         question='Do the 10M answers hold with 3x the data (the 30M set)?',
         setup='H8 synthesis with all rows and K = 4 by token and time; H8 cut with and without NextLat (0.3); '
               'matched backprop; two seeds.',
         success='Our reading: the 10M gaps (E6 +0.15/+0.19, E14 0.00, H8 to backprop +0.16) grow, hold or shrink.',
         gate='H8 validation.'),
    dict(id='e19', name='E19 the rate check (added)', priority='P1', reference=CUT,
         question='Is part of H8\'s gap to backprop, or of E6\'s partial-row cost, the record\'s rate?',
         setup='H8 cut mode at 0.5-2x every rate; E6\'s K = 4 by token at 0.35-1.4x; PR #40\'s recipe otherwise.',
         success='Our reading: the best rate per estimator and the gap left there (cut to e1.bp_b4_pr40, K = 4 to '
                 'e1.h8_synth).',
         gate='H8 validation.'),
    dict(id='e20', name='E20 the batch check (added)', priority='P1', reference='e20.synth_b8',
         question='Does E6\'s partial-row cost shrink with a larger batch (more tokens averaged per update)?',
         setup='H8 synthesis at batch 8 and 16 (606 and 303 updates on 10M), every row and K = 4 by token; three seeds.',
         success='Our reading: K = 4\'s cost at batch 16 against +0.150 at batch 4 (E6\'s bar: 0.02).',
         gate='H8 validation.'),
    dict(id='e8', name='E8 SNR-scaled steps (follow-up)', priority='P1', reference=BP8,
         question='Does an orthogonalizing optimizer amplify estimator noise, and does SNR scaling fix it?',
         setup='Control, SNR and rate arms in backprop and in H8 synthesis at K = 4; built if the diagnostic says.',
         success='With partial rows, SNR scaling recovers at least half the loss the partial rows cost and beats '
                 'the rate control.', gate='H6 snr_spectrum; submitted by hand from configs/followups/e8.'),
    dict(id='e9', name='E9 pre-polar true Fisher (follow-up)', priority='P3', reference=BP8,
         question='Does preconditioning the momentum by the true Fisher help?',
         setup='Backprop at batch 8; (F + delta I)^(-1/2) before ANVIL II\'s orthogonalization.',
         success='Beats e1.bp_b8 by 0.02.', gate='E8 shows per-direction scaling matters.'),
    dict(id='e12', name='E12 30M (follow-up)', priority='P3',
         question='How do the winners do on the 30M set?', setup='Backprop on 30M with PR #46\'s recipe.',
         success='-', gate='Winners first.'),
]


def mean_sd(values):
    return (statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0) if values else (None, None)


def change_of(spec):
    """What a config sets beyond the common run options, as 'key=value' text."""
    args = {k: v for k, v in (spec.get('args') or {}).items() if k not in COMMON}
    return ', '.join(f'{k}={v}' for k, v in args.items()) or 'defaults'


def activity(arm, runs_):
    """(acted, note) for a rule arm whose logs say whether its rule changed anything: E11's runway (the share of tokens
    at the grid's largest move: all of them means every weight is 1), E10's clip (its gradient's cosine to the
    unclipped one) and E10's agree (the share of weight-update entries it kept). (True, '') for any other arm."""
    logs = [m for r in runs_ for m in read_jsonl(Path(r['run_dir']) / 'metrics.jsonl')]
    if arm.startswith('e11.runway'):
        shares = [s[-1] for m in logs if 'runway' in m for s in m['runway'].values()]
        if shares:
            top = statistics.fmean(shares)
            tail = ', so the weights are almost all 1' if top >= 0.99 else ''
            return top < 0.99, f'{top:.2%} of tokens pick the grid\'s largest move{tail}'
    if arm.startswith('e10.clip'):
        cos = [m['cosine_to_identity'] for m in logs if 'cosine_to_identity' in m]
        if cos:
            mean = statistics.fmean(cos)
            return mean < 0.995, f'its gradient keeps cosine {mean:.4f} to the unclipped one'
    if arm.startswith('e10.agree'):
        kept = [m['agree_kept'] for m in logs if 'agree_kept' in m]
        if kept:
            return True, f'it keeps {statistics.fmean(kept):.2%} of the weight-update entries'
    return True, ''


NO_LOSS = ('gate', 'diagnose', 'compare')  # tasks whose runs report no validation loss


def chain_pairs(by_arm, experiment, variants):
    """'variant +d (seeds ...)' for each chain arm with runs, d its mean difference from e17.chain_record on the
    seeds both have: the chain arms are paired with the record draw for draw, so a two-seed mean against a one-seed
    arm would add a seed's noise to a difference expected near 0."""
    record = {r['seed']: r['val_loss'] for r in by_arm.get('e17.chain_record', []) if counts(r)}
    out = []
    for v in variants:
        mine = {r['seed']: r['val_loss'] for r in by_arm.get(f'{experiment}.{v}', []) if counts(r)}
        seeds = sorted(set(mine) & set(record))
        if seeds:
            d = statistics.fmean(mine[s] - record[s] for s in seeds)
            out.append(f'{v} {d:+.4f} (seed{"s" if len(seeds) > 1 else ""} {", ".join(map(str, seeds))})')
    return out


def seed_mean(runs_, seeds):
    vals = [r['val_loss'] for r in runs_ if r['seed'] in seeds and counts(r)]
    return statistics.fmean(vals) if vals else None


def verdict(e, by_key, by_arm=None, runs_root=RUNS):
    """(state, sentence) for an experiment from its arms' means (by_key: arm key -> arm row; by_arm: arm key -> its
    runs): 'waiting' until every arm has a counted run; 'null' when every arm's rule never acted."""
    by_arm = by_arm or {}
    if e['id'] == 'verify':
        return 'n/a', 'Checks: each passes or fails on its own (see the jobs).'
    # an experiment's own arms, and its follow-ups once they have run (E14's second stage)
    if e['id'] == 'e5':  # v2's decision, on the chain's way of differencing in bf16 (readout_delta)
        found, plain = {}, {}
        for c in ('c20', 'c100'):
            path = Path(runs_root) / 'e5' / f'readout_delta_{c}' / 's0' / 'deep_readout_check.json'
            if path.exists():
                data = {k: v for k, v in read_json(path).items() if k.startswith('S')}
                found[c] = {k: v['cosine ridge 0.001']['median'] for k, v in data.items()}  # the build's ridged solve
                plain[c] = min(v['cosine ridge 0.0']['median'] for v in data.values())
        if len(found) < 2:
            return 'waiting', 'Waiting for the bf16 delta readouts at 20% and 100%.'
        low = {c: min(v.values()) for c, v in found.items()}
        failing = {c: sorted(k for k, x in v.items() if x < 0.95) for c, v in found.items()}
        if not any(failing.values()):
            note = (f'Every site at median cosine 0.95 or more with the build\'s ridge (lowest {low["c20"]:.4f} at 20%, '
                    f'{low["c100"]:.4f} at 100%; unridged {plain["c20"]:.4f} and {plain["c100"]:.4f}). v2: ask the '
                    f'maintainers, then build; built as a measurement, not admissible until the linear-consumer '
                    f'question is settled.')
            paired = chain_pairs(by_arm, 'e5', ['chain_deep', 'chain_deep_paired'])
            if paired:
                note += ' On the chain, against the record unchanged on the same seeds: ' + ', '.join(paired) + '.'
            return 'met', note
        return 'not met', f'Sites under 0.95: {failing}.'
    if e['id'] == 'e9':  # gated on E8's intervention, which did not help
        return 'skipped', 'Its gate (E8 shows per-direction scaling matters) did not pass: E8 is not met.'
    if e['id'] == 'e12':
        return 'skipped', 'No experiment produced a winner to take to 30M; E18 runs the length check in H8.'
    if e['id'] == 'e17':  # read as arms land: H8 arms against the cut's gap, chain arms against the record's chain
        low, high = by_key.get(BP4, {}).get('val_mean'), by_key.get(CUT, {}).get('val_mean')
        mine = {a['variant']: a for a in by_key.values() if a['experiment'] == 'e17' and a['task'] not in NO_LOSS}
        done = {v: a for v, a in mine.items() if a['val_mean'] is not None}
        if low is None or high is None or not done:
            return 'waiting', f'Waiting for {CUT}, {BP4} and the first arms.'
        h8_ = {v: a for v, a in done.items() if a['task'] == 'h8'}
        top = sorted(h8_.items(), key=lambda kv: kv[1]['val_mean'])
        shares = ', '.join(f'{v} {(high - a["val_mean"]) / (high - low):.0%}' for v, a in top)
        out = f'Share of the {high - low:.3f} gap recovered: {shares}.'
        paired = chain_pairs(by_arm, 'e17', [v for v, a in done.items() if a['task'] == 'cut' and v != 'chain_record'])
        if paired:
            out += ' Real chain, paired with chain_record seed by seed: ' + ', '.join(paired) + '.'
        left = len(mine) - len(done)
        return 'n/a', out + (f' ({left} arms still running or queued.)' if left else '')
    if e['id'] == 'e19':
        cells19 = {a['variant']: a for a in by_key.values() if a['experiment'] == 'e19' and a['val_mean'] is not None}
        out = []
        for prefix, one, target in (('cut_lr', CUT, BP4), ('token_k4_lr', 'e6.token_k4', SYNTH)):
            arms = {v: a['val_mean'] for v, a in cells19.items() if v.startswith(prefix)}
            if one in by_key and by_key[one]['val_mean'] is not None:
                arms[prefix + '1'] = by_key[one]['val_mean']
            if len(arms) < 2 or by_key.get(target, {}).get('val_mean') is None:
                continue
            v, best = min(arms.items(), key=lambda kv: kv[1])
            rate = v[len(prefix):].replace('p', '.')
            out.append(f'{prefix[:-3]}: best at {rate}x ({best - arms.get(prefix + "1", best):+.4f} '
                       f'against 1x), {best - by_key[target]["val_mean"]:+.3f} from {target}')
        return ('n/a', '; '.join(out) + '.') if out else ('waiting', 'Waiting for the first rates.')
    if e['id'] == 'e20':  # the test is batch 16's pair; batch 8's is read as it lands
        base = by_key.get(SYNTH, {}).get('val_mean')
        found = {}
        for batch in (8, 16):
            a, b = by_key.get(f'e20.token_k4_b{batch}', {}), by_key.get(f'e20.synth_b{batch}', {})
            if a.get('val_mean') is not None and b.get('val_mean') is not None and base is not None:
                found[batch] = (a['val_mean'] - b['val_mean'], b['val_mean'] - base)
        text = '; '.join(f'at batch {n} K = 4 costs {c:+.4f} (every row there: {d:+.4f} from batch 4)'
                         for n, (c, d) in sorted(found.items()))
        if 16 not in found:
            return 'waiting', 'Waiting for batch 16\'s two arms' + (f'; so far {text}.' if text else '.')
        return ('met' if found[16][0] <= 0.02 else 'not met'), f'Batch 4: K = 4 costs +0.150; {text}.'
    # (gates and diagnostics have no loss: they never hold a verdict back)
    own = [a for a in by_key.values() if a['experiment'] == e['id'] and a['task'] not in NO_LOSS]
    cells = {a['variant']: a for a in own if not a['followup'] or a['val_mean'] is not None}
    if not own and any(a['experiment'] == e['id'] for a in by_key.values()):
        return 'n/a', 'Diagnostics and checks: read each arm\'s summary.'
    if not cells or any(a['val_mean'] is None for a in cells.values()):
        return 'waiting', 'Waiting for every arm to finish.'
    ref = e.get('reference')
    base = by_key[ref]['val_mean'] if ref in by_key else None
    best = min(cells.values(), key=lambda a: a['val_mean'])
    if e['id'] in ('e10', 'e11'):
        acts = {v: activity(a['key'], by_arm.get(a['key'], [])) for v, a in cells.items()}
        notes = '; '.join(f'{v}: {n}' for v, (_, n) in acts.items() if n)
        if base is not None and not any(acted for acted, _ in acts.values()):
            return 'null', f'The rule never acted ({notes}); best arm at {best["val_mean"] - base:+.4f}.'
        if notes:
            e = dict(e, note=notes)
    if e['id'] in ('e2', 'e3', 'e10', 'e11'):
        if base is None:
            return 'waiting', f'Waiting for {ref}.'
        ok = best['val_mean'] <= base - 0.02
        note = f' ({e["note"]})' if e.get('note') else ''
        return ('met' if ok else 'not met'), \
            f'Best arm {best["variant"]} at {best["val_mean"] - base:+.4f} (needs -0.02){note}.'
    if e['id'] == 'e4':
        a = cells.get('adapt_nowd')
        if base is None or a is None:
            return 'waiting', f'Waiting for {ref}.'
        d = a['val_mean'] - base
        return ('met' if abs(d) <= 0.01 else 'not met'), f'adapt_nowd at {d:+.4f} (needs within 0.01).'
    if e['id'] == 'e6':
        if base is None:
            return 'waiting', f'Waiting for {ref}.'
        k4 = [cells[v]['val_mean'] - base for v in cells if v.endswith('_k4')]
        k16 = [cells[v]['val_mean'] - base for v in cells if v.endswith('_k16')]
        if not k4 or not k16:
            return 'waiting', 'Waiting for the K = 4 and K = 16 arms.'
        k4, k16 = min(k4), min(k16)
        return ('met' if k4 <= 0.02 and k16 <= 0.05 else 'not met'), \
            f'Best K = 4 at {k4:+.4f} (needs 0.02 or less), best K = 16 at {k16:+.4f} (needs 0.05 or less).'
    if e['id'] == 'e7':
        a = cells.get('prior_m64')
        if base is None or a is None:
            return 'waiting', f'Waiting for {ref}.'
        d = a['val_mean'] - base
        return ('met' if d <= 0.02 else 'not met'), f'prior_m64 at {d:+.4f} (needs 0.02 or less).'
    if e['id'] == 'e13':
        if base is None:
            return 'waiting', f'Waiting for {ref}.'
        d = best['val_mean'] - base
        return ('met' if d <= 0.02 else 'not met'), f'Best theta {best["variant"]} at {d:+.4f} (needs 0.02 or less).'
    if e['id'] in ('e14', 'e15'):
        low = by_key.get(BP4, {}).get('val_mean')
        if base is None or low is None:
            return 'waiting', f'Waiting for {CUT} and {BP4}.'
        closed = (base - best['val_mean']) / (base - low) if base != low else 0.0
        ok = best['val_mean'] <= base - 0.02 or (e['id'] == 'e14' and closed >= 0.5)
        return ('met' if ok else 'not met'), \
            f'Best arm {best["variant"]} at {best["val_mean"] - base:+.4f}, ' \
            f'closing {closed:.0%} of the gap to backprop.'
    if e['id'] == 'e8':  # the H8 setting carries the success test
        h = {v: cells[v]['val_mean'] for v in ('h8_exact_control', 'h8_control', 'h8_snr', 'h8_rate') if v in cells}
        if len(h) < 4:
            return 'waiting', 'Waiting for the H8 setting\'s four arms.'
        cost = h['h8_control'] - h['h8_exact_control']
        recovered = (h['h8_control'] - h['h8_snr']) / cost if cost else 0.0
        ok = recovered >= 0.5 and h['h8_snr'] < h['h8_rate']
        return ('met' if ok else 'not met'), \
            f'Partial rows cost {cost:+.4f}; SNR scaling recovers {recovered:.0%} of it (needs 50%) and ' \
            f'{"beats" if h["h8_snr"] < h["h8_rate"] else "trails"} the rate control ({h["h8_snr"]:.4f} against ' \
            f'{h["h8_rate"]:.4f}).'
    if e['id'] == 'e18':
        g = lambda v: cells[v]['val_mean'] if v in cells else None
        pairs = [('token_k4', 'h8_synth'), ('time_k4', 'h8_synth'), ('nextlat_0p3', 'h8_cut'), ('nextlat_1', 'h8_cut'), ('nextlat_3', 'h8_cut'),
                 ('h8_cut', 'bp_b4_pr40'), ('live_keys', 'h8_cut'), ('live_keys_b357', 'h8_cut')]
        found = [f'{a} - {b} {g(a) - g(b):+.3f}' for a, b in pairs if g(a) is not None and g(b) is not None]
        return 'n/a', 'At 30M: ' + '; '.join(found) + ' (10M: +0.150, +0.184, -0.001, +0.160).'
    if e['id'] == 'e16':  # the fixed model's cells on the track-3 arms' seeds (0-1)
        seeds = {0, 1}
        four = [seed_mean(by_arm.get(k, []), seeds) for k in (BP4, 'e16.bp_track3', CUT, 'e16.h8_track3')]
        cells = {k: v for k, v in cells.items() if not k.endswith('_plain')}  # the closer pair reads held-out loss
        if any(v is None for v in four):
            return 'waiting', 'Waiting for the four arms.'
        bp, h8 = four[0] - four[1], four[2] - four[3]
        return ('met' if (bp > 0) == (h8 > 0) else 'not met'), \
            f'Track-3 gain on seeds 0-1: backprop {bp:+.4f}, H8 {h8:+.4f} (leaderboard on 30M: +0.127).'
    return 'n/a', 'Read the arms and their checks.'


def build(configs=REPO / 'configs', queue_root=QUEUE, runs_root=RUNS, out=OUT):
    planned = [(j, False) for j in plan([configs], None, None)]
    followups = configs / 'followups'
    if followups.exists():
        planned += [(j, True) for j in plan([followups], None, None)]
    queue = JobQueue(queue_root, create=False) if Path(queue_root).exists() else None
    state_of, finished = {}, queue.finished() if queue else ({}, {})
    if queue:
        for state in ('queued', 'running', 'done', 'failed', 'cancelled'):
            for job in queue.jobs(state):
                state_of[job['id']] = (state, job)
    rows, _ = runs(runs_root)
    by_arm = {}
    for r in rows:
        by_arm.setdefault(f'{r["experiment"]}.{r["variant"]}', []).append(r)

    jobs = []
    for job, followup in planned:
        state, record = state_of.get(job['id'], (None, None))
        detail = ''
        if state == 'queued':
            readiness, why = queue.dependency_state(record, finished)
            state, detail = {'ready': ('ready', ''), 'waiting': ('waiting', f'for {why}'),
                             'blocked': ('blocked', why), 'skipped': ('skipped', why)}[readiness]
        elif state is None:
            state = 'follow-up' if followup else 'not submitted'
        elif state == 'failed':
            detail = ' '.join(str(x) for x in (record.get('reason'), (record.get('log_tail') or [''])[-1]) if x)[:160]
        spec = job['spec']
        jobs.append(dict(id=job['id'], experiment=spec['experiment'], variant=spec['variant'], seed=spec['seed'],
                         task=spec['task'], state=state, detail=detail, priority=job['priority'],
                         followup=followup))

    arms, seen = [], {}
    for job, followup in planned:
        spec = job['spec']
        key = f'{spec["experiment"]}.{spec["variant"]}'
        if key in seen:
            seen[key]['seeds'] += 1
            continue
        counted = [r for r in by_arm.get(key, []) if counts(r)]
        mean, sd = mean_sd([r['val_loss'] for r in counted])
        seen[key] = dict(key=key, experiment=spec['experiment'], variant=spec['variant'], task=spec['task'],
                         change=change_of(spec), seeds=1, counted=len(counted), val_mean=mean, val_sd=sd,
                         followup=followup, priority=job['priority'])
        arms.append(seen[key])
    by_key = {a['key']: a for a in arms}
    state_counts = {}
    for j in jobs:
        a = seen[f'{j["experiment"]}.{j["variant"]}']
        a.setdefault('states', {}).setdefault(j['state'], 0)
        a['states'][j['state']] += 1
        state_counts.setdefault(j['experiment'], {}).setdefault(j['state'], 0)
        state_counts[j['experiment']][j['state']] += 1

    experiments = []
    for order, e in enumerate(EXPERIMENTS):
        ref = by_key.get(e.get('reference') or '')
        for a in arms:
            if a['experiment'] == e['id']:
                base = ref['val_mean'] if ref else None
                a['reference'] = e.get('reference') or ''
                a['delta'] = None if base is None or a['val_mean'] is None else a['val_mean'] - base
        state, sentence = verdict(e, by_key, by_arm, runs_root)
        mine = [a for a in arms if a['experiment'] == e['id']]
        own = [j for j in jobs if j['experiment'] == e['id'] and not j['followup']
               and j['task'] not in ('diagnose', 'gate', 'compare')]
        if own and all(j['state'] == 'skipped' for j in own):  # its gate said skip: nothing will run
            state, sentence = 'skipped', 'Skipped by its gate: ' + ', '.join(sorted({j['detail'] for j in own}))
        label, title = e['name'].split(' ', 1) if e['id'][0] in 'eh' else ('Checks', e['name'])
        n = state_counts.get(e['id'], {})
        experiments.append(dict(
            id=e['id'], order=order, name=e['name'], label=label, title=title[:1].upper() + title[1:],
            priority=e['priority'], question=e['question'],
            followup=bool(mine) and all(a['followup'] for a in mine),
            setup=e['setup'], success=e['success'], gate=e['gate'], reference=e.get('reference') or '',
            reference_val=ref['val_mean'] if ref else None, reference_n=ref['counted'] if ref else 0,
            verdict=state, verdict_note=sentence, jobs=sum(n.values()),
            **{k.replace(' ', '_').replace('-', '_'): n.get(k, 0) for k in (
                'done', 'running', 'ready', 'waiting', 'blocked', 'skipped', 'failed', 'cancelled', 'not submitted',
                'follow-up')}))
    for a in arms:
        a.pop('states', None)
        a.setdefault('reference', '')
        a.setdefault('delta', None)

    run_rows = []
    for r in rows:
        result = read_json(Path(r['run_dir']) / 'result.json')
        note = ', '.join(f'{k} {result[k]:.4g}' for k in ('min_cosine',) if isinstance(result.get(k), (int, float)))
        run_rows.append(dict(key=f'{r["experiment"]}.{r["variant"]}.s{r["seed"]}', arm=f'{r["experiment"]}.'
                             f'{r["variant"]}', experiment=r['experiment'], variant=r['variant'], seed=r['seed'],
                             val_loss=r['val_loss'], test_loss=r['test_loss'], heldout_loss=r['heldout_loss'],
                             train_minutes=None if r['train_seconds'] is None else r['train_seconds'] / 60,
                             check=r['check'] or '', counted=counts(r), note=note))

    curves = []
    for key, members in by_arm.items():
        series = []
        for r in filter(counts, members):
            run_dir = Path(r['run_dir'])
            seqs = {m['step']: m.get('seqs') for m in read_jsonl(run_dir / 'metrics.jsonl')}
            points = {seqs.get(h['step'], 0) or 0: h['heldout_loss'] for h in read_jsonl(run_dir / 'heldout.jsonl')
                      if own_score(h) and h.get('heldout_loss') is not None}
            if points:
                series.append(points)
        if not series:
            continue
        xs = sorted(set.intersection(*(set(s) for s in series)))
        stride = max(1, len(xs) // 40)
        exp, variant = key.split('.', 1)
        curves += [dict(arm=key, experiment=exp, variant=variant, seqs=x,
                        heldout=statistics.fmean(s[x] for s in series),
                        seeds=len(series)) for x in xs[::stride] + ([xs[-1]] if xs[::stride][-1:] != xs[-1:] else [])]

    out.mkdir(parents=True, exist_ok=True)
    tables = dict(experiments=experiments, arms=arms, runs=run_rows, curves=curves, jobs=jobs)
    for name, data in tables.items():
        write_json(out / f'{name}.json', data)
    return {name: len(data) for name, data in tables.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    print(build(out=args.out), '->', args.out)


if __name__ == '__main__':
    main()
