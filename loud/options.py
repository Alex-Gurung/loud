"""Task registry and option tables (no torch import, so submission checks run on the CPU pod)."""
import json
import re

from .launch import settings_fields
from .paths import PINS, REPO

TASKS = {
    'proxy': ('loud.proxy', 'run'),               # H1 backprop proxy
    'identity': ('loud.identity', 'run'),         # H5 identity-path forward-only estimator (PR #26)
    'h8': ('loud.h8', 'run'),                     # H8 autograd emulation of Boost and Cut (development only)
    'population': ('loud.population', 'run'),     # E3: copies of the backprop proxy on split data
    'cut': ('loud.cut', 'run'),                   # H5 the record's Boost -> Cut chain (PR #40)
    'score': ('loud.score', 'run_task'),          # official rescoring of a checkpoint
    'reference': ('loud.reference', 'run_task'),  # a pinned external script, unmodified, on one GPU
    'compare': ('loud.compare', 'run_task'),      # two runs' per-update losses side by side
    'diagnose': ('loud.diagnostics', 'run_task'), # H6: one diagnostic tool on one checkpoint
    'gate': ('loud.gate', 'run_task'),            # a plan decision rule on finished diagnostics, as a check
    'noop': ('loud.noop', 'run_task'),            # queue/worker plumbing check, no GPU work
}

# Training ranges (the first N rows of train.pt) and model presets (n_layer, width: the Dust paper's ladder plus PR
# #26's 'tiny'; only '38m' is the benchmark model, the others are for smoke tests).
RANGES = {'1M': 488, '10M': 4880, '30M': 14648}
PRESETS = {'tiny': (4, 128), '2m': (2, 128), '7m': (4, 256), '38m': (8, 512), '243m': (16, 1024)}
CHECKPOINT_DTYPES = ('bfloat16', 'float32')

# What each task's result.json holds that `expect` may check.
TRAINING_RESULTS = ('val_loss', 'test_loss', 'heldout_loss', 'train_seconds', 'train_loss_last', 'peak_cuda_gb')
RESULTS = {'proxy': TRAINING_RESULTS, 'identity': TRAINING_RESULTS, 'h8': TRAINING_RESULTS, 'population': TRAINING_RESULTS,
           'cut': TRAINING_RESULTS + ('val_loss_plain', 'test_loss_plain', 'holdout_plain', 'holdout_merged',
                                      'merge_seconds'),
           'score': ('val_loss', 'test_loss'),
           'reference': ('val_loss', 'test_loss', 'train_seconds'),
           'compare': ('train_loss_max_diff', 'train_loss_mean_diff', 'train_loss_max_diff_all', 'val_loss_diff'),
           'diagnose': (),
           'gate': (),  # its args name its values
           'noop': ('val_loss',)}


def range_size(sequences):
    n = RANGES.get(sequences) if isinstance(sequences, str) else sequences
    if not isinstance(n, int) or not 0 < n <= RANGES['30M']:
        raise ValueError(f'sequences {sequences!r}: give one of {sorted(RANGES)} or a row count up to {RANGES["30M"]}')
    return n


def number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_expect(task, expect, metrics=None):
    """expect: {metric: [target, tolerance] or {min: a, max: b} (either or both)} over the metrics the task's
    result.json holds (`metrics`, when the task names them in its args, as gate does)."""
    if expect is None:
        return
    if not isinstance(expect, dict):
        raise ValueError('expect must map metrics to [target, tolerance] or {min, max}')
    allowed = RESULTS[task] if metrics is None else tuple(metrics)
    for metric, bound in expect.items():
        if metric not in allowed:
            raise ValueError(f'expect: {task} results have no {metric!r}; checkable: {list(allowed)}')
        interval = isinstance(bound, dict) and bound and set(bound) <= {'min', 'max'} and all(
            number(v) for v in bound.values())
        if not interval and (not isinstance(bound, list) or len(bound) != 2 or not all(number(v) for v in bound)
                             or bound[1] < 0):
            raise ValueError(f'expect: {metric} needs [target, tolerance >= 0] or {{min, max}}, not {bound!r}')


# Every training task (loud/train.py): data range, H3, H4, monitoring, scoring.
TRAINING = dict(
    sequences='10M',            # training range: the first N rows of train.pt ('1M', '10M', '30M' or a count)
    heldout=32,                 # H3: last sequences of the order that feed no update (0: all feed updates)
    batch=8,                    # sequences per update
    preset='38m',               # model size; only '38m' is the benchmark model (the others are for smoke tests)
    stop_after=0,               # stop after this many updates, schedule unchanged (log matching); 0 = full epoch
    checkpoints=False,          # H4 fraction checkpoints (ckpt_00/10/20/50/100.pt)
    checkpoint_dtype='bfloat16',
    checkpoint_optimizer=False, # also ANVIL II's state with each checkpoint (about 0.2 GB each), for H6 direction_stats
    means=False,                # H4: the record's nine trajectory means from 60%, scored on the held-out set at the end
    save_means=False,           # also write them to means.pt (2-D weights in checkpoint_dtype: about 0.7 GB in bf16)
    heldout_every=0,            # score the held-out set every N updates (monitoring only)
    eval_every=0,               # untimed val.pt curve every N updates: a plot only, never a decision
    test=True,                  # also score test.pt at the end
    official_eval=True,
)

PROXY = dict(
    TRAINING,
    recipe='boostcut',          # reference trainer: 'boostcut' (backprop_preset.py), 'pr40' (PR #40's rates, the 10M
                                # record) or 'pr46' (PR #46's, the 30M record)
    micro_batch=1,              # sequences per forward/backward (1 = one GPU's share in the 8-GPU script)
    lr_mult=1.0,                # scales both Adam and ANVIL II rates, as backprop_preset.py --lr-mult
    per_sequence_grads=False,   # H2
    grad_noise_every=0,         # log the H2 across-sequence spread every N updates (needs per_sequence_grads)
    warmdown=None,              # the recipe's warmdown fraction (None), or another; 0 holds every rate constant
    # one-change variants (loud/variants.py), at most one per run:
    smooth_tau=0.0,             # E2: antithetic blur at the nine probing sites, in PR #40 probe steps (0: off)
    step_control='none',        # E4: 'reject' undoes an update that raised its batch's loss; 'adapt' also scales rates
    accept_up=1.05,             # E4 adapt: rate multiplier after an accepted update
    accept_down=0.8,            # E4 adapt: rate multiplier after a rejected one
    token_shaping='none',       # E10: 'cap' or 'agree', on every nn.Linear weight gradient
    cap_quantile=0.95,          # E10 cap: each token's credit norm is capped at this quantile over the batch
    agree_threshold=0.7,        # E10 agree: the share of tokens whose contributions must agree in sign
    agree_over='active',        # E10 agree: 'active' (tokens with a nonzero contribution) or 'all' (the plan's literal)
    curvature_damping=0.0,      # E8: lambda, in squared ANVIL II steps (loud/curvature.py); 0 = off
    damp_directions=16,         # E8: tracked top singular directions of each block matrix's pending update
    damp_every=50,              # E8: updates between refreshes of the directions and their curvature
    damp_where='step',          # E8: 'step' (after the orthogonalization) or 'pending' (the plan's literal placement)
    fisher_damping=0.0,         # E9: true-Fisher left factor in place of ANVIL II's polar (loud/fisher.py); 0 = off
    end_merge=False,            # the record's end merge of the nine means after the last update (recipes pr40, pr46)
)
STEP_CONTROLS, SHAPINGS = ('none', 'reject', 'adapt'), ('none', 'cap', 'agree')
# Each variant option, and the setting (of another option) it acts under.
VARIANT_OPTIONS = {'accept_up': ('step_control', 'adapt'), 'accept_down': ('step_control', 'adapt'),
                   'cap_quantile': ('token_shaping', 'cap'), 'agree_threshold': ('token_shaping', 'agree'),
                   'agree_over': ('token_shaping', 'agree')}

# E3 (loud/population.py): copies of the proxy branched from one trajectory. Only POPULATION_FREE options may change;
# the rest stay at their defaults (one change per variant, and the population loop has no H4 or monitoring).
POPULATION = dict(
    PROXY,
    batch=4,                    # the batch-4 baseline's (e1.bp_b4)
    branch=0.2,                 # fraction of the single run's updates done before the copies branch
    copies=4,
    rule=None,                  # 'average' | 'select' | 'ensemble'
    every=10,                   # N: updates per copy between averages or selections (ensemble: between logs)
)
POPULATION_FREE = ('sequences', 'heldout', 'batch', 'preset', 'test', 'official_eval', 'recipe', 'micro_batch',
                   'lr_mult', 'branch', 'copies', 'rule', 'every', 'checkpoints', 'checkpoint_dtype')

# H8 (loud/h8.py) on PR #40's recipe: Boost for the first 20% of updates, then Cut, 4 sequences per update.
H8 = dict(
    TRAINING,
    batch=4,                    # PR #40's, in both phases
    mode='synthesis',           # the Cut phase: 'synthesis' (credits at the sites, a probing scheme, block-local
                                # gradients) or 'cut' (one backward of the cut graph)
    boost=True,                 # PR #40's Boost phase first (false: Cut from the first update)
    freeze_queries=False,       # also detach the query's input: record #9's Cut, without own queries
    rows=None,                  # synthesis: Hadamard rows per token per site in the Cut phase (None: all 512)
    row_assign='token',         # 'time' (all tokens share rows, rotating across updates) or 'token' (offset by t mod K)
    prior='none',               # 'identity': read only the residual from the identity path's guess (E7)
    compare_every=0,            # every N updates, the cosine of the block-local gradient to the identity scheme's (E6)
    rank_every=0,               # every N updates, the stable rank of ANVIL II's orthogonalization input (E6)
    # latent objectives (loud/latent.py), cut mode, one per run:
    nextlat=0.0,                # E14: lambda of NextLat's latent-consistency loss (0: off)
    nextlat_target='ema',       # E14: the target latent, 'ema' (H4's teacher) or 'live' (the network's, detached)
    nextlat_onset='handoff',    # E14: from the Boost -> Cut handoff ('handoff') or the first update ('start')
    latent_boost='none',        # E15: Boost's sublayers also ('plus') or only ('only') predict the teacher's latent
    latent_lambda=1.0,          # E15 'plus': the latent loss's weight beside the logit-lens readouts
)

IDENTITY = dict(
    TRAINING,
    rows=None,                  # Hadamard rows per token in the top-stream probe (None: PR #26's full cycle, 512)
    row_offset_per_step=False,  # E6: update s uses rows [(s mod K) * rows, ...), K = 512 / rows
    drift_every=0,              # E6: every N updates, a full-cycle estimate on a held-out sequence vs the momentum
)

CUT = dict(
    TRAINING,
    batch=4,                    # the record's batch: 4 sequences, each on 2 of the 8 virtual GPUs
    means=True,                 # the record's nine trajectory means (the end merge's candidates)
    virtual_gpus=8,             # the record's GPUs, emulated in turn; virtual_gpus / batch of them share a sequence
    end_merge=True,             # the record's end merge on the held-out sequences, timed with the training
    merge_batch=False,          # the merge scores its candidates in one batch, not rank by rank (bf16 rounding only)
    reverse_sum=False,          # the step's 8-way gradient sum in reversed rank order: calibrates its rounding
    rows=None,                  # E6: Hadamard rows per token at each chain site (None: the full cycle, 512)
    row_offset_per_step=False,  # E6: update s reads rows [(s mod K) * rows, ...) of a basis held for K updates
    drift_every=0,              # E6: every N updates (Cut phase), a full-cycle estimate on a held-out row vs momentum
    rows_tiled=False,           # E7's partial cycle at fixed rows: windows tiled across tokens, around the guess
    rows_start=None,            # E7: rows per token at the first Cut update, ramped linearly up to the full cycle...
    rows_full_at=None,          # ...reached at this fraction of the run's updates (tiled, as rows_tiled)
    rows_check=0.0,             # E7's self-check: rows rise when the guess's relative error past them exceeds this
    target_damping=0.0,         # E11: per-token targets, the writers' ridge fit (x mean diag A^T A); 0 = off
    curvature_damping=0.0,      # E8: lambda, in squared ANVIL II steps (loud/curvature.py), Cut phase; 0 = off
    damp_directions=16,         # E8: tracked top singular directions of each block matrix's pending update
    damp_every=50,              # E8: Cut updates between refreshes of the directions and their curvature
)
CUT_SWITCH = 0.2                # PR #40's Boost fraction (Settings.switch): Cut, and its chain, start after it
ROW_QUANTUM = 64                # E7's ramp moves in steps of 64 rows, so only a few chain shapes compile

SCORE = dict(checkpoint=None, splits=['val'])

DIAGNOSE = dict(
    tool=None,                  # cos_to_autograd | credit_dump | token_map_svd | direction_stats | linearity_check |
                                # credit_recovery
    checkpoint=None,            # '<job id>:<file>' (e.g. e1.bp_b4.s0:ckpt_20.pt) or a path relative to the repo
    row=0,                      # the held-out sequence (0-31 of the 10M order's last 32) measured
    batch=8,                    # direction_stats: sequences for the gradient, its curvature and its spread
)
TOOLS = ('cos_to_autograd', 'credit_dump', 'token_map_svd', 'direction_stats', 'linearity_check', 'credit_recovery')


def positive(o, *keys):
    for key in keys:
        if not isinstance(o[key], int) or isinstance(o[key], bool) or o[key] < 1:
            raise ValueError(f'{key} must be a positive integer')


def check_types(o, table):
    """Each option has its default's type: booleans are booleans, counts are integers >= 0, numbers are numbers."""
    for key, default in table.items():
        value = o[key]
        if isinstance(default, bool):
            ok = isinstance(value, bool)
        elif isinstance(default, int):
            ok = isinstance(value, int) and not isinstance(value, bool) and value >= 0
        elif isinstance(default, float):
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        else:
            continue  # strings and None defaults are checked by name below
        if not ok:
            kind = 'true or false' if isinstance(default, bool) else 'an integer >= 0' if isinstance(default, int) \
                else 'a number'
            raise ValueError(f'{key} must be {kind}, not {value!r}')


def training_options(args, table, task):
    unknown = set(args) - set(table)
    if unknown:
        raise ValueError(f'unknown {task} options: {sorted(unknown)}')
    o = dict(table, **args)
    check_types(o, table)
    positive(o, 'batch')
    n = range_size(o['sequences'])
    if not isinstance(o['heldout'], int) or not 0 <= o['heldout'] < n:
        raise ValueError(f'heldout must be an integer in [0, {n})')
    if (n - o['heldout']) // o['batch'] < 1:
        raise ValueError(f'{n - o["heldout"]} training sequences make no update of {o["batch"]}')
    if o['preset'] not in PRESETS:
        raise ValueError(f'unknown preset {o["preset"]!r}; have {sorted(PRESETS)}')
    if o['checkpoint_dtype'] not in CHECKPOINT_DTYPES:
        raise ValueError(f'checkpoint_dtype must be one of {CHECKPOINT_DTYPES}')
    if o['heldout_every'] and not o['heldout']:
        raise ValueError('heldout_every needs held-out sequences (heldout > 0)')
    if o['save_means'] and not o['means']:
        raise ValueError('save_means needs means')
    if o['checkpoint_optimizer'] and not o['checkpoints']:
        raise ValueError('checkpoint_optimizer needs checkpoints')
    return o


def proxy_options(args):
    o = training_options(args, PROXY, 'proxy')
    if o['recipe'] not in ('boostcut', 'pr40', 'pr46'):
        raise ValueError(f'unknown recipe {o["recipe"]!r}')
    if o['lr_mult'] <= 0:
        raise ValueError('lr_mult must be positive')
    if o['checkpoint_optimizer'] and o['recipe'] != 'boostcut':
        raise ValueError("checkpoint_optimizer reads boost-cut's ANVIL II state; PR #40's batched one is not checked")
    positive(o, 'micro_batch')
    if o['batch'] % o['micro_batch']:
        raise ValueError('batch must be a multiple of micro_batch')
    if o['per_sequence_grads'] and o['micro_batch'] != 1:
        raise ValueError('per-sequence gradients need micro_batch: 1')
    if o['grad_noise_every'] and o['batch'] < 2:
        raise ValueError('grad_noise_every needs batch >= 2: the noise is a variance across the sequences')
    if o['grad_noise_every'] and not o['per_sequence_grads']:
        raise ValueError('grad_noise_every needs per_sequence_grads')
    warmdown = o['warmdown']
    if warmdown is not None and (isinstance(warmdown, bool) or not isinstance(warmdown, (int, float))
                                 or not 0 <= warmdown <= 1):
        raise ValueError("warmdown must be a fraction in [0, 1] (0: constant rates), or null for the recipe's")
    if o['step_control'] not in STEP_CONTROLS or o['token_shaping'] not in SHAPINGS:
        raise ValueError(f'step_control must be one of {STEP_CONTROLS}, token_shaping one of {SHAPINGS}')
    variants = [key for key, on in (('smooth_tau', o['smooth_tau'] != 0), ('step_control', o['step_control'] != 'none'),
                                    ('token_shaping', o['token_shaping'] != 'none'),
                                    ('curvature_damping', o['curvature_damping'] != 0),
                                    ('fisher_damping', o['fisher_damping'] != 0)) if on]
    if len(variants) > 1:
        raise ValueError(f'one change per variant: {variants} are separate experiments')
    for key, (mode, value) in VARIANT_OPTIONS.items():
        if key in args and o[mode] != value:
            raise ValueError(f'{key} acts only with {mode}: {value}')
    if o['smooth_tau'] < 0:
        raise ValueError('smooth_tau must be >= 0')
    if not (o['accept_up'] >= 1 and 0 < o['accept_down'] <= 1):
        raise ValueError('accept_up must be >= 1 and accept_down in (0, 1]')
    if o['agree_over'] not in ('active', 'all'):
        raise ValueError("agree_over must be 'active' or 'all'")
    if not 0 < o['cap_quantile'] <= 1:
        raise ValueError('cap_quantile must be in (0, 1]')
    if not 0.5 < o['agree_threshold'] <= 1:
        raise ValueError('agree_threshold must be in (0.5, 1]: at least half the tokens always share the majority sign')
    if o['per_sequence_grads'] and variants and variants[0] not in ('step_control', 'curvature_damping',
                                                                      'fisher_damping'):
        raise ValueError(f"per_sequence_grads reads each sequence's plain gradient, which {variants[0]} replaces")
    for key in ('damp_directions', 'damp_every', 'damp_where'):
        if key in args and not o['curvature_damping']:
            raise ValueError(f'{key} acts only with curvature_damping')
    if o['curvature_damping'] < 0 or o['fisher_damping'] < 0:
        raise ValueError('curvature_damping and fisher_damping must be >= 0 (0 = off)')
    if o['end_merge'] and not (o['recipe'] in ('pr40', 'pr46') and o['means'] and o['heldout']):
        raise ValueError("end_merge is the record's (recipe pr40 or pr46) and needs means and held-out sequences")
    if (o['curvature_damping'] or o['fisher_damping']) and o['recipe'] != 'boostcut':
        raise ValueError("E8 and E9 hook boost-cut's ANVIL II (recipe: boostcut)")
    if o['curvature_damping']:
        positive(o, 'damp_directions', 'damp_every')
        if not o['per_sequence_grads']:
            raise ValueError('curvature_damping reads its noise from per-sequence gradients: set per_sequence_grads')
        if o['batch'] < 2:
            raise ValueError('curvature_damping needs batch >= 2: its noise is a variance across the sequences')
        if o['damp_where'] not in ('step', 'pending'):
            raise ValueError("damp_where must be 'step' or 'pending'")
        if o['damp_directions'] > PRESETS[o['preset']][1]:
            raise ValueError('damp_directions must be at most the width')
    return o


def population_options(args):
    o = training_options(args, POPULATION, 'population')
    changed = sorted(k for k in POPULATION if k not in POPULATION_FREE and o[k] != POPULATION[k])
    if changed:
        raise ValueError(f'population runs keep {changed} at their defaults; they may set {list(POPULATION_FREE)}')
    proxy_options({k: o[k] for k in PROXY if o[k] != PROXY[k]})  # as if set in a proxy config
    if o['rule'] not in ('average', 'select', 'ensemble'):
        raise ValueError("rule must be 'average', 'select' or 'ensemble'")
    if not 0 < o['branch'] < 1:
        raise ValueError('branch must be a fraction in (0, 1)')
    positive(o, 'copies', 'every')
    if o['copies'] < 2:
        raise ValueError('a population needs at least two copies')
    if not o['heldout']:
        raise ValueError('the copies are scored on the held-out sequences: heldout must be > 0')
    if branch_rows(o)[1] // o['copies'] < o['batch']:
        raise ValueError('the sequences left after the branch give a copy no update')
    return o


def branch_rows(o):
    """(single-run updates before the branch, update sequences left after it)."""
    n = range_size(o['sequences']) - o['heldout']
    prefix = round(o['branch'] * (n // o['batch']))
    return prefix, n - prefix * o['batch']


def h8_options(args):
    o = training_options(args, H8, 'h8')
    width = PRESETS[o['preset']][1]
    if o['mode'] not in ('synthesis', 'cut'):
        raise ValueError("mode must be 'synthesis' or 'cut'")
    if o['row_assign'] not in ('time', 'token') or o['prior'] not in ('none', 'identity'):
        raise ValueError("row_assign must be 'time' or 'token', prior 'none' or 'identity'")
    if o['rows'] is not None:
        positive(o, 'rows')
        if o['rows'] & (o['rows'] - 1) or o['rows'] > width:
            raise ValueError(f'rows must be a power of two up to the width ({width})')
    scheme = o['rows'] not in (None, width) or o['prior'] != 'none'
    if o['mode'] == 'cut' and (scheme or o['compare_every']):
        raise ValueError('rows, prior and compare_every are synthesis-mode options')
    for key in ('row_assign',):
        if key in args and o['rows'] in (None, width):
            raise ValueError(f'{key} needs rows below the width')
    if o['nextlat'] < 0 or o['latent_lambda'] < 0:
        raise ValueError('nextlat and latent_lambda must be >= 0')
    if o['nextlat_target'] not in ('ema', 'live') or o['nextlat_onset'] not in ('handoff', 'start'):
        raise ValueError("nextlat_target must be 'ema' or 'live', nextlat_onset 'handoff' or 'start'")
    if o['latent_boost'] not in ('none', 'plus', 'only'):
        raise ValueError("latent_boost must be 'none', 'plus' or 'only'")
    latent = [k for k, on in (('nextlat', o['nextlat'] > 0), ('latent_boost', o['latent_boost'] != 'none')) if on]
    if len(latent) > 1:
        raise ValueError('nextlat (E14) and latent_boost (E15) are separate experiments')
    if latent and o['mode'] != 'cut':
        raise ValueError(f'{latent[0]} runs in cut mode (the plan\'s E14 and E15)')
    if latent and not o['boost'] and (o['latent_boost'] != 'none' or o['nextlat_onset'] == 'handoff'):
        raise ValueError(f'{latent[0]} needs the Boost phase (boost: true)')
    for key, on in (('nextlat_target', o['nextlat'] > 0), ('nextlat_onset', o['nextlat'] > 0),
                    ('latent_lambda', o['latent_boost'] == 'plus')):
        if key in args and not on:
            raise ValueError(f'{key} acts only with ' + ('nextlat' if key != 'latent_lambda' else 'latent_boost: plus'))
    return o


def identity_options(args):
    o = training_options(args, IDENTITY, 'identity')
    if o['rows'] is not None:
        positive(o, 'rows')
        width = PRESETS[o['preset']][1]
        if o['rows'] & (o['rows'] - 1) or o['rows'] > width:
            raise ValueError(f'rows must be a power of two up to the width ({width}): a share of the Hadamard cycle')
        if o['rows'] == width:
            o['rows'] = None
    if o['row_offset_per_step'] and o['rows'] is None:
        raise ValueError('row_offset_per_step needs rows below the full cycle')
    if o['drift_every'] and not o['heldout']:
        raise ValueError('the drift check measures a held-out sequence: it needs heldout > 0')
    return o


def cut_options(args):
    o = training_options(args, CUT, 'cut')
    width = PRESETS[o['preset']][1]
    world, batch = o['virtual_gpus'], o['batch']
    if world not in (1, 2, 4, 8):
        raise ValueError('virtual_gpus must be 1, 2, 4 or 8, as the record supports')
    split = max(1, world // batch)
    if world % split or batch % (world // split):
        raise ValueError(f'batch {batch} does not divide over {world} virtual GPUs as the record does')
    if o['end_merge'] and not (o['means'] and o['heldout']):
        raise ValueError('the end merge needs the means and held-out sequences to score them')
    if o['checkpoint_optimizer']:
        raise ValueError("checkpoint_optimizer reads boost-cut's ANVIL II state; PR #40's batched one is not checked")
    if o['rows'] is not None:
        positive(o, 'rows')
        if o['rows'] & (o['rows'] - 1) or not split <= o['rows'] <= width:
            raise ValueError(f'rows must be a power of two from {split} (the GPUs of a sequence) to {width}')
        if o['rows'] == width:
            o['rows'] = None
    if o['row_offset_per_step'] and o['rows'] is None:
        raise ValueError('row_offset_per_step needs rows below the full cycle')
    if (o['rows_start'] is None) != (o['rows_full_at'] is None):
        raise ValueError('the E7 ramp needs both rows_start and rows_full_at')
    if o['rows_start'] is not None:
        if o['rows'] is not None:
            raise ValueError('give rows (E6) or the rows_start ramp (E7), not both')
        if (not isinstance(o['rows_start'], int) or isinstance(o['rows_start'], bool)
                or o['rows_start'] % ROW_QUANTUM or not 0 < o['rows_start'] < width):
            raise ValueError(f'rows_start must be a multiple of {ROW_QUANTUM} below {width}')
        f = o['rows_full_at']
        if not isinstance(f, (int, float)) or isinstance(f, bool) or not CUT_SWITCH < f <= 1:
            raise ValueError(f'rows_full_at is a fraction of the run after the Boost phase ({CUT_SWITCH}, 1]')
    if o['drift_every'] and not o['heldout']:
        raise ValueError('the drift check measures a held-out sequence: it needs heldout > 0')
    if o['rows_tiled'] and (o['rows'] is None or o['row_offset_per_step']):
        raise ValueError("rows_tiled is E7's partial cycle at a fixed count: rows below the full cycle, no offset")
    for key in ('rows_check', 'target_damping', 'curvature_damping'):
        if o[key] < 0:
            raise ValueError(f'{key} must be >= 0 (0 = off)')
    if o['rows_check'] and not (o['rows_tiled'] or o['rows_start'] is not None):
        raise ValueError("rows_check checks E7's guess: it needs tiled rows (rows_tiled or the rows_start ramp)")
    if o['curvature_damping']:
        positive(o, 'damp_directions', 'damp_every')
        if o['damp_directions'] > width:
            raise ValueError('damp_directions must be at most the width')
        if not 2 <= batch <= world:
            raise ValueError('curvature_damping takes its noise across the batch, one sequence per virtual GPU group: '
                             '2 <= batch <= virtual_gpus')
    return o


def score_options(args):
    unknown = set(args) - set(SCORE)
    if unknown or not args.get('checkpoint'):
        raise ValueError(f'score needs a checkpoint and accepts only {sorted(SCORE)}')
    o = dict(SCORE, **args)
    if not set(o['splits']) <= {'val', 'test'}:
        raise ValueError(f'score splits must be val and/or test, not {o["splits"]}')
    return o


def reference_options(args):
    allowed = {'script', 'cwd', 'argv', 'result', 'settings', 'train_rows', 'script_seed'}
    if set(args) - allowed or not isinstance(args.get('script'), list) or len(args['script']) != 2:
        raise ValueError(f'reference needs script: [repo, path] and accepts only {sorted(allowed)}')
    for arg in args.get('argv', []):
        if set(re.findall(r'{(\w*)}', arg)) - {'data', 'out', 'seed'}:
            raise ValueError(f'reference argv {arg!r}: the placeholders are {{data}}, {{out}} and {{seed}}')
    if not isinstance(args.get('settings', {}), dict):
        raise ValueError("reference settings: a mapping of the script's Settings fields to values")
    if args.get('settings'):
        repo, relpath = args['script']
        script = REPO / json.loads(PINS.read_text())['repos'][repo]['path'] / relpath
        unknown = set(args['settings']) - settings_fields(script)
        if unknown:
            raise ValueError(f'reference settings: {relpath} has no Settings field {sorted(unknown)}')
    rows = args.get('train_rows', 0)
    if not isinstance(rows, int) or isinstance(rows, bool) or not 0 <= rows <= RANGES['30M']:
        raise ValueError(f'reference train_rows: a row count up to {RANGES["30M"]} (0: all of train.pt)')
    if not isinstance(args.get('script_seed', False), bool):
        raise ValueError('reference script_seed: true or false')
    return args


def compare_options(args):
    if set(args) - {'a', 'b', 'first'} or not ({'a', 'b'} <= set(args)):
        raise ValueError('compare needs a and b (job ids or [repo, metrics path]) and accepts first')
    for side in (args['a'], args['b']):
        if not (isinstance(side, str) or (isinstance(side, list) and len(side) == 2)):
            raise ValueError(f'compare side {side!r} must be a job id or [repo, path]')
    return args


def diagnose_options(args):
    unknown = set(args) - set(DIAGNOSE)
    if unknown:
        raise ValueError(f'unknown diagnose options: {sorted(unknown)}')
    o = dict(DIAGNOSE, **args)
    if o['tool'] not in TOOLS:
        raise ValueError(f'tool must be one of {TOOLS}')
    if not isinstance(o['checkpoint'], str) or not o['checkpoint']:
        raise ValueError('checkpoint: give <job id>:<file> or a path')
    check_types(o, DIAGNOSE)
    positive(o, 'batch')
    if o['tool'] == 'direction_stats' and o['batch'] < 2:
        raise ValueError('direction_stats needs batch >= 2 (the noise is a variance across sequences)')
    if o['row'] + (o['batch'] if o['tool'] == 'direction_stats' else 1) > 32:
        raise ValueError('the sequences must be among the 32 held-out ones')
    return o


def gate_options(args):
    values = args.get('values')
    if set(args) != {'values'} or not isinstance(values, dict) or not values or not all(
            isinstance(v, list) and len(v) in (3, 4) and all(isinstance(x, str) for x in v) for v in values.values()):
        raise ValueError('gate needs values: {name: [job id, summary file, field, optional site prefix]}')
    return args


def noop_options(args):
    if set(args) - {'seconds', 'fail', 'val_loss'}:
        raise ValueError('noop accepts only seconds, fail, val_loss')
    return args


VALIDATE = {'proxy': proxy_options, 'identity': identity_options, 'h8': h8_options, 'population': population_options,
            'cut': cut_options, 'score': score_options,
            'reference': reference_options, 'compare': compare_options, 'diagnose': diagnose_options,
            'gate': gate_options, 'noop': noop_options}
