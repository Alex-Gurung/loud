"""Task registry and option tables (no torch import, so submission checks run on the CPU pod)."""
import re

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
PRESETS = {'tiny': (4, 128), '2m': (2, 128), '7m': (4, 256), '38m': (8, 512), '243m': (16, 1024),
           'track3': (12, 1024)}  # track3: track-3 record #5's shape (E16, with arch: ngram)
TRACK3_ROWS = 1048576  # track-3 record #5's n-gram table rows (train/arch/model.py's GPTConfig default)
CHECKPOINT_DTYPES = ('bfloat16', 'float32')

# What each task's result.json holds that `expect` may check.
TRAINING_RESULTS = ('val_loss', 'test_loss', 'heldout_loss', 'train_seconds', 'train_loss_last', 'peak_cuda_gb')
RESULTS = {'proxy': TRAINING_RESULTS, 'identity': TRAINING_RESULTS, 'h8': TRAINING_RESULTS,
           'population': TRAINING_RESULTS,
           'cut': TRAINING_RESULTS + ('val_loss_plain', 'test_loss_plain', 'holdout_plain', 'holdout_merged',
                                      'merge_seconds'),
           'score': ('val_loss', 'test_loss'),
           'reference': ('val_loss', 'test_loss', 'train_seconds'),
           'compare': ('train_loss_max_diff', 'train_loss_mean_diff', 'train_loss_max_diff_all', 'val_loss_diff'),
           'diagnose': ('min_cosine', 'min_cosine_matrices', 'min_cosine_scalars', 'share_099'),  # h8_validation's
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
    checkpoints=False,          # H4 fraction checkpoints (ckpt_00/10/20/50/100.pt): true, false, or the seeds that keep
                                # them (e.g. [0]: only seed 0's, the ones the diagnostics read)
    checkpoint_dtype='bfloat16',
    checkpoint_optimizer=False, # also ANVIL II's state with each checkpoint (about 0.2 GB each), for H6 snr_spectrum
    means=False,                # H4: the record's nine trajectory means from 60%, scored on the held-out set at the end
    save_means=False,           # also write them to means.pt (2-D weights in checkpoint_dtype: about 0.7 GB in bf16)
    heldout_every=0,            # score the held-out set every N updates (monitoring only)
    eval_every=0,               # untimed val.pt curve every N updates: a plot only, never a decision
    test=True,                  # also score test.pt at the end
    official_eval=True,
    final_dtype='bfloat16',     # after the official score of the exact weights, final.pt (and final_plain.pt) keep
                                # their 2-D weights in this dtype (score.json keeps the exact file's sha256); 'float32'
                                # keeps them exact (verifications, E1). The disk is shared
)

JUDGE_ROWS = 4  # E4: held-out rows each update is judged on, rotating through the 32 (v2's; judge_rows sets it)

PROXY = dict(
    TRAINING,
    recipe='boostcut',          # reference trainer: 'boostcut' (backprop_preset.py), 'pr40' (PR #40's rates, the 10M
                                # record) or 'pr46' (PR #46's, the 30M record)
    micro_batch=1,              # sequences per forward/backward (1 = one GPU's share in the 8-GPU script)
    lr_mult=1.0,                # scales both Adam and ANVIL II rates, as backprop_preset.py --lr-mult
    per_sequence_grads=False,   # H2
    grad_noise_every=0,         # log the H2 across-sequence spread every N updates (needs per_sequence_grads)
    warmdown=None,              # the recipe's warmdown fraction (None), or another; 0 holds every rate constant
    # one-change variants, at most one per run:
    blur=0.0,                   # E2: antithetic displacement at the nine sites, per-coordinate RMS = blur x the
                                # token's RMS (loud/variants.py); 0 = off
    step_control='none',        # E4: 'reject' undoes an update that raised the loss on 4 held-out rows; 'adapt' also
                                # scales every rate by the outcomes (loud/variants.py)
    accept_up=1.1,              # E4 adapt: rate multiplier after an accepted update
    accept_down=0.87,           # E4 adapt: after a rejected one (with 1.1: steady at 59% acceptance)
    judge_rows=JUDGE_ROWS,      # E4: held-out rows each update is judged on (v2: 4)
    snr_scaling='none',         # E8: 'control', 'snr' or 'rate' in place of ANVIL II's polar factor (loud/snr.py)
    fisher=False,               # E9: momentum left-preconditioned by the damped true-Fisher factor (loud/fisher.py)
    end_merge=False,            # the record's end merge of the nine means after the last update (recipes pr40, pr46)
    arch='fixed',               # 'ngram' (E16): track-3 record #5's design (loud/arch.py), with recipe pr40
    ngram_rows=TRACK3_ROWS,     # E16: rows of each hashed n-gram table (the record's)
)
STEP_CONTROLS, SNR_ARMS = ('none', 'reject', 'adapt'), ('none', 'control', 'snr', 'rate')
# Each variant option, and the setting (of another option) it acts under.
VARIANT_OPTIONS = {'accept_up': ('step_control', 'adapt'), 'accept_down': ('step_control', 'adapt')}
STEP_OPTIONS = ('judge_rows',)  # E4: act only with step_control on

# E3 (loud/population.py): local SGD over copies branched from one trajectory. Only POPULATION_FREE options may
# change; the rest stay at their defaults (one change per variant).
POPULATION = dict(
    PROXY,
    branch=0.2,                 # fraction of the single run's updates done before the copies branch
    copies=4,                   # K: each update's batch is split evenly over the copies
    rule=None,                  # 'average' | 'select' | 'ensemble'
    every=10,                   # H: updates between averages (average) or between selections (select)
    midpoints_every=10,         # updates between the pairwise-midpoint scores (monitoring)
)
POPULATION_FREE = ('sequences', 'heldout', 'batch', 'preset', 'test', 'official_eval', 'final_dtype', 'recipe',
                   'micro_batch',
                   'lr_mult', 'branch', 'copies', 'rule', 'every', 'midpoints_every', 'checkpoints',
                   'checkpoint_dtype')

# H8 (loud/h8.py) on PR #40's recipe: Boost for the first 20% of updates, then Cut, 4 sequences per update.
H8 = dict(
    TRAINING,
    batch=4,                    # PR #40's, in both phases
    mode='synthesis',           # the Cut phase: 'synthesis' (credits at the sites, a probing scheme, block-local
                                # gradients) or 'cut' (one backward of the cut graph)
    boost=True,                 # PR #40's Boost phase first (false: Cut from the first update)
    lr_mult=1.0,                # scales PR #40's Adam and ANVIL II rates (E19), as the proxy's lr_mult
    cut_lr_mult=1.0,            # scales them in the Cut phase only (E19's Cut-step arms; Boost keeps the record's)
    freeze_queries=False,       # also detach the query's input: record #9's Cut, without own queries
    rows=None,                  # synthesis: Hadamard rows per token per site in the Cut phase (None: all 512)
    row_assign='token',         # 'time' (all tokens share rows, rotating across updates) or 'token' (offset by t mod K)
    prior='none',               # 'identity': read only the residual from the identity path's guess (E7)
    compare_every=0,            # every N updates, the cosine of the block-local gradient to the identity scheme's (E6)
    rank_every=0,               # every N updates, the stable rank of ANVIL II's orthogonalization input (E6)
    arch='fixed',               # 'ngram' (E16): track-3 record #5's design (hashed n-gram value embeddings)
    ngram_rows=TRACK3_ROWS,     # E16: rows of each hashed n-gram table (the record's)
    # latent objectives (loud/latent.py), cut mode, one per run:
    nextlat=0.0,                # E14: lambda of NextLat's latent-consistency loss (0: off)
    nextlat_target='ema',       # E14: the target latent, 'ema' (H4's teacher) or 'live' (the network's, detached)
    nextlat_onset='handoff',    # E14: from the Boost -> Cut handoff ('handoff') or the first update ('start')
    latent_boost='none',        # E15: Boost's sublayers also ('plus') or only ('only') predict the teacher's latent
    latent_lambda=1.0,          # E15 'plus': the latent loss's weight beside the logit-lens readouts
    # synthesis-mode experiments (loud/h8.py), at most one per run, each a fixed rule:
    clip_credit=False,          # E10 (a): each token's credit norm at a site capped at 3x the site's batch median
    agree=False,                # E10 (b): block weight-update entries kept where >= 70% of the tokens agree in sign
    runway=False,               # E11: each token's credit at a site weighted by how far its loss keeps falling
    budget_theta=0.0,           # E13: 64 rows per site where max(n-gram loss, model loss) <= theta nats (0: off)
    clip_factor=3.0,            # E10 (a)'s cap, in the site's batch median token norm (v2: 3)
    agree_share=0.7,            # E10 (b)'s share of tokens that must back an entry's sign (v2: 0.7)
    runway_grid=[],             # E11's moves (in the token's RMS per coordinate); empty: v2's 0.01, 0.02, 0.04, 0.08
    live_paths=[],              # cut mode: cross-token paths left live, 'keys' and/or 'values' (both: the full graph)
    live_blocks=[],             # with live_paths: only these blocks' paths (empty: every block)
    key_draws=0,                # with keys live: each block's key-input credit read through this many random
                                # Hadamard rows, the record's input-key probes (0: exact; the record: 512)
    key_assign='shared',        # with key_draws: 'shared' (every token the same rows, the record's) or 'token'
    snr_scaling='none',         # E8's H8 setting: 'control', 'snr' or 'rate' in place of ANVIL II's polar factor in the
                                # Cut phase, from each sequence's synthesized gradients (loud/snr.py)
)

IDENTITY = dict(TRAINING)

CUT = dict(
    TRAINING,
    batch=4,                    # the record's batch: 4 sequences, each on 2 of the 8 virtual GPUs
    means=True,                 # the record's nine trajectory means (the end merge's candidates)
    virtual_gpus=8,             # the record's GPUs, emulated in turn; virtual_gpus / batch of them share a sequence
    end_merge=True,             # the record's end merge on the held-out sequences, timed with the training
    merge_batch=False,          # the merge scores its candidates in one batch, not rank by rank (bf16 rounding only)
    reverse_sum=False,          # the step's 8-way gradient sum in reversed rank order: calibrates its rounding
    rows=None,                  # Hadamard rows per token at each chain site in the Cut phase (None: the full cycle)
    row_assign='token',         # 'time': all tokens read the same rows, rotating across updates; 'token': token t's
                                # window starts at (t mod K) rows, so every update covers the basis (K = 512 / rows)
    prior='none',               # 'identity' (E7): the mid sites measure the identity-path guess's residual
    row_budget=None,            # E13: a loud.cut.ROW_BUDGETS strategy, each token's own row count (token windows):
                                # 'ngram', LOW_ROWS where the n-gram model and the network agree a token is easy
    budget_theta=0.0,           # E13 'ngram': the low tier is max(n-gram loss, model loss) <= theta nats
    key_blocks=[],              # E17: the blocks whose input-key probes run (empty: all, the record's)
    key_probes=True,            # E17: false runs no input-key probes (key_path_draws 0; c_k keeps its key errors)
    key_transport=False,        # E17: the K-branch input credit = the raw key errors times c_k's weight, no probes
    deep_readout=False,         # E5: one S1 cycle reads S2..S8 by a per-token solve (loud/cut.py)
    profile=False,              # seconds per part of each Cut estimate in metrics.jsonl (cut_profile; GPU-synced)
)
ROW_ASSIGNS, PRIORS = ('time', 'token'), ('none', 'identity')

SCORE = dict(checkpoint=None, splits=['val'])

DIAGNOSE = dict(
    tool=None,                  # one of TOOLS (loud/diagnostics.py)
    checkpoint=None,            # '<job id>:<file>' (e.g. e1.h8_synth.s0:ckpt_20.pt) or a path relative to the repo
    row=0,                      # the held-out sequence (0-31 of the 10M order's last 32) measured
    sequences=64,               # snr_spectrum: the training sequences measured (the run's next, after the checkpoint)
    batch=8,                    # snr_spectrum: the update batch the SNR is stated at (and the checkpoint run's)
    estimators=['backprop', 'synthesis', 'time_k4', 'token_k4'],  # snr_spectrum: per-sequence gradients (ESTIMATORS)
    estimator='identity',       # cos_to_autograd: 'identity' (PR #26), 'h8' (H8 synthesis) or 'cut' (PR #40's Cut)
    precision='bf16',           # h8_validation, deep_readout_check: 'float32' runs without bf16; deep_readout_check
                                # 'bf16_delta': changes carried in bf16 products, as the chain's kernels do
)
TOOLS = ('cos_to_autograd', 'credit_dump', 'deep_readout_check', 'key_paths', 'linearity_check', 'h8_validation',
         'ngram_tiers', 'snr_spectrum', 'synth_vs_real')
# snr_spectrum's per-sequence gradients: autograd through the whole model, or H8 synthesis reading width / K rows
# per token, assigned by time or by token (K = 1: the exact credits; E6's two assignments at E8's K = 4).
ESTIMATORS = {'backprop': None, 'synthesis': (1, 'time'), 'time_k4': (4, 'time'), 'token_k4': (4, 'token')}


def positive(o, *keys):
    for key in keys:
        if not isinstance(o[key], int) or isinstance(o[key], bool) or o[key] < 1:
            raise ValueError(f'{key} must be a positive integer')


def checkpoints_on(o, seed):
    """Whether this seed's run keeps H4's fraction checkpoints (the `checkpoints` option: true, false or seeds)."""
    return o['checkpoints'] is True or (isinstance(o['checkpoints'], list) and seed in o['checkpoints'])


def arch_checks(o, args):
    """E16's arch and ngram_rows (H8 and the proxy)."""
    if o['arch'] not in ('fixed', 'ngram'):
        raise ValueError("arch must be 'fixed' or 'ngram'")
    if 'ngram_rows' in args and o['arch'] != 'ngram':
        raise ValueError('ngram_rows acts only with arch: ngram')
    positive(o, 'ngram_rows')
    if o['arch'] == 'ngram' and o['official_eval'] and (PRESETS[o['preset']] != PRESETS['track3']
                                                        or o['ngram_rows'] != TRACK3_ROWS):
        raise ValueError("the official scorer builds track 3's model from its GPTConfig() defaults (12 x 1024, "
                         f"{TRACK3_ROWS:,} rows per table): preset track3 and the default ngram_rows, or "
                         "official_eval: false")


def check_types(o, table):
    """Each option has its default's type: booleans are booleans, counts are integers >= 0, numbers are numbers;
    `checkpoints` may also be a list of seeds."""
    for key, default in table.items():
        value = o[key]
        if key == 'checkpoints' and isinstance(value, list):
            if not all(isinstance(s, int) and not isinstance(s, bool) for s in value):
                raise ValueError(f'checkpoints must be true, false or a list of seeds, not {value!r}')
            continue
        if isinstance(default, bool):
            ok = isinstance(value, bool)
        elif isinstance(default, int):
            ok = isinstance(value, int) and not isinstance(value, bool) and value >= 0
        elif isinstance(default, float):
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        else:
            continue  # strings and None defaults are checked by name below
        if not ok:
            kind = ('true, false or a list of seeds' if key == 'checkpoints' else 'true or false') \
                if isinstance(default, bool) else 'an integer >= 0' if isinstance(default, int) else 'a number'
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
    if o['final_dtype'] not in CHECKPOINT_DTYPES:
        raise ValueError(f'final_dtype must be one of {CHECKPOINT_DTYPES}')
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
    if o['step_control'] not in STEP_CONTROLS or o['snr_scaling'] not in SNR_ARMS:
        raise ValueError(f'step_control must be one of {STEP_CONTROLS}, snr_scaling one of {SNR_ARMS}')
    variants = [key for key, on in (('blur', o['blur'] != 0), ('step_control', o['step_control'] != 'none'),
                                    ('snr_scaling', o['snr_scaling'] != 'none'), ('fisher', o['fisher'])) if on]
    if len(variants) > 1:
        raise ValueError(f'one change per variant: {variants} are separate experiments')
    for key, (mode, value) in VARIANT_OPTIONS.items():
        if key in args and o[mode] != value:
            raise ValueError(f'{key} acts only with {mode}: {value}')
    for key in STEP_OPTIONS:
        if key in args and o['step_control'] == 'none':
            raise ValueError(f'{key} acts only with step_control')
    if o['blur'] < 0:
        raise ValueError('blur must be >= 0')
    if not (o['accept_up'] >= 1 and 0 < o['accept_down'] <= 1):
        raise ValueError('accept_up must be >= 1 and accept_down in (0, 1]')
    if o['step_control'] != 'none' and not 1 <= o['judge_rows'] <= o['heldout']:
        raise ValueError(f'step_control judges each update on judge_rows ({o["judge_rows"]}) held-out rows: heldout '
                         f'must be at least that')
    if o['per_sequence_grads'] and variants and variants[0] not in ('step_control', 'snr_scaling', 'fisher'):
        raise ValueError(f"per_sequence_grads reads each sequence's plain gradient, which {variants[0]} replaces")
    if o['snr_scaling'] != 'none' and not (o['per_sequence_grads'] and o['batch'] >= 2):
        raise ValueError('snr_scaling reads per-sequence gradients (per_sequence_grads) over a batch of at least 2')
    if o['end_merge'] and not (o['recipe'] in ('pr40', 'pr46') and o['means'] and o['heldout']):
        raise ValueError("end_merge is the record's (recipe pr40 or pr46) and needs means and held-out sequences")
    if (o['snr_scaling'] != 'none' or o['fisher']) and o['recipe'] != 'boostcut':
        raise ValueError("E8 and E9 hook boost-cut's ANVIL II (recipe: boostcut)")
    arch_checks(o, args)
    if o['arch'] == 'ngram' and (o['recipe'] != 'pr40' or variants):
        raise ValueError("arch ngram (E16) trains under PR #40's recipe (recipe: pr40), with no variant")
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
    positive(o, 'copies', 'every', 'midpoints_every')
    if o['copies'] < 2:
        raise ValueError('a population needs at least two copies')
    if o['batch'] % o['copies'] or (o['batch'] // o['copies']) % o['micro_batch']:
        raise ValueError('each copy takes batch / copies sequences of every update: a whole number of micro-batches')
    if not o['heldout']:
        raise ValueError('the copies are scored on the held-out sequences: heldout must be > 0')
    branch, steps = branch_step(o)
    if not 1 <= branch < steps:
        raise ValueError('the branch must come after at least one update and before the last')
    return o


def branch_step(o):
    """(the update after which the copies branch, the single run's updates): 121 of 606 at 10M and batch 8."""
    steps = (range_size(o['sequences']) - o['heldout']) // o['batch']
    return round(o['branch'] * steps), steps


def h8_options(args):
    o = training_options(args, H8, 'h8')
    width = PRESETS[o['preset']][1]
    if o['mode'] not in ('synthesis', 'cut'):
        raise ValueError("mode must be 'synthesis' or 'cut'")
    if not (o['lr_mult'] > 0 and o['cut_lr_mult'] > 0):
        raise ValueError('lr_mult and cut_lr_mult must be positive')
    if o['row_assign'] not in ('time', 'token') or o['prior'] not in ('none', 'identity'):
        raise ValueError("row_assign must be 'time' or 'token', prior 'none' or 'identity'")
    if o['rows'] is not None:
        positive(o, 'rows')
        if o['rows'] & (o['rows'] - 1) or o['rows'] > width:
            raise ValueError(f'rows must be a power of two up to the width ({width})')
    scheme = o['rows'] not in (None, width) or o['prior'] != 'none'
    changes = [key for key, on in (('rows/prior', scheme), ('clip_credit', o['clip_credit']), ('agree', o['agree']),
                                   ('runway', o['runway']), ('budget_theta', o['budget_theta'] != 0)) if on]
    if o['snr_scaling'] not in SNR_ARMS:
        raise ValueError(f'snr_scaling must be one of {SNR_ARMS}')
    if not isinstance(o['live_paths'], list) or set(o['live_paths']) - {'keys', 'values'} or (
            o['live_paths'] and o['mode'] != 'cut'):
        raise ValueError("live_paths: a list of 'keys' and 'values', in cut mode")
    blocks = o['live_blocks']
    if not isinstance(blocks, list) or not all(isinstance(b, int) and 0 <= b < PRESETS[o['preset']][0] for b in blocks) \
            or (blocks and not o['live_paths']):
        raise ValueError('live_blocks: block indices of this preset, with live_paths')
    draws = o['key_draws']
    if type(draws) is not int or not 0 <= draws <= width or (draws and 'keys' not in o['live_paths']):
        raise ValueError(f'key_draws: 0 (exact) or 1..{width} rows, with the keys live')
    if o['key_assign'] not in ('shared', 'token') or (o['key_assign'] == 'token' and (not draws or width % draws)):
        raise ValueError("key_assign: 'shared' or 'token' (token: key_draws dividing the width)")
    if not (o['clip_factor'] > 0 and 0 <= o['agree_share'] <= 1):
        raise ValueError('clip_factor must be > 0 and agree_share in [0, 1]')
    grid = o['runway_grid']
    if not isinstance(grid, list) or not all(isinstance(d, (int, float)) and d > 0 for d in grid):
        raise ValueError('runway_grid: a list of positive moves (empty: v2\'s)')
    for key, on in (('clip_factor', o['clip_credit']), ('agree_share', o['agree']), ('runway_grid', o['runway'])):
        if key in args and not on:
            raise ValueError(f'{key} acts only with its rule on')
    if o['snr_scaling'] != 'none':  # E8: its arms run on a partial-row scheme (E6's best at K = 4), the setting
        if o['mode'] != 'synthesis' or o['batch'] < 2 or o['prior'] != 'none':
            raise ValueError('snr_scaling (E8) runs in synthesis mode at a batch of at least 2, without a prior')
        changes = [c for c in changes if c != 'rows/prior'] + ['snr_scaling']
    if o['mode'] == 'cut' and (changes or o['compare_every']):
        raise ValueError('rows, prior, clip_credit, agree, runway, budget_theta, snr_scaling and compare_every are '
                         'synthesis-mode options')
    if len(changes) > 1:
        raise ValueError(f'one change per arm: {changes} are separate experiments')
    if o['budget_theta'] < 0:
        raise ValueError('budget_theta must be >= 0 nats (0: off)')
    if ('row_assign' in args or o['prior'] != 'none') and o['rows'] in (None, width):
        raise ValueError('row_assign and prior need rows below the width (with every row they change nothing)')
    if o['checkpoint_optimizer']:
        raise ValueError("checkpoint_optimizer reads boost-cut's ANVIL II state; PR #40's batched one is not checked")
    if o['nextlat'] < 0 or o['latent_lambda'] < 0:
        raise ValueError('nextlat and latent_lambda must be >= 0')
    if o['nextlat_target'] not in ('ema', 'live') or o['nextlat_onset'] not in ('handoff', 'start'):
        raise ValueError("nextlat_target must be 'ema' or 'live', nextlat_onset 'handoff' or 'start'")
    if o['latent_boost'] not in ('none', 'plus', 'only'):
        raise ValueError("latent_boost must be 'none', 'plus' or 'only'")
    arch_checks(o, args)
    if o['arch'] == 'ngram' and o['mode'] != 'cut':
        raise ValueError("arch ngram runs in cut mode (E16): its n-gram tables feed every block, so synthesis's "
                         "block-local gradients are not defined for them")
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
    return training_options(args, IDENTITY, 'identity')


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
        if o['rows'] == width and o['row_budget'] is None:  # a budget's high tier reads every row of its window
            o['rows'] = None
    if o['row_assign'] not in ROW_ASSIGNS or o['prior'] not in PRIORS:
        raise ValueError(f'row_assign must be one of {ROW_ASSIGNS}, prior one of {PRIORS}')
    if o['prior'] != 'none' and o['rows'] is None:
        raise ValueError('prior acts on a partial cycle: it needs rows below the full cycle')
    if o['row_budget'] is not None:
        if not isinstance(o['row_budget'], str):
            raise ValueError('row_budget names a loud.cut.ROW_BUDGETS strategy')
        if o['rows'] is None or o['row_assign'] != 'token':
            raise ValueError('row_budget gives each token its own count up to rows: it needs rows and token windows')
    if (o['row_budget'] == 'ngram') != (o['budget_theta'] > 0) or o['budget_theta'] < 0:
        raise ValueError("row_budget 'ngram' needs budget_theta > 0 nats, and budget_theta acts only with it")
    blocks = o['key_blocks']
    if not isinstance(blocks, list) or not all(isinstance(b, int) and 0 <= b < PRESETS[o['preset']][0]
                                               for b in blocks) or len(set(blocks)) < len(blocks):
        raise ValueError('key_blocks: distinct block indices of this preset (empty: every block)')
    if o['deep_readout'] and (o['rows'] is not None or o['row_budget'] is not None):
        raise ValueError('deep_readout solves full cycles: not with rows or a row budget')
    if o['key_transport'] and (blocks or not o['key_probes']):
        raise ValueError('key_transport replaces every block\'s key probes: not with key_blocks or key_probes false')
    if blocks and not o['key_probes']:
        raise ValueError('key_blocks chooses blocks for the key probes: not with key_probes false')
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
    allowed = {'script', 'cwd', 'argv', 'result'}
    if set(args) - allowed or not isinstance(args.get('script'), list) or len(args['script']) != 2:
        raise ValueError(f'reference needs script: [repo, path] and accepts only {sorted(allowed)}')
    for arg in args.get('argv', []):
        if set(re.findall(r'{(\w*)}', arg)) - {'data', 'out', 'seed'}:
            raise ValueError(f'reference argv {arg!r}: the placeholders are {{data}}, {{out}} and {{seed}}')
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
    if o['row'] >= 32:
        raise ValueError('row must be one of the 32 held-out sequences (0-31)')
    snr = [k for k in ('sequences', 'batch', 'estimators') if o[k] != DIAGNOSE[k]]
    if snr and o['tool'] != 'snr_spectrum':
        raise ValueError(f'{snr} apply to snr_spectrum only')
    if o['estimator'] not in ('identity', 'h8', 'cut') or (
            o['estimator'] != 'identity' and o['tool'] != 'cos_to_autograd'):
        raise ValueError("estimator ('identity', 'h8' or 'cut') applies to cos_to_autograd only")
    if o['precision'] not in ('bf16', 'float32', 'bf16_delta') or (
            o['precision'] != 'bf16' and o['tool'] not in ('h8_validation', 'deep_readout_check')) or (
            o['precision'] == 'bf16_delta' and o['tool'] != 'deep_readout_check'):
        raise ValueError("precision: 'bf16' or 'float32' (h8_validation, deep_readout_check), or 'bf16_delta' "
                         "(deep_readout_check: the chain's way of differencing)")
    positive(o, 'sequences', 'batch')
    names = o['estimators']
    if not names or set(names) - set(ESTIMATORS) or len(set(names)) < len(names):
        raise ValueError(f'estimators: distinct names from {sorted(ESTIMATORS)}')
    if o['sequences'] < 2 * o['batch']:
        raise ValueError('sequences must be at least twice batch (the SNR needs the variance over sequences)')
    return o


def gate_options(args):
    values = args.get('values')
    if set(args) != {'values'} or not isinstance(values, dict) or not values or not all(
            isinstance(v, list) and len(v) in (3, 4) and all(isinstance(x, str) for x in v) for v in values.values()):
        raise ValueError('gate needs values: {name: [job id, summary file, field, optional site pattern]}')
    return args


def noop_options(args):
    if set(args) - {'seconds', 'fail', 'val_loss'}:
        raise ValueError('noop accepts only seconds, fail, val_loss')
    return args


def jobs_read(task, args):
    """[(job id, the dependency lists that may name it)] for each job whose files the task's args read, so it must be
    done first: a diagnostic's checkpoint run must pass its check (`after`), a comparison runs whether or not its runs
    passed theirs (`after_done`: it explains a failed check), and a gate reads finished diagnostics either way."""
    if task == 'diagnose' and ':' in args['checkpoint']:
        return [(args['checkpoint'].split(':', 1)[0], ('after',))]
    if task == 'compare':
        return [(side, ('after_done',)) for side in (args['a'], args['b']) if isinstance(side, str)]
    if task == 'gate':
        return [(value[0], ('after', 'after_done')) for value in args['values'].values()]
    return []


VALIDATE = {'proxy': proxy_options, 'identity': identity_options, 'h8': h8_options, 'population': population_options,
            'cut': cut_options, 'score': score_options,
            'reference': reference_options, 'compare': compare_options, 'diagnose': diagnose_options,
            'gate': gate_options, 'noop': noop_options}
