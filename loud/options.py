"""Task registry and option tables (no torch import, so submission checks run on the CPU pod)."""
import re

TASKS = {
    'proxy': ('loud.proxy', 'run'),               # H1 backprop proxy
    'identity': ('loud.identity', 'run'),         # H5 identity-path forward-only estimator (PR #26)
    'score': ('loud.score', 'run_task'),          # official rescoring of a checkpoint
    'reference': ('loud.reference', 'run_task'),  # a pinned external script, unmodified, on one GPU
    'compare': ('loud.compare', 'run_task'),      # two runs' per-update losses side by side
    'diagnose': ('loud.diagnostics', 'run_task'), # H6: one diagnostic tool on one checkpoint
    'noop': ('loud.noop', 'run_task'),            # queue/worker plumbing check, no GPU work
}

# Training ranges (the first N rows of train.pt) and model presets (n_layer, width: the Dust paper's ladder plus PR
# #26's 'tiny'; only '38m' is the benchmark model, the others are for smoke tests).
RANGES = {'1M': 488, '10M': 4880, '30M': 14648}
PRESETS = {'tiny': (4, 128), '2m': (2, 128), '7m': (4, 256), '38m': (8, 512), '243m': (16, 1024)}
CHECKPOINT_DTYPES = ('bfloat16', 'float32')

# What each task's result.json holds that `expect` may check.
TRAINING_RESULTS = ('val_loss', 'test_loss', 'heldout_loss', 'train_seconds', 'train_loss_last', 'peak_cuda_gb')
RESULTS = {'proxy': TRAINING_RESULTS, 'identity': TRAINING_RESULTS, 'score': ('val_loss', 'test_loss'),
           'reference': ('val_loss', 'test_loss', 'train_seconds'),
           'compare': ('train_loss_max_diff', 'train_loss_mean_diff', 'train_loss_max_diff_all', 'val_loss_diff'),
           'diagnose': (),
           'noop': ('val_loss',)}


def range_size(sequences):
    n = RANGES.get(sequences) if isinstance(sequences, str) else sequences
    if not isinstance(n, int) or not 0 < n <= RANGES['30M']:
        raise ValueError(f'sequences {sequences!r}: give one of {sorted(RANGES)} or a row count up to {RANGES["30M"]}')
    return n


def check_expect(task, expect):
    """expect: {metric: [target, tolerance]} over the metrics the task's result.json holds."""
    if expect is None:
        return
    if not isinstance(expect, dict):
        raise ValueError('expect must map metrics to [target, tolerance]')
    for metric, bound in expect.items():
        if metric not in RESULTS[task]:
            raise ValueError(f'expect: {task} results have no {metric!r}; checkable: {list(RESULTS[task])}')
        if (not isinstance(bound, list) or len(bound) != 2
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bound) or bound[1] < 0):
            raise ValueError(f'expect: {metric} needs [target, tolerance >= 0], not {bound!r}')


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
    recipe='boostcut',          # reference trainer: 'boostcut' (backprop_preset.py) or 'pr40' (PR #40's rates)
    micro_batch=1,              # sequences per forward/backward (1 = one GPU's share in the 8-GPU script)
    lr_mult=1.0,                # scales both Adam and ANVIL II rates, as backprop_preset.py --lr-mult
    per_sequence_grads=False,   # H2
    grad_noise_every=0,         # log the H2 across-sequence spread every N updates (needs per_sequence_grads)
)

IDENTITY = dict(
    TRAINING,
    rows=None,                  # Hadamard rows per token in the top-stream probe (None: PR #26's full cycle, 512)
    row_offset_per_step=False,  # E6: update s uses rows [(s mod K) * rows, ...), K = 512 / rows
    drift_every=0,              # E6: every N updates, a full-cycle estimate on a held-out sequence vs the momentum
)

SCORE = dict(checkpoint=None, splits=['val'])

DIAGNOSE = dict(
    tool=None,                  # cos_to_autograd | credit_dump | token_map_svd | direction_stats | linearity_check
    checkpoint=None,            # '<job id>:<file>' (e.g. e1.bp_b4.s0:ckpt_20.pt) or a path relative to the repo
    row=0,                      # the held-out sequence (0-31 of the 10M order's last 32) measured
    batch=8,                    # direction_stats: sequences for the gradient, its curvature and its spread
)
TOOLS = ('cos_to_autograd', 'credit_dump', 'token_map_svd', 'direction_stats', 'linearity_check')


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
    if o['recipe'] not in ('boostcut', 'pr40'):
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
    if o['grad_noise_every'] and not o['per_sequence_grads']:
        raise ValueError('grad_noise_every needs per_sequence_grads')
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
    positive(o, 'batch')
    if o['tool'] == 'direction_stats' and o['batch'] < 2:
        raise ValueError('direction_stats needs batch >= 2 (the noise is a variance across sequences)')
    if o['row'] + (o['batch'] if o['tool'] == 'direction_stats' else 1) > 32:
        raise ValueError('the sequences must be among the 32 held-out ones')
    return o


def noop_options(args):
    if set(args) - {'seconds', 'fail', 'val_loss'}:
        raise ValueError('noop accepts only seconds, fail, val_loss')
    return args


VALIDATE = {'proxy': proxy_options, 'identity': identity_options, 'score': score_options,
            'reference': reference_options, 'compare': compare_options, 'diagnose': diagnose_options,
            'noop': noop_options}
