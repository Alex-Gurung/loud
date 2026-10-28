"""Task 'reference': run a pinned external script unmodified and record its own log as the run's.

Used to check the ports against their sources: e.g. backprop_preset.py run as published (one process, so all of a
batch's sequences go through one forward) against loud.proxy with micro_batch equal to the batch, which must give
the same per-step losses, and loud.proxy's default one-sequence micro-batches, which reproduce the 8-GPU layout.
With `gpus: N` in the config the script runs under torchrun on the worker's N GPUs (E12's 8-GPU record runs).

    args:
      script: [boost-cut, noise/analysis/backprop_preset.py]   # must be pinned in external.json
      cwd: noise                                               # relative to that checkout (default: its root)
      argv: [--data-dir, '{data}', --output, '{out}/ref', --batch, '8', --test-file, test.pt]   # also {seed}
      result: ref/result.json                                  # the script's result, relative to the run directory
      settings: {lr: 0.0018}    # optional: fields of the script's Settings replaced (loud/launch.py), file unchanged
      train_rows: 8788          # optional: the epoch is the first this many rows of train.pt (loud/launch.py)
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch  # noqa: F401  (the script runs on this GPU; importing torch records it in config.json)

from .common import read_json, write_json
from .paths import DATA, REPO
from . import refs
from .runlog import Run


def run_task(spec, run_dir):
    args = spec['args']
    repo, relpath = args['script']
    script = refs.verified(repo, relpath)
    checkout = REPO / refs.pins()['repos'][repo]['path']
    cwd = checkout / args.get('cwd', '.')
    argv = [a.format(data=DATA, out=run_dir, seed=spec['seed']) for a in args.get('argv', [])]
    command = [str(script), *argv]
    if args.get('settings') or args.get('train_rows'):
        command = [str(Path(__file__).with_name('launch.py')), str(script), json.dumps(args.get('settings') or {}),
                   str(args.get('train_rows') or 0), *argv]
    gpus = spec.get('gpus', 1)
    launcher = ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={gpus}'] if gpus > 1 else []
    command = [sys.executable, '-u', *launcher, *command]
    run = Run(run_dir, spec, dict(command=command, cwd=str(cwd)))
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)  # the script resolves its own checkout's modules
    start = time.time()
    with (run_dir / 'reference.log').open('w') as log:
        code = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        raise RuntimeError(f'{relpath} exited {code}; see {run_dir / "reference.log"}')
    result = read_json(run_dir / args.get('result', 'ref/result.json'))
    metrics = run_dir / Path(args.get('result', 'ref/result.json')).parent / 'metrics.jsonl'
    if metrics.exists():  # the script's per-step log is the run's metrics
        run.metrics.handle.write(metrics.read_text())
        run.metrics.handle.flush()
    write_json(run_dir / 'script_result.json', result)
    result = dict(result, script_seconds=round(time.time() - start, 1), complete=True)
    return run.finish(result, spec.get('expect'))
