"""Task 'reference': run a pinned external script unmodified, on this one GPU, and record its own log as the run's.

Used to check the ports against their sources: e.g. backprop_preset.py run as published (one process, so all of a
batch's sequences go through one forward) against loud.proxy with micro_batch equal to the batch, which must give
the same per-step losses, and loud.proxy's default one-sequence micro-batches, which reproduce the 8-GPU layout.

    args:
      script: [boost-cut, noise/analysis/backprop_preset.py]   # must be pinned in external.json
      cwd: noise                                               # relative to that checkout (default: its root)
      argv: [--data-dir, '{data}', --output, '{out}/ref', --batch, '8', --test-file, test.pt]   # also {seed}
      result: ref/result.json                                  # the script's result, relative to the run directory
"""
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
    run = Run(run_dir, spec, dict(command=[sys.executable, str(script), *argv], cwd=str(cwd)))
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)  # the script resolves its own checkout's modules
    start = time.time()
    with (run_dir / 'reference.log').open('w') as log:
        code = subprocess.run([sys.executable, '-u', str(script), *argv], cwd=cwd, env=env, stdout=log,
                              stderr=subprocess.STDOUT).returncode
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
