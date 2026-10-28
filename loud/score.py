"""The official score of a checkpoint, in a fresh process: utils/eval.py's evaluate() on val.pt (the benchmark number),
and the same computation on test.pt (boost-cut's held-out test split; analysis only).

    python -m loud.score final.pt --splits val,test

Runs apart from the trainer so the score loads the weights into a fresh copy of the fixed model (strict state-dict
match) with no training code imported.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from .common import file_digest, write_json
from .paths import DATA, REPO
from . import refs


def official_eval_module():
    import importlib.util
    path = refs.verified('noise', 'utils/eval.py')
    spec = importlib.util.spec_from_file_location('noise_eval', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # puts external/noise first on sys.path, imports its utils.model
    if file_digest(sys.modules['utils.model'].__file__) != refs.pins()['fixed_model_sha256']:
        raise RuntimeError('utils/eval.py did not load the fixed model')
    return module


def score(checkpoint, splits=('val',)):
    ev = official_eval_module()
    torch.cuda.set_device(0)
    state_dict = torch.load(checkpoint, map_location='cpu', weights_only=True)
    out = dict(checkpoint=Path(checkpoint).name, checkpoint_sha256=file_digest(checkpoint), torch=torch.__version__)
    for split in splits:
        if split == 'val':
            out['val_loss'] = ev.evaluate(state_dict, DATA)  # the benchmark's own function, unchanged
        elif split == 'test':  # evaluate()'s computation, model loading and batch size, on test.pt
            from .data import load_split
            from .heldout import mean_loss
            device = torch.device('cuda', 0)
            out['test_loss'] = mean_loss(ev.load_model(state_dict, device), load_split('test.pt'), device, 16)
        else:
            raise ValueError(f'unknown split {split!r}')
    return out


def score_in_subprocess(checkpoint, splits, out_path):
    """Score in a child process (the trainer has another checkout's utils/ imported) and return its numbers."""
    env = dict(os.environ)
    code = env.get('LOUD_CODE_DIR') or str(Path(__file__).resolve().parents[1])
    env['PYTHONPATH'] = code
    cmd = [sys.executable, '-u', '-m', 'loud.score', str(checkpoint), '--splits', ','.join(splits), '--out',
           str(out_path)]
    proc = subprocess.run(cmd, cwd=code, env=env, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f'scoring failed ({proc.returncode}):\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}')
    result = json.loads(Path(out_path).read_text())
    return {k: result[k] for k in ('val_loss', 'test_loss') if k in result}


def run_task(spec, run_dir):
    """Task 'score': rescore a checkpoint (verification 1)."""
    from .runlog import Run
    args = spec.get('args', {})
    checkpoint = Path(args['checkpoint'])
    if not checkpoint.is_absolute():
        checkpoint = REPO / checkpoint
    run = Run(run_dir, spec)
    result = score(checkpoint, tuple(args.get('splits', ['val'])))
    write_json(run_dir / 'score.json', result)
    return run.finish(result, spec.get('expect'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--splits', default='val')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = score(args.checkpoint, tuple(args.splits.split(',')))
    if args.out:
        write_json(args.out, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
