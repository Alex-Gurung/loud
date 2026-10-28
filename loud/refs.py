"""Pinned external code: the Noise benchmark, boost-cut and the record scripts, imported only after their hashes match
external.json.

    python -m loud.refs --check      # verify every pinned file, download and data split (CPU only)
"""
import argparse
import importlib.util
import json
import sys

from .common import file_digest, read_json
from .paths import DATA, PINS, REPO


def pins():
    return read_json(PINS)


def verified(repo, relpath):
    """Path of a pinned file inside an external checkout, after checking its sha256."""
    entry = pins()['repos'][repo]
    if relpath not in entry['files']:
        raise KeyError(f'{repo}:{relpath} is not pinned in external.json')
    path = REPO / entry['path'] / relpath
    actual = file_digest(path)
    if actual != entry['files'][relpath]:
        raise RuntimeError(f'{path} changed (sha256 {actual}); expected the pinned {entry["files"][relpath]} '
                           f'at {repo}@{entry["commit"][:7]}')
    return path


def download(name):
    entry = pins()['downloads'][name]
    path = REPO / entry['path']
    if file_digest(path) != entry['sha256']:
        raise RuntimeError(f'{path} does not match its pinned sha256')
    return path


# Reference trainers whose model construction, init, optimizer (and, for pr26, estimator) the ports reuse verbatim.
TRAINERS = {
    # backprop_preset.py @539386e imports this module: the record's recipe as of boost-cut (lr 0.003, betas (0.7, 0.95),
    # warmdown 0.3 to zero, ANVIL II 0.023, per-matrix polar). Backprop with it: 3.7081 (seed 0), PR #20.
    'boostcut': ('boost-cut', 'noise/train/boostcut/run.py'),
    # PR #40 (10M record #14): lr 0.0021, betas (0.5, 0.98), warmdown 0.4 to 30%, head std 0.02, ANVIL II 0.026,
    # fast weight 1.0, batched polar (Gram Newton-Schulz on wide matrices).
    'pr40': ('noise', 'archive/10M/train/val/run.py'),
    # PR #26 (track 2, 4.1772): the identity-path forward-only estimator that loud.identity runs unchanged.
    'pr26': ('noise-pr26', 'train/speedrun/run.py'),
}
_loaded = {}


def trainer_module(key):
    """Import one reference trainer. Each one puts its own checkout's utils/ first on sys.path, so a process may load
    only one of them; the utils.model it imports must be the fixed model."""
    if key in _loaded:
        return _loaded[key]
    if _loaded:
        raise RuntimeError(f'{next(iter(_loaded))} is already loaded; one reference trainer per process')
    if 'utils' in sys.modules:
        raise RuntimeError('a utils package is already imported; load the reference trainer first')
    path = verified(*TRAINERS[key])
    spec = importlib.util.spec_from_file_location(f'ref_{key}', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model_file = sys.modules['utils.model'].__file__
    if file_digest(model_file) != pins()['fixed_model_sha256']:
        raise RuntimeError(f'{key} imported {model_file}, which is not the fixed model')
    _loaded[key] = module
    return module


def check_all(data=True):
    report = {}
    p = pins()
    for repo, entry in p['repos'].items():
        for relpath in entry['files']:
            verified(repo, relpath)
        report[repo] = f'{len(entry["files"])} files match {entry["commit"][:7]}'
    for name in p['downloads']:
        download(name)
        report[name] = 'sha256 OK'
    if data:
        from . import data as d
        for name in ('train.pt', 'val.pt', 'test.pt'):
            if not (DATA / name).exists():
                raise FileNotFoundError(f'{DATA / name} is missing; build it with scripts/prepare_data.sh')
            d.load_split(name)
            report[name] = 'token hash OK'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--no-data', action='store_true')
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check_all(data=not args.no_data), indent=2))


if __name__ == '__main__':
    main()
