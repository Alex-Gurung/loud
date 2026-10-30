"""Keep the shared volume small: delete the weights of finished runs once they are scored, keeping only what later
work reads. Every deletion is appended to artifacts/deleted_weights.txt; scores, logs and diagnostics stay.

    python3 -m loud.prune              # what it would delete
    python3 -m loud.prune --delete     # delete it

Kept: E1's runs (the baselines every comparison and diagnostic reads), the verifications that score their weights
(verify.bp_b8*, verify.identity), and the H4 checkpoints a config reads (`checkpoint: <job>:<file>` in any config,
with the .optim.pt beside it). A run is pruned only when its result.json says complete (or it failed its check), so a
running or crashed attempt is never touched; attempt and superseded directories are pruned the same way.
"""
import argparse
from pathlib import Path
import re
import time

from .common import read_json
from .paths import ARTIFACTS, REPO, RUNS

KEEP_EXPERIMENTS = {'e1'}
KEEP_RUNS = {('verify', 'bp_b8'), ('verify', 'bp_b8_onefwd'), ('verify', 'identity'), ('verify', 'score_record')}
MANIFEST = ARTIFACTS / 'deleted_weights.txt'


def checkpoints_read(configs=REPO / 'configs'):
    """(experiment, variant, seed, file) for every `checkpoint: <job id>:<file>` in a config, followups included."""
    found = set()
    for path in Path(configs).rglob('*.yaml'):
        for jid, name in re.findall(r'checkpoint:\s*([\w.-]+\.s\d+):([\w.]+)', path.read_text()):
            experiment, variant, seed = jid.rsplit('.', 2)
            found.add((experiment, variant, seed, name))
            found.add((experiment, variant, seed, name.replace('.pt', '.optim.pt')))
    return found


def candidates(runs=RUNS):
    keep = checkpoints_read()
    for run in sorted(Path(runs).glob('*/*/s*')):
        experiment, variant = run.parent.parent.name, run.parent.name
        if experiment in KEEP_EXPERIMENTS or (experiment, variant) in KEEP_RUNS:
            continue
        result = run / 'result.json'
        if not result.exists():
            continue  # running, crashed or never finished: leave it alone
        info = read_json(result)
        if not (info.get('complete') or (info.get('check') or {}).get('status') == 'fail'):
            continue
        seed = run.name.split('.')[0]
        for f in sorted(run.glob('*.pt')):
            weights = f.name.startswith(('final', 'ckpt_', 'means'))  # diagnostic tensors (<tool>.pt) stay
            if weights and (experiment, variant, seed, f.name) not in keep:
                yield f


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--delete', action='store_true')
    args = parser.parse_args()
    files = list(candidates())
    total = sum(f.stat().st_size for f in files)
    if args.delete and files:
        with MANIFEST.open('a') as log:
            log.write(f'# {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} loud.prune: finished runs\' weights\n')
            for f in files:
                log.write(f'{f.stat().st_size} {f.relative_to(REPO)}\n')
                f.unlink()
    print(f'{"deleted" if args.delete else "would delete"} {len(files)} files, {total / 1e9:.2f} GB')


if __name__ == '__main__':
    main()
