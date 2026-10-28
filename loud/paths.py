"""Where everything lives on the shared volume.

Code runs from frozen snapshots under artifacts/code/, so the shared state (data, external checkouts, artifacts) comes
from LOUD_REPO (default /mnt/disk/loud), not from this file's location. SOURCE is the checkout (or snapshot) this code
itself sits in: what a new snapshot freezes.
"""
import os
from pathlib import Path

REPO = Path(os.environ.get('LOUD_REPO', '/mnt/disk/loud'))
DATA = Path(os.environ.get('LOUD_DATA', REPO / 'data'))
ARTIFACTS = Path(os.environ.get('LOUD_ARTIFACTS', REPO / 'artifacts'))
QUEUE = Path(os.environ.get('LOUD_QUEUE_DIR', ARTIFACTS / 'queue'))
RUNS = ARTIFACTS / 'runs'
CODE = ARTIFACTS / 'code'
RESULTS = ARTIFACTS / 'results'
SOURCE = Path(__file__).resolve().parents[1]
# The pins travel with the code: a snapshot reads its own copy, so editing external.json never changes queued jobs.
PINS = SOURCE / 'external.json'


def run_dir(experiment, variant, seed):
    return RUNS / experiment / variant / f's{seed}'
