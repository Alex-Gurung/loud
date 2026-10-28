"""The benchmark's data, hash-checked, in the records' order, with the H3 held-out sequences split off.

Training range: the first N rows of data/train.pt (1M = 488, 10M = 4,880, the archived 10M set; 30M = 14,648).
Order: every record permutes its range with torch.randperm(N, Generator().manual_seed(SEED + 1)), SEED = 0.
Held-out (H3): the last 32 sequences of that order feed no update (as the record's end merge); with heldout: 0
the run uses all N, as backprop_preset.py does.
"""
from dataclasses import dataclass
import hashlib

import torch

from .options import RANGES, range_size
from .paths import DATA
from .refs import pins

SEQ = 2049
ORDER_SEED = 1
HOLDOUT = 32


def token_hash(rows):
    """utils/prepare_data.token_hash: sha256 of the int32 token bytes."""
    return hashlib.sha256(rows.to(torch.int32).contiguous().numpy().tobytes()).hexdigest()


def load_split(name):
    """A pinned split (train.pt, val.pt or test.pt) as int64 [rows, 2049]."""
    expected = pins()['data'][name]
    data = torch.load(DATA / name, map_location='cpu', weights_only=True)
    if tuple(data.shape) != (expected['rows'], SEQ):
        raise ValueError(f'{name}: shape {tuple(data.shape)}, expected ({expected["rows"]}, {SEQ})')
    if token_hash(data) != expected['token_sha256']:
        raise ValueError(f'{name}: token hash mismatch; rebuild it (scripts/prepare_data.sh)')
    return data.long()


@dataclass
class Split:
    train: torch.Tensor      # [n - heldout, 2049], in update order
    heldout: torch.Tensor    # [heldout, 2049]
    order: torch.Tensor      # row indices into train.pt, update order then held-out
    sequences: int

    def describe(self):
        return dict(range_rows=self.sequences, update_sequences=len(self.train), heldout_sequences=len(self.heldout),
                    order_seed=ORDER_SEED, first_rows=self.order[:8].tolist(),
                    heldout_rows=self.order[len(self.train):].tolist())


def benchmark_split(sequences='10M', heldout=HOLDOUT):
    n = range_size(sequences)
    rows = load_split('train.pt')[:n]
    if n == RANGES['10M'] and token_hash(rows) != pins()['data']['train.pt[:4880]']['token_sha256']:
        raise ValueError('the first 4,880 rows are not the archived 10M set')
    order = torch.randperm(n, generator=torch.Generator().manual_seed(ORDER_SEED))
    ordered = rows[order]
    if not 0 <= heldout < n:
        raise ValueError('held-out count must be in [0, n)')
    cut = n - heldout
    return Split(train=ordered[:cut], heldout=ordered[cut:], order=order, sequences=n)
