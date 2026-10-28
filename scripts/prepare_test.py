"""Held-out test split (analysis only; never trains, never feeds a decision).

The next 544 rows of the benchmark's FineWeb stream after the validation pool, as in boost-cut's
noise/analysis/prepare_test.py @539386e, rebuilt against the current 30M prepare_data constants. The regenerated
validation split must match its pinned hash before anything is written; test.pt must match the hash published in
PR #20 and boost-cut's ledger.

    python scripts/prepare_test.py --noise external/noise --data-dir data
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

TEST_SHA256 = 'cb7a29f079ae2de5e96dd472497b67a6ec3c827bf6efbd8c263c7ab1001f6d7d'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.noise.resolve()))
    from utils.prepare_data import (EXPECTED, POOL_TOKENS, SEQ, SKIP_DOCS, TOKENIZER, TOKENIZER_SHA256, VAL_POOL_TOKENS,
                                    VAL_ROWS, encode_split, token_hash)
    from datasets import load_dataset
    from tokenizers import Tokenizer
    if hashlib.sha256(TOKENIZER.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise SystemExit('not the fixed tokenizer')
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    documents = iter(load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=True))
    for _ in range(SKIP_DOCS):
        next(documents)
    encode_split(documents, tokenizer, POOL_TOKENS)
    val = encode_split(documents, tokenizer, VAL_POOL_TOKENS)[:VAL_ROWS]
    if token_hash(val) != EXPECTED['val.pt']:
        raise SystemExit('regenerated val.pt does not match the pinned split')
    test = encode_split(documents, tokenizer, VAL_POOL_TOKENS)[:VAL_ROWS].clone()
    assert tuple(test.shape) == (VAL_ROWS, SEQ)
    digest = token_hash(test)
    if digest != TEST_SHA256:
        raise SystemExit(f'test.pt hash {digest} does not match the published {TEST_SHA256}')
    torch.save(test, args.data_dir / 'test.pt')
    info = dict(rows=len(test), prediction_tokens=len(test) * (SEQ - 1), token_sha256=digest)
    (args.data_dir / 'test_manifest.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
