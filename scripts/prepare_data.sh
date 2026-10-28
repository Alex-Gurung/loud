#!/usr/bin/env bash
# CPU: build data/train.pt, data/val.pt (utils/prepare_data.py, pinned hashes) and data/test.pt (boost-cut's held-out
# test split) with the Noise repo's own pinned packages (torch 2.10.0, tokenizers 0.23.1, datasets 4.8.4).
# The venv and the Hugging Face cache go on pod-local disk; only the ~130 MB of token files land on /mnt/disk.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${LOUD_CPU_VENV:-/opt/loud-cpu-venv}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
(cd external/noise && UV_PROJECT_ENVIRONMENT="$VENV" uv sync)
mkdir -p data artifacts/data_prep
(cd external/noise && "$VENV/bin/python" utils/prepare_data.py --output "$PWD/../../data") | tee artifacts/data_prep/prepare_data.log
# The streaming reader can abort at interpreter exit after the file is written; the hash check below decides.
"$VENV/bin/python" scripts/prepare_test.py --noise external/noise --data-dir data | tee artifacts/data_prep/prepare_test.log || true
PYTHONPATH="$PWD" "$VENV/bin/python" -m loud.refs --check
