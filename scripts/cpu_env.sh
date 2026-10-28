#!/usr/bin/env bash
# CPU pod setup, idempotent: the pinned CPU Python for tests, data prep and reviews, and the unprivileged reviewer
# user. Pod-local disk does not survive a pod restart, so run this again after one (review/codex.sh does).
#
#   scripts/cpu_env.sh            # as root
set -euo pipefail
VENV="${LOUD_CPU_VENV:-/opt/loud-cpu-venv}"

if ! "$VENV/bin/python" -c 'import torch, yaml, pytest' 2>/dev/null; then
    uv venv --quiet --python 3.12 "$VENV"
    # torch as pinned by the Noise repo (2.10.0), CPU build; the rest for data prep and tests
    uv pip install --quiet --python "$VENV/bin/python" --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
    uv pip install --quiet --python "$VENV/bin/python" numpy pyyaml pytest tokenizers==0.23.1 datasets==4.8.4 tqdm
fi
id loud-auditor >/dev/null 2>&1 || useradd --system --create-home --home-dir /home/loud-auditor --shell /bin/bash loud-auditor
"$VENV/bin/python" -c 'import torch; print("cpu env ok: torch", torch.__version__)'
