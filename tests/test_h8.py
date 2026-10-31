"""H8 on CPU (tiny models, PR #40 run eagerly as the reference), each check in its own process: see h8_checks.py."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
CHECKS = ('h8_gradients', 'h8_boost', 'h8_schemes', 'h8_reforward', 'h6_tools', 'h8_run', 'h8_latent',
          'h8_latent_run', 'h8_shaping', 'h8_runway', 'h8_ngram', 'h8_variants_run', 'h8_arch', 'h8_end_merge', 'h6_snr', 'h8_snr', 'h6_synth_vs_real', 'h8_knobs')


@pytest.mark.parametrize('check', CHECKS)
def test_h8(check, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(REPO), LOUD_ARTIFACTS=str(tmp_path / 'artifacts'),
               LOUD_QUEUE_DIR=str(tmp_path / 'queue'), PYTHONDONTWRITEBYTECODE='1')
    proc = subprocess.run([sys.executable, str(REPO / 'tests' / 'h8_checks.py'), check, str(tmp_path)], cwd=REPO,
                          env=env, capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-6000:]
