"""The training methods on CPU (tiny models, short sequences), each check in its own process: see method_checks.py."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
CHECKS = ('proxy_gradient', 'proxy_run', 'identity_window', 'identity_step', 'identity_run', 'diagnostics')


@pytest.mark.parametrize('check', CHECKS)
def test_method(check, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(REPO), LOUD_ARTIFACTS=str(tmp_path / 'artifacts'),
               LOUD_QUEUE_DIR=str(tmp_path / 'queue'), PYTHONDONTWRITEBYTECODE='1')
    proc = subprocess.run([sys.executable, str(REPO / 'tests' / 'method_checks.py'), check, str(tmp_path)], cwd=REPO,
                          env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-6000:]
