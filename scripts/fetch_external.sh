#!/usr/bin/env bash
# Recreate external/ (gitignored) at the commits and files pinned in external.json, and fetch the pinned downloads.
# Then `python3 -m loud.refs --check --no-data` must pass.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
import hashlib, json, subprocess, urllib.request
from pathlib import Path
pins = json.loads(Path('external.json').read_text())
for name, repo in pins['repos'].items():
    path = Path(repo['path'])
    if not path.exists():
        subprocess.run(['git', 'clone', repo['url'], str(path)], check=True)
    subprocess.run(['git', '-C', str(path), 'fetch', '--quiet', 'origin', repo['commit']], check=False)
    subprocess.run(['git', '-C', str(path), 'checkout', '--quiet', repo['commit']], check=True)
    print(name, repo['commit'][:7])
for name, d in pins['downloads'].items():
    path = Path(d['path'])
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != d['sha256']:
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(d['url'], path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == d['sha256'], name
    print(name, 'OK')
PY
