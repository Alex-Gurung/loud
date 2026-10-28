"""Small helpers shared by the harness, the queue and the worker (stdlib only)."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a line being written when a reader looks
                pass
    return rows


def atomic_text(path, text):
    """Write by atomic replacement with fsync, so readers on the shared volume never see half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.writing-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(temporary, 0o644)  # mkstemp makes 0600; other pods and users read these records
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, default=str) + '\n')


class JsonlLog:
    """Append-only JSON lines, flushed per record."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a')

    def write(self, record):
        self.handle.write(json.dumps(record, default=float) + '\n')
        self.handle.flush()

    def close(self):
        self.handle.close()


def git_info(repo, paths):
    """HEAD of a checkout, and whether any of `paths` differs from it (modified or untracked); None fields when git is
    unavailable (e.g. inside a slim image)."""
    def run(*args):
        try:
            return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, timeout=30,
                                  check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    commit = run('rev-parse', 'HEAD')
    status = run('status', '--porcelain', '--untracked-files=all', '--', *paths)
    return dict(commit=commit, dirty=None if status is None else bool(status))


def load_config(path):
    """A run config: YAML (preferred) or JSON."""
    with open(path) as f:
        if Path(path).suffix in ('.yaml', '.yml'):
            import yaml
            return yaml.safe_load(f)  # parse errors name the file
        return json.load(f)
