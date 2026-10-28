"""Content-addressed code snapshots under artifacts/code/<digest>/.

Every submitted job records the snapshot it runs; every worker runs its own loop from the snapshot taken at startup.
Editing the repository never changes queued or running work (the stylo worker convention), and the run's config.json
names the exact code. Snapshots are never deleted.
"""
import os
from pathlib import Path
import shutil
import uuid

from .common import digest, file_digest, git_info, read_json, utc_now, write_json
from .paths import CODE, SOURCE

INCLUDE = ('loud/*.py', 'run.py', 'collect.py', 'external.json')


def files_of(repo):
    repo = Path(repo)
    paths = sorted({p for pattern in INCLUDE for p in repo.glob(pattern) if p.is_file()})
    return {str(p.relative_to(repo)): file_digest(p) for p in paths}


def snapshot(repo=SOURCE, root=CODE):
    files = files_of(repo)
    key = digest(files)[:16]
    destination = Path(root) / key
    if (destination / 'MANIFEST.json').exists():
        verify(destination)
        return destination
    temporary = Path(root) / f'.partial-{key}-{uuid.uuid4().hex[:8]}'
    for rel in files:
        target = temporary / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(repo) / rel, target)
    write_json(temporary / 'MANIFEST.json', dict(digest=key, created_at=utc_now(), source=str(repo),
                                                git=git_info(repo, list(files)), files=files))
    try:
        os.rename(temporary, destination)
    except OSError:  # another process made the same snapshot first
        shutil.rmtree(temporary, ignore_errors=True)
    verify(destination)
    return destination


def verify(destination):
    manifest = read_json(Path(destination) / 'MANIFEST.json')
    for rel, expected in manifest['files'].items():
        if file_digest(Path(destination) / rel) != expected:
            raise RuntimeError(f'frozen code changed: {destination}/{rel}')
    return manifest
