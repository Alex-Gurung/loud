"""Run a pinned record script with some of its recipe replaced, the file itself unchanged (E12's 30M re-tunes).

    python launch.py <script> '<settings JSON>' <train rows or 0> [script args...]

The settings replace fields of the script's Settings dataclass where main() builds it; the script's own config.json
records the recipe it ran (asdict(settings)). Train rows > 0 keep only the first that many rows of the hash-checked
train.pt, so the epoch, its schedule and its warmdown are defined on them. Imports only the standard library: the
script resolves its own checkout's modules, and torchrun runs this file once per GPU.
"""
import dataclasses
import functools
import importlib.util
import json
from pathlib import Path
import sys


def main():
    script, settings, rows, *argv = sys.argv[1:]
    settings, rows = json.loads(settings), int(rows)
    spec = importlib.util.spec_from_file_location('record', script)
    module = importlib.util.module_from_spec(spec)
    sys.modules['record'] = module
    spec.loader.exec_module(module)  # defines everything; the script's own `if __name__ == '__main__'` does not run
    fields = {f.name for f in dataclasses.fields(module.Settings) if f.init}
    if set(settings) - fields:
        raise SystemExit(f'{script}: Settings has no {sorted(set(settings) - fields)}')
    module.Settings = functools.partial(module.Settings, **settings)
    if rows:
        load_rows = module.load_rows

        def first_rows(path, count):
            data = load_rows(path, count)
            return data[:rows] if Path(path).name == 'train.pt' else data
        module.load_rows = first_rows
    sys.argv = [script, *argv]
    module.main()


if __name__ == '__main__':
    main()
