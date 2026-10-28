"""Run a pinned record script with some of its recipe replaced, the file itself unchanged (E12's 30M re-tunes).

    python launch.py <script> '<settings JSON>' <train rows or 0> <seed or -> [script args...]

The settings replace fields of the script's Settings dataclass where main() builds it; the script's own config.json
records the recipe it ran (asdict(settings)). Train rows > 0 keep only the first that many rows of the hash-checked
train.pt, so the epoch, its schedule and its warmdown are defined on them. A seed replaces the script's SEED constant,
which seeds what the record's own seeds do (its init, data order and probe draws; '-' keeps the published 0). Imports
only the standard library: the
script resolves its own checkout's modules, and torchrun runs this file once per GPU.
"""
import ast
import dataclasses
import functools
import importlib.util
import json
from pathlib import Path
import sys


def settings_fields(script):
    """The init fields of the script's Settings dataclass, read from its source without importing it (submission runs
    on the CPU pod, without torch): every annotated field not declared field(..., init=False)."""
    tree = ast.parse(Path(script).read_text())
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Settings'), None)
    if cls is None:
        raise ValueError(f'{script} has no Settings class')
    fields = set()
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = node.value
            hidden = (isinstance(value, ast.Call) and any(k.arg == 'init' and isinstance(k.value, ast.Constant)
                                                          and k.value.value is False for k in value.keywords))
            if not hidden:
                fields.add(node.target.id)
    return fields


def main():
    script, settings, rows, seed, *argv = sys.argv[1:]
    settings, rows = json.loads(settings), int(rows)
    spec = importlib.util.spec_from_file_location('record', script)
    module = importlib.util.module_from_spec(spec)
    sys.modules['record'] = module
    spec.loader.exec_module(module)  # defines everything; the script's own `if __name__ == '__main__'` does not run
    fields = {f.name: f for f in dataclasses.fields(module.Settings) if f.init}
    if set(settings) - set(fields):
        raise SystemExit(f'{script}: Settings has no {sorted(set(settings) - set(fields))}')
    settings = {k: tuple(v) if isinstance(fields[k].default, tuple) else v for k, v in settings.items()}  # JSON lists
    module.Settings = functools.partial(module.Settings, **settings)
    if seed != '-':
        if not hasattr(module, 'SEED'):
            raise SystemExit(f'{script} has no SEED constant to replace')
        module.SEED = int(seed)
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
