"""The log layout every run writes, matching the benchmark's: config.json, metrics.jsonl, result.json (plus heldout.jsonl
and the H4 checkpoints where a task makes them)."""
import os
import platform
import socket
import sys
import time

from .common import JsonlLog, read_json, utc_now, write_json
from .paths import PINS


def environment():
    env = dict(host=socket.gethostname(), python=platform.python_version(), image=os.environ.get('LOUD_RUNTIME_IMAGE'))
    torch = sys.modules.get('torch')  # a task that trains has imported it; plumbing tasks (noop, compare) need not
    if torch is None:
        return env
    env.update(torch=torch.__version__, cuda=torch.version.cuda)
    if torch.cuda.is_available():
        env.update(gpu=torch.cuda.get_device_name(0), gpu_count=torch.cuda.device_count())
    try:
        import triton
        env['triton'] = triton.__version__
    except ImportError:
        pass
    return env


def code_identity():
    """The frozen snapshot this process runs from (set by the worker or loud.run), and the pins it carries."""
    snapshot = os.environ.get('LOUD_CODE_DIR')
    manifest = None
    if snapshot:
        try:
            manifest = read_json(f'{snapshot}/MANIFEST.json')
        except FileNotFoundError:
            pass
    pins = read_json(PINS)
    return dict(snapshot=snapshot, digest=manifest and manifest['digest'], git=manifest and manifest.get('git'),
                external={k: v['commit'] for k, v in pins['repos'].items()})


class Run:
    """One run directory. Creates it empty, writes config.json at start and result.json at the end."""

    def __init__(self, run_dir, spec, extra_config=None):
        self.dir = run_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        if (self.dir / 'result.json').exists() or (self.dir / 'metrics.jsonl').exists():
            raise FileExistsError(f'{self.dir} already holds a run; resubmit under a new variant or with --rerun')
        self.start_wall = time.time()
        self.config = dict(spec, started_at=utc_now(), code=code_identity(), environment=environment(),
                           **(extra_config or {}))
        write_json(self.dir / 'config.json', self.config)
        self.metrics = JsonlLog(self.dir / 'metrics.jsonl')

    def update_config(self, **fields):
        self.config.update(fields)
        write_json(self.dir / 'config.json', self.config)

    def finish(self, result, expect=None):
        result = dict(result, wall_seconds=round(time.time() - self.start_wall, 1), finished_at=utc_now(),
                      gpu=self.config['environment'].get('gpu'))
        if expect:
            result['check'] = check(result, expect)
        write_json(self.dir / 'result.json', result)
        self.metrics.close()
        return result


def check(result, expect):
    """expect: {metric: [target, tolerance] or {min, max}}; pass only if every metric is within its bounds."""
    details = {}
    for metric, bound in expect.items():
        value = result.get(metric)
        if isinstance(bound, dict):
            ok = value is not None and bound.get('min', value) <= value <= bound.get('max', value)
            details[metric] = dict(value=value, **bound, ok=ok)
        else:
            target, tolerance = bound
            details[metric] = dict(value=value, target=target, tolerance=tolerance,
                                   ok=value is not None and abs(value - target) <= tolerance)
    return dict(status='pass' if all(d['ok'] for d in details.values()) else 'fail', metrics=details)
