"""Task 'noop': a plumbing check for the queue and workers (no GPU work). Sleeps, then succeeds or fails.

    args: {seconds: 5, fail: false, val_loss: null}
"""
import time

from .runlog import Run


def run_task(spec, run_dir):
    args = spec.get('args', {})
    run = Run(run_dir, spec)
    for step in range(int(args.get('seconds', 0))):
        run.metrics.write(dict(step=step + 1, train_loss=0.0, train_seconds=step + 1))
        time.sleep(1)
    if args.get('fail'):
        raise RuntimeError('noop asked to fail')
    return run.finish(dict(val_loss=args.get('val_loss'), complete=True, train_seconds=args.get('seconds', 0)),
                      spec.get('expect'))
