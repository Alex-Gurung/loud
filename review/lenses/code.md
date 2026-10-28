# Lens: code quality and efficiency

Would a strong engineer accept this code as it is? Look for these, in order of importance.

1. **Correctness.** Bugs and edge cases:
   - a zero-size held-out set, `stop_after`, a retried attempt;
   - races in the queue (`flock` on CephFS, several workers);
   - error paths that leave the queue or a run directory inconsistent.
2. **Slop**, meaning code with no reason to exist:
   - dead options and parameters;
   - an abstraction with a single caller;
   - defensive branches for states that cannot occur;
   - comments that narrate or restate the code, docstrings that overclaim;
   - logic duplicated across files, options that no config uses, two names for one thing.
3. **Efficiency on the GPU path**, meaning the training loop in `loud/proxy.py` and everything it calls:
   - a host-device sync every step (`.item()`, `torch.cuda.synchronize`, logging that forces a sync);
   - per-step allocations and extra float32 copies;
   - recompilation (shape changes, `requires_grad_` toggles under `torch.compile`);
   - per-step work that could be done once, diagnostics that cost time while disabled.

   Estimate the cost where you can.
4. **Tests.** Do they test behavior that matters? Would they catch the bugs above? Are any slow or flaky?

Run the tests from the source copy: `PYTHONPATH=. /opt/loud-cpu-venv/bin/python -m pytest -q -p no:cacheprovider tests`.
