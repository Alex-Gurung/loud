# usability review: Codex

- source: frozen copy `e32e300d93b77497` at 20261028T150538Z (git 3bb0197 plus the working tree)
- reviewer: Codex 0.164.0, model gpt-6.1-sol, session 01a1830b-9cd7-7370-85ad-d1be24ea9310

The CPU workflow exposes four usability defects: failed checks enter reported averages, config typos silently change the experiment, resubmission ignores scheduling changes, and low-disk workers appear idle. All 12 CPU tests passed, but they do not cover these cases; GPU execution was not tested.

### 1. [major] Failed-check runs enter the results averages despite the documented exclusion
- where: `loud/collect.py:67`, `loud/collect.py:107`
- evidence: `collect.py --help` says “incomplete or failed-check runs are listed apart.” Supplying complete results with losses 3.7 (`pass`) and 9.0 (`fail`) produced `n=2`, mean `6.3500 ± 3.7477`. The `check failed` flag does not prevent inclusion. A real CPU noop job with a failed expectation also appeared in the main table.
- fix: Exclude rows whose check is `fail` from aggregates and include them in the separate excluded-run listing.

### 2. [major] Misspelled config keys silently remove seeds and verification dependencies
- where: `loud/run.py:28`, `loud/run.py:38`, `loud/run.py:48`
- evidence: Top-level keys have no unknown-key validation. A config containing `seedz: [1, 2]` passed `loud.submit --dry-run` and planned only `demo.variant.s0`. With `afterr: [verify.bp_b8.s0]`, dry-run succeeded and printed `after -`. These mistakes silently change the requested runs or bypass their verification gate.
- fix: Reject unknown top-level keys before applying defaults. Document the seed-0 default when neither `seed` nor `seeds` is supplied.

### 3. [major] Resubmitting a queued job silently ignores dependency and priority changes
- where: `loud/run.py:53`, `loud/jobqueue.py:101`
- evidence: The config hash excludes scheduling metadata. I submitted a job blocked by `demo.badcheck.s0`, changed its YAML to `after: []` and `priority: 1`, then resubmitted with `--priority 2`. Submission printed `unchanged`; the queued record retained priority 100 and its old dependency. Even the newer-code branch updates only code metadata.
- fix: Apply changes to dependencies, priority and other submission metadata for queued jobs, or reject those changes with an explicit message instead of reporting `unchanged`.

### 4. [major] Disk-blocked workers immediately overwrite their status with `idle`
- where: `loud/worker.py:92`, `loud/worker.py:102`
- evidence: With `free_gb()` injected to return 1 GB against a 2 GB threshold, one loop iteration reported `waiting: shared disk nearly full`, then immediately reported `idle` before sleeping. Both reports overwrite the same worker record, so status normally shows idle while ready jobs cannot start. Eventually the worker reports `exited: idle`.
- fix: Preserve the disk-wait state through the polling interval and identify disk exhaustion in the exit reason if the idle timeout still applies.

### 5. [minor] Invalid batch sizes pass submission validation
- where: `loud/options.py:41`, `loud/proxy.py:152`
- evidence: A proxy config with `args: {batch: 0}` passed `loud.submit --dry-run`. Validation only checks divisibility; execution later evaluates `len(split.train) // o['batch']`, which divides by zero.
- fix: Validate that `batch` and `micro_batch` are positive integers before checking divisibility.

## Checked and fine

- Ran the README’s CPU-safe status, verification/E1 dry-run submission, collection with scratch output, and test commands.
- All 12 tests passed, including CPU worker execution, SIGTERM release, lease recovery and preservation of partial runs.
- Unknown tasks and misspelled task options fail during submission with specific error messages.
- A crashing CPU job appeared under `failed` with its exception; a failed expectation appeared as `check fail` and blocked its dependent job.
- Adding a variant requires one YAML file; experiment, variant and seed names remain consistent across queue IDs, run directories and table rows.
- All review-created files and queue activity stayed in the supplied scratch directory.