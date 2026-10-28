# usability review: Codex

- source: frozen copy `d26f79e0562383f4` at 20261028T162515Z (git 3bb0197 plus the working tree)
- reviewer: Codex 0.164.0, model gpt-6.1-sol, session 01a18354-8287-7660-8d53-d95df194d7d4
- focus: Round 2. Round 1's findings and fixes are in review/LEDGER.md; check that the major ones marked fixed are. The README was rewritten: check every command and claim in it against the code, especially the queue semantics (attempt directories, resubmission, cancel, gpu requirement), the gates in 'What runs, and the gates', and the 'When a job fails or is blocked' steps. Use a scratch queue (LOUD_QUEUE_DIR, LOUD_ARTIFACTS) for anything that writes.

Three verified issues remain: the documented recovery procedure cannot restart a failed-check gate, changed configs can revive cancelled jobs, and invalid option values still reach workers. The CPU suite passes all 28 tests, but does not cover these cases.

### 1. [major] Following the failed-gate recovery instructions leaves dependents blocked
- where: `README.md:117`, `loud/jobqueue.py:111`
- evidence: The README says to fix the cause and resubmit the failed job’s config. A failed check is stored in `done`; resubmitting the same config returns `unchanged`, even after a code fix. I reproduced `FAILED CHECK RESUBMIT: unchanged state: done`. Dependents remain blocked by its recorded failed check.
- fix: Document that failed-check jobs require `--rerun`, with a concrete command. Distinguish them from crashed jobs, which plain resubmission requeues.

### 2. [major] Changing a cancelled job’s config silently requeues it without `--rerun`
- where: `loud/jobqueue.py:111–115`
- evidence: The README and submit help promise cancelled jobs change only with `--rerun`. I submitted a noop config, cancelled it, changed `args.val_loss` from 1 to 2, and resubmitted without that flag. Submission returned `queued`, and the job became runnable. The cancelled-state protection applies only when the config hash is unchanged.
- fix: Handle `cancelled` independently of config equality: require `--rerun` before reactivating it. Add a changed-config cancellation regression test.

### 3. [major] Invalid training option values still pass submission
- where: `loud/options.py:90`, `loud/train.py:63`, `README.md:163`
- evidence: A proxy config containing `args: {stop_after: wrong}` passes `loud.submit --dry-run` and prints `queued`. Validation accepts the string unchanged; evaluating the training loop’s `min(steps, o['stop_after'] or steps)` raises `TypeError`. This contradicts the README’s claim that bad values fail at submission.
- fix: Validate `stop_after` and diagnostic intervals as nonnegative integers, and validate the remaining numeric and boolean options before submission.

## Checked and fine

- All 28 CPU tests passed in 138.45 seconds.
- README submission, dry-run, status, collection, and CLI help commands ran against scratch state.
- `loud.refs --check` passed for pinned code, downloads, and all three data splits.
- Misspelled options and unknown tasks fail with the config path; omitting seeds defaults to seed 0 as documented in help.
- Attempt isolation, stale-lease rejection, scheduling updates, and failed-check exclusion from averages passed existing tests.
- Status displayed crash details, expired leases, lost workers, failed-check blockers, and disk-pause reasons.
- Verification targets and experiment dependency lists match the README’s described gates.
- GPU execution was not exercised.