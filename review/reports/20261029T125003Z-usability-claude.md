# Usability review (Claude), round 7: commit a387d8d

**Verdict.** Config validation is solid: misspelled options, bad values, unknown tasks or keys, bad `expect` metrics, unknown `after` ids and E16's unscorable presets all stop at submission with a clear message, and adding a variant takes one file. The first-run path on the real volume is broken, though. The shared queue still holds 208 cancelled v1 jobs, so the README's first command leaves every gate cancelled and nothing can run. `loud.status` then calls most of the stuck jobs "waiting". On one GPU, the README's account of the run order, the disk budget and how to read plots and H6 results is wrong or missing.

### 1. [major] On the live volume, `python3 -m loud.submit configs` queues nothing that can run
- where: `README.md:81`, `loud/jobqueue.py:119`, `loud/submit.py:105,111`
- evidence: `/mnt/disk/loud/artifacts/queue` has 0 queued, 0 done and 208 cancelled jobs (204 cancelled at 2026-10-29T10:54, 4 at 00:34). I copied it to scratch and ran `python3 -m loud.submit configs` against the copy. Of the 156 jobs, 54 come back `unchanged` because `_decide` returns `'unchanged'` for any cancelled job without `--rerun`. These include `verify.smoke`, `verify.score_record`, `verify.bp_b8*`, `verify.cut_*`, `verify.identity*`, `e1.bp_b8.s0-2`, `e1.bp_b4`, `e1.identity`, `e7.precondition`, nine H6 diagnostics, twelve E3 cells and seven of E4's nine cells (`e4.adapt_nowd` gets seeds 1–2 queued but seed 0 left cancelled). `loud.status` on the result shows `queued (102)`, of which **0 are ready**: 93 say `blocked: verify.score_record.s0 cancelled` (or `verify.smoke.s0` / `e1.bp_b8.s0 cancelled`) and 9 are waiting on blocked jobs. A worker launched now would sit idle for 20 minutes and exit. The dry run gives no warning: it prints `unchanged` for each of the 54, the same word it uses for an identical job that is still queued. The README calls the command "queue every seed of every config (idempotent)" and never mentions the cancelled jobs.
- fix: in `submit`, print the state with the action, e.g. `unchanged (cancelled: --rerun to queue it)`. Add one README line: the queue holds v1's cancelled jobs, so the first v2 submission is `python3 -m loud.submit configs --rerun`. That is safe while nothing is done; I checked it against the copy.

### 2. [major] `loud.status` shows jobs that can never run as "waiting"
- where: `loud/jobqueue.py:164-177` (`dependency_state`), `loud/status.py:64-67`
- evidence: `dependency_state` reports `blocked` only when a *direct* dependency is stopped or failed its check. A queued dependency that is itself blocked shows as `waiting`. I simulated this in a scratch queue by claiming and finishing every job in order. With `verify.bp_b8.s0` failing its check, the worker drains 55 jobs and stops. Status then shows 101 queued: 39 `blocked` and 62 `waiting`. Of the 62, 59 say `waiting: h6.h8_validation.s0`, which waits on the blocked `e1.bp_b8.s0`. So every H8 experiment (E6, E7, E10, E11, E13–E16) looks pending when it is in fact blocked by verification 2. With `verify.smoke.s0` crashed, status shows 14 `blocked` and 141 `waiting`. On one GPU the worker exits as `idle (no ready job)` while status still lists those 141 as `waiting`.
- fix: in `dependency_state`, recurse through queued dependencies (with a memo) and return `blocked: <dep> blocked (<root cause>)`. The README's "When a job fails or is blocked" section should also say that a verification-2 failure blocks every H8 experiment through `h6.h8_validation`.

### 3. [major] The README promises "a first read of every arm early" on one GPU, but the code orders by exact priority value, not by band
- where: `README.md:101-102` and `README.md` band list item 6 ("P1 experiments (55–64)"); `loud/jobqueue.py:41-44` (`queue_order`)
- evidence: The README says "within a priority band every arm's seed 0 runs before any seed 1, so one GPU gives a first read of every arm early", and its band list treats 55–64 as one band. `queue_order` sorts by the integer `priority` first, so all three seeds of E6 and E7 (55) run before any seed of E14 and E15 (60), which run before E2 and E4 (62) and E3 (64). I simulated the one-GPU claim order using the README's per-run estimates. The first E6 seed 1 runs at 7.8 h, the first E14 at 10.4 h, E2 at 12.0 h and E3 at 14.5 h. Every P1 arm has its seed 0 by **14.9 h**; with all P1 cells at one priority it would be **9.5 h**. Total time is the same (21.6 h).
- fix: give every config in a README band one priority value (P1: 55; E1 already shares 50), or reword the README to "within one priority value" and list the actual order.

### 4. [major] The disk budget does not fit today's free space and the 15 GB floor
- where: `README.md:56-59`, `README.md` "Disk" bullet, `worker/gpu_worker.yaml` (`LOUD_MIN_FREE_GB: "15"`)
- evidence: The README says the queue writes "about 30 GB" and that "workers pause when the shared volume has 15 GB left". `df -h /mnt/disk` shows 38 GB free, and `loud.status` reports `39.9 GB free`. 40 − 30 = 10 GB, which is under the 15 GB floor, so the queue stops before the end. In the claim order the last band is E16, whose track-3 `final.pt` files are about 1.5 GB each. The worker then reports `waiting: shared disk nearly full` and exits after 20 idle minutes, and the README gives no next step. It says only "Nothing is ever deleted", and does not say what is safe to remove (superseded runs, failed attempts, consumed H4 checkpoints) or that the floor is an env var in the worker YAML.
- fix: state the requirement in the README Status section: about 30 GB plus the 15 GB floor, so about 45 GB free before launching, and check it with `loud.status`. Add a sentence on what to free or how to lower `LOUD_MIN_FREE_GB` in `worker/gpu_worker*.yaml`.

### 5. [major] `python3 -m loud.plots <exp>` fails under plain `python3` once a run exists
- where: `README.md:78,86`, `loud/plots.py:27-30`
- evidence: The README introduces the CPU commands, plots included, with "(plain `python3`; no torch needed)". `python3 -c "import matplotlib"` gives `ModuleNotFoundError`. I made fake runs under a scratch `LOUD_ARTIFACTS`. `python3 -m loud.plots e4` then fails in `pyplot()` with `ModuleNotFoundError: No module named 'matplotlib'`, while `/opt/loud-cpu-venv/bin/python -m loud.plots e4` writes `final.png` and `curves.png`. With no runs, both say "nothing to plot yet", so the failure only shows up once there are results. `scripts/cpu_env.sh` (README:89-90) installs matplotlib in the venv "for tests and plots", which contradicts line 78.
- fix: write the plots command as `/opt/loud-cpu-venv/bin/python -m loud.plots e4`, or catch the ImportError in `pyplot()` and name the venv.

### 6. [major] Four of H6's six diagnostics have no table, and `plots --help` says h6 has tables
- where: `loud/plots.py:9` ("Diagnostics (e5, h6) are tables"), `loud/plots.py:484-487` (`TABLES` has no `h6`), `README.md` Plots paragraph
- evidence: `TABLES` covers e2, e5–e8, e10–e16. H6 outputs are read only by E7's table (`credit_dump`) and E8's table (`snr_spectrum`). `cos_to_autograd`, `linearity_check`, `synth_vs_real` (the check on whether H8's rankings carry over to the real chain) and `h8_validation` have no reader. `python3 -m loud.plots h6` prints `h6: nothing to plot yet` and always will, because h6 has no training runs. `collect.py` lists these runs under "not in the table … (unscored)". Neither the README nor `--help` says their output is `artifacts/runs/h6/<variant>/s0/<tool>.json` (written at `loud/diagnostics.py:370`).
- fix: add an `h6` entry to `TABLES` that prints one per-site or per-matrix table per tool, or at least correct the docstring and add the JSON path to the README's H6 section.

### 7. [minor] `--dry-run` crashes when the queue directory does not exist yet
- where: `loud/submit.py:88`, `loud/jobqueue.py:65`
- evidence: `LOUD_QUEUE_DIR=<scratch>/queue python3 -m loud.submit configs --dry-run` with no queue directory raises `FileNotFoundError: …/queue/lock` with a full traceback. Round 2's nit #14 set `create=False` for dry runs, but `submit(dry_run=True)` still takes the lock file. A fresh queue (or a mistyped `LOUD_QUEUE_DIR`) is exactly where a newcomer would try a dry run first.
- fix: in `JobQueue.locked`, skip locking (or treat the queue as empty) when `self.root` does not exist.

### 8. [minor] `--cancel` ignores `--dry-run`, is not atomic, and reports an unknown id as "is None"
- where: `loud/submit.py:88-93`, `loud/jobqueue.py:150-156`
- evidence: In a scratch queue, `python3 -m loud.submit --cancel e6.time_k4.s0 --dry-run` printed `cancelled e6.time_k4.s0` and moved the job to `cancelled/`. `--cancel e6.time_k8.s0 e6.nope.s0 e6.token_k4.s0` cancelled the first job and then crashed with `ValueError: e6.nope.s0 is None; only queued jobs can be cancelled` plus a traceback, leaving the third queued. Submission is checked in full first; cancellation is not.
- fix: check every id before cancelling any, honor `--dry-run`, print "not in the queue" for a missing id, and exit with `SystemExit` rather than a traceback.

### 9. [minor] Job ids inside `args` are never checked, so mistakes fail on the GPU
- where: `loud/options.py:465-483` (`diagnose_options`), `gate_options`, `compare_options`; `loud/submit.py:58-75` (`check_dependencies` reads only `after`/`after_done`)
- evidence: each of these dry-runs reports `queued` against a fully submitted scratch queue:
  - an H6 config with its `after:` line removed (`checkpoint: e1.h8_synth.s0:ckpt_20.pt`, shown as `after -`). It is ready at once at priority 30, so a single GPU claims it before E1 finishes and it crashes on a missing file.
  - `checkpoint: e1.h8_synth.s1:…` with `after: [e1.h8_synth.s0]`.
  - `ckpt_30.pt`, a fraction no run saves.
  - `e1.bp_b4.s0:ckpt_20.pt` (bp_b4 keeps no checkpoints).
  - a gate reading `h6.credit_dump_c55.s0`.
  - a compare against `verify.bp_b8_scrip.s0`.
- fix: in `plan`, collect the job ids named by `checkpoint`, gate `values` and compare `a`/`b`. Require each to be in the job's `after`/`after_done`, and require a checkpoint's fraction to be one of 00/10/20/50/100.

### 10. [minor] The dry run hides `after_done`, so compare and gate jobs look unconditioned
- where: `loud/submit.py:105`
- evidence: the dry run prints only `job["after"]`. `verify.bp_b8_match.s0`, `verify.identity_match.s0`, `e7.precondition.s0` and `e13.gate_theta*.s0` all print `after -`, though each waits on two runs. The 40-character column also overflows on most experiment cells.
- fix: print `after_done` too (e.g. `after … | after_done …`), one dependency list per line when it is long.

### 11. [minor] Stale names and numbers in the README and `--help`
- where: `README.md:54`, `loud/plots.py:528`, `loud/plots.py:526-534`
- evidence:
  - "The 154 jobs": a directory submission queues **156** (the newest commit added `h6.synth_vs_real_c20/c100`).
  - `plots --help` says "e1 ... e12, or 'all'". The experiments are e1–e16, h6 and verify, and E12 is now a follow-up.
  - `python3 -m loud.plots foo` and `plots E4` both create `artifacts/plots/<name>/` and print "nothing to plot yet" instead of rejecting a name with no runs and no table.
- fix: update the count and the help string, and reject names that have neither runs nor a `TABLES` entry.

### 12. [minor] No guidance for a single GPU on what to launch and when to relaunch
- where: `README.md:64-72`, `worker/gpu_worker.yaml:5,20` (`backoffLimit: 0`), `loud/status.py:52`
- evidence: the launch block says "one H100 (comparable wall-clocks); start several", and the YAML header shows `for i in 1 2`. The user has one GPU, maybe two, and the README never says to launch `gpu_worker.yaml` once, or `gpu_worker_2.yaml` for two GPUs. A worker ends on idle (20 minutes), on the disk floor, on preemption, or when the pod dies (`backoffLimit: 0`, `restartPolicy: Never`), and in every case only a manual `kubectl create` brings work back. Status then says `LEASE EXPIRED: requeued at the next claim`, and on one GPU there is no next claim.
- fix: add a short "one GPU" paragraph: which YAML to launch, that the worker must be relaunched after it exits (with the status states that mean that), and change the status note to `requeued; start a worker if none is running`.

### 13. [minor] When a gate says "skip", its arms stay queued as "blocked" forever, and cancelling them takes one id per job
- where: `configs/e7/precondition.yaml`, `configs/e13/gate_theta*.yaml`, `README.md:167-177`
- evidence: when `e7.precondition` or `e13.gate_theta*` fails its `expect`, the plan says skip. The six E7 arms (or three per θ) then show `blocked: e7.precondition.s0 failed its check`, the same wording as a broken verification. The README explains blocked jobs only as errors to fix, and `--cancel` takes explicit job ids.
- fix: in the README, say that a failed `gate` job means skip and its arms should be cancelled. Let `--cancel` take a config path or an `<experiment>.<variant>` prefix.

### 14. [minor] Jargon is used before it is defined
- where: `README.md:10-21` and following
- evidence: the Terms list relies on undefined words:
  - "Boost and Cut" (inside the H8 definition, never defined);
  - "track 1/2/3", "the 10M record", "boost-cut";
  - the ranges `1M`/`10M`/`30M`, which are 488/4,880/14,648 sequences (`loud/options.py:20`);
  - "P1"/"P3", used at lines 47-56 but explained only later, by the band list.
- fix: add one-line Terms for Boost, Cut, the tracks, the ranges and P1–P3.

### 15. [minor] Submitting `configs/followups/e8` as its proxy configs instruct also queues the H8 arms with a placeholder setting
- where: `configs/followups/e8/snr.yaml:6` ("submit this directory then"), `configs/followups/e8/h8_{control,rate,snr}.yaml:2,19`
- evidence: the proxy arms' build condition says to submit the directory. That directory also holds the H8 setting, whose headers say "set row_assign to E6's better assignment before submitting", and whose value is a guess (`token`). Submitting the directory as told queues those arms with the guess, and changing it afterwards needs `--rerun` or a new variant name.
- fix: split into `followups/e8/proxy/` and `followups/e8/h8/`, or make `row_assign` a required edit (leave it unset so validation fails until it is chosen).

### 16. [nit] Leaving out `seeds` silently gives a single seed 0
- where: `loud/run.py:65`
- evidence: removing `seeds:` from `configs/e6/time_k4.yaml` queues only `e6.noseeds.s0`, with no note. The README never mentions the default; only `run.py`'s docstring does.
- fix: print a note when the default is used.

### 17. [nit] Config errors print a traceback, unlike dependency errors
- where: `loud/submit.py:97` (no `except` around `plan`)
- evidence: `row_asign` prints about 10 traceback lines before `ValueError: …: unknown h8 options: ['row_asign']`. Unknown `after` ids exit cleanly through `SystemExit`.
- fix: catch `ValueError` in `main` and `raise SystemExit(str(exc))`.

### 18. [nit] On the live queue, the status cancelled section is one unbroken 4 KB line
- where: `loud/status.py:74-75`
- evidence: `cancelled (208): e1.bp_b4.s0, …` is one unbroken line.
- fix: print the count and, say, the first ten ids.

### 19. [nit] `loud.status` never shows which queue it read
- where: `loud/status.py:29-32`
- evidence: with a mistyped `LOUD_QUEUE_DIR` it prints an empty queue, no error and no path.
- fix: print `queue: <path>` on the first line, and warn if it does not exist.

### 20. [nit] The README layout misses modules and config directories
- where: `README.md` Layout block
- evidence: the layout lists neither `gate.py` (the `gate` task), `noop.py`, `runlog.py`, `paths.py`, `common.py` nor `pergrad.py` (which the Status table names). `configs/` is described as "one YAML per variant", with no mention of `verify/`, `h6/` or `followups/`.
- fix: add them.

## Checked and fine
- With a populated scratch queue, `python3 -m loud.submit configs --dry-run` lists 156 jobs and skips every `followups/` file. Naming `configs/followups/e8` (or a single file in it) submits it, smoke runs included. A single follow-up file without its smoke run is rejected with "after names unknown jobs".
- Bad inputs all fail at submission with the file path and a clear message: a misspelled option, a bad value (`row_assign: tim`), `rows: 100`, `rows: "128"`, an unknown task (`h9`), an unknown key (`afer`), `expect: {val_los: …}`, a gate `expect` on an unknown value, `seeds: 0`, a missing `experiment`, an unknown `after` id, an `after` id without `.s<seed>`, an unknown diagnose `tool`, and `arch: ngram` at `preset: 38m` or with a non-default `ngram_rows`.
- Adding a variant: one copied file, an E6 arm at K = 32 (`rows: 16`, `variant: time_k32`), validates and dry-runs as three queued jobs. `delta_table` and the plots pick it up with no code change. A copy that keeps the old variant name is reported as a conflict.
- On a missing queue, `loud.status` prints empty sections. For a crashed job it shows the exit code, last log line and log path, and a direct dependent shows `blocked: <dep> failed` or `cancelled`.
- `collect.py --out <scratch>` on fake runs gives a readable mean ± sd table with seeds, GPU, code and commit, and lists runs that are not counted. Names match from config to job id, run directory, table and plot titles.
- `python -m loud.refs --check` against the live `data/` and `external/` passes every pin in about 5 s.
- The README's priority bands, smoke list, verification targets and tolerances (3.944812/3.936956 ± 1e-5; 3.7083/3.7014 ± 0.02; 4.1772 ± 0.02; prefix 0.003/0.01) and E6/E7/E13 gates match the configs.
- Using the README's per-run estimates, the simulated one-GPU claim order totals about 21.6 h. That agrees with "about 21 GPU-hours", and with P1 including E1 taking about 14 h.
- The worker and queue constants match the README: 1200 s idle exit, 15 GB floor, 30 s heartbeat, 300 s lease, 3 expiries.
- The README's example config validates.
- Tests: `pytest --collect-only` finds 110 tests, as the README says. `tests/test_queue.py` and `tests/test_worker.py` pass (13 tests, 24 s). The full suite was unfinished when I stopped it: 66 of 110 passed in 36 min, no failures, with load average 21–38 from other reviewers' suites. So the README's "about 15 minutes" is **unverified** here; an idle pod would settle it.
