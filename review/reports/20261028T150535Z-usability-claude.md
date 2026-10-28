# usability review: Claude

- source: frozen copy at /var/tmp/loud-review/20261028T150535Z-claude/src (git 3bb0197 plus the working tree, 2026-10-28 15:05 UTC)
- reviewer: Claude (loud-reviewer instructions, general-purpose agent), lens usability

## Verdict

The happy path works. Every README command that runs without a GPU ran as written. The noop worker flow, dependency blocking and resubmitting a failed job behave as documented, and the 12 tests pass. But submission checks only option *names*. Misspelled top-level keys, a malformed `expect`, a copied config with the old variant name, and an `after` id that doesn't exist are all queued; two of these cost a full GPU run or mislabel one. `loud.status` misreports two failures the lens asks about (a full disk, a dead pod), the git commit recorded for every run is false, and the README's step-for-step verification has nothing behind it.

### 1. [major] Misspelled or misplaced top-level config keys are silently dropped, so the job runs defaults under the variant's name
- where: `loud/run.py:27-50` (no key whitelist; `args`/`expect` read with `.get` at 46; seeds default at 38)
- evidence: a config with `arg: {batch: 4}` and `expct: {...}` dry-runs clean as `e1.bp_b4y.s0/s1` with no error. It would train proxy defaults (batch 8, no check). A config with no `seeds` silently becomes seed 0 (README:73 says "seeds listed inside"). `seed: [0, 1, 2]` gives a raw `TypeError: int() argument must be ... not 'list'`, and `seeds: 3` gives `'int' object is not iterable`. `--dry-run` never prints args, so the user can't see the drop.
- fix: in `specs_from_config`, reject keys outside `{experiment, variant, task, seeds, priority, after, args, expect, notes}`, require `seeds` as a non-empty list of ints, and print args in `--dry-run`.

### 2. [major] `expect` is not validated at submission; a malformed one kills the run after training and scoring, before `result.json` is written
- where: `loud/runlog.py:61-66,74`; only `args` is validated, at `loud/run.py:34`
- evidence: `expect: {val_loss: 3.5}` was queued without complaint. The worker run failed with `TypeError: cannot unpack non-iterable float object`, and the run directory has no `result.json`. In `proxy.run` this point comes after the full epoch and `score_in_subprocess` (`loud/proxy.py:229-234`), so a whole GPU run is lost to collect. A misspelled metric (`val_los`) passes too, then always fails its check and blocks dependents.
- fix: validate at submission that `expect` maps known result keys (`val_loss`, `test_loss`, `heldout_loss`, `train_seconds`, …) to `[number, number]`.

### 3. [major] Adding a variant by copying a YAML and forgetting `variant:` queues the copy under the old name, then stops halfway
- where: `loud/submit.py:38-58` (no duplicate-id check; dry-run at 43-47 never reads the queue; conflicts raise mid-loop)
- evidence: I copied `configs/e1/bp_b4.yaml` to `bp_b2.yaml` with `batch: 2` and submitted the directory. `--dry-run` listed `e1.bp_b4.s0` twice with no warning. The real submit queued `e1.bp_b4.s0-2` with `batch: 2`, then raised `ValueError: e1.bp_b4.s0 is queued with a different config`, so the real `bp_b4` and all of `bp_b8` were never queued.
- fix: plan everything first. Reject duplicate job ids within one submission, and check each job against the queue (in `--dry-run` too, printing queued/updated/unchanged/conflict) before writing anything. Warn when the file stem differs from `variant`.

### 4. [major] Changing `after` or `priority` of a queued job is silently ignored ("unchanged"), and so is `--priority`
- where: `loud/jobqueue.py:100-110`; `config_hash` excludes both fields (`loud/run.py:53-54`); `--priority` help at `loud/submit.py:31`
- evidence: I edited a blocked job's config to `after: []` and `priority: 0` and resubmitted: output `unchanged t.after_checkfail.s0`, and status still shows `priority 4 blocked after t.checkfail.s0`. `--priority 0` on another job: also `unchanged`. Only `--rerun` applied it, and the README documents `--rerun` only for finished jobs. This is exactly the path a user needs when a verification check fails and all of E1 sits `blocked`.
- fix: on resubmitting a queued job, update `priority`, `after`, `config_path` and `notes` and report `updated`. Add a README paragraph on unblocking.

### 5. [major] A full shared disk shows as `idle`, then `exited: idle`
- where: `loud/worker.py:91-104`
- evidence: with `--min-free-gb 100000`, I sampled the worker record 40 times: `waiting: shared disk nearly full` once, `idle` 6 times, `exited: idle` 33 times. The disk-full report is overwritten by `report('idle')` in the same loop pass. The pod log says only `idle for 8.0s; exiting`, and status shows `exited: idle` next to a `ready` job. The README never mentions the 2 GB floor or `LOUD_MIN_FREE_GB`, and the real volume has 6.3 GB free (100% used).
- fix: when space is low, sleep and `continue` without reporting idle, and exit as `exited: shared disk nearly full`. Print the reason, and state the floor in the README.

### 6. [major] Every run records git commit `3bb0197`, "not dirty", but that commit contains none of the code
- where: `loud/common.py:90-92` (`--untracked-files=no`); README:97 claims `config.json` records the git commit
- evidence: `git -C /mnt/disk/loud ls-files` lists only the handoff `.md`; every `loud/` file is untracked. The live queued jobs and the snapshot manifest still say `{"commit": "3bb0197…", "dirty": false}`, so the table's `commit` column will show `3bb0197` with no `+dirty`.
- fix: compute dirty over the snapshot's own files, untracked included (`git status --porcelain -- loud run.py collect.py external.json`). Show the code digest in the table.

### 7. [major] The pod's pre-flight check passes with the data missing, although the README says it checks the data hashes
- where: `loud/refs.py:83-90`; README:36; `worker/launch_worker.sh:40-43`
- evidence: `LOUD_DATA=<empty dir> python3 -m loud.refs --check` prints `"train.pt": "missing"` (and the same for val and test) and exits 0. With `test.pt` missing, each proxy job would fail only at final scoring. It doesn't bite today because the data exists.
- fix: raise or exit non-zero on any missing split.

### 8. [major] Nothing performs the README's "must match it step for step" check, and E1 does not wait for it
- where: README:57 ("The E1 cells wait for the verification runs"), README:109-110; `configs/e1/bp_b8.yaml:8`, `configs/e1/bp_b4.yaml:8`
- evidence: grep finds no code that compares `bp_b8_onefwd` with `bp_b8_script`. Both configs check only `val_loss` within ±0.02. E1's `after` lists only `verify.score_record.s0` and `verify.bp_b8.s0`, not onefwd, script or smoke. A new researcher has no command that tells them whether verification 2 passed.
- fix: add a small compare task (per-step `train_loss` of two runs within a tolerance, recorded as `check`) and put it, plus `verify.smoke.s0`, in E1's `after`. Otherwise correct README:57 and 110.

### 9. [major] The `reference` task never passes the job's seed to the script, so extra seeds are copies
- where: `loud/reference.py:31` (formats only `{data}` and `{out}`); `loud/options.py:59-63`; `configs/verify/bp_b8_script.yaml:12`
- evidence: argv has no `--seed`, and a `{seed}` placeholder would raise `KeyError` at run time (not at submission). `submit configs/verify/bp_b8_script.yaml --seeds 0,1,2` therefore queues three runs of the script's default seed, and `collect` reports their mean ± sd as a seed spread.
- fix: format argv with `seed=spec['seed']`. Reject a reference config with more than one seed unless its argv contains `{seed}`.

### 10. [major] `collect.py --help` says failed-check runs are listed separately; they enter the means, and crashed runs are not mentioned at all
- where: `loud/collect.py:8` (docstring shown by `--help`), `:64-68`, `:107-110`
- evidence: `t.checkfail` (check fail) appears as a table cell, flagged `check failed` and averaged in. `t.crash` and `t.expect_shape`, which have run directories but no `result.json`, appear nowhere in the output.
- fix: leave `check == 'fail'` runs out of the means and list them under "not in the table". Also list seed directories without `result.json` as "no result".

### 11. [minor] A dead pod's job and worker stay `running` in status after the lease expires
- where: `loud/status.py:30-44`; recovery runs only inside `claim()` (`loud/jobqueue.py:171`); README:58-59 implies automatic requeue
- evidence: I SIGKILLed a worker and its child. At heartbeat age 317 s (lease 300 s), status still listed `t.slow.s0` under running and `rev-deadpod running`. Only rows already in an `exited…` state are ever hidden (`status.py:34`), so dead-pod rows stay forever. A pod that fails `refs --check` at bootstrap never writes a worker record, and status never reads `bootstrap_logs/`.
- fix: label running jobs past `LEASE_SECONDS` as "lease expired (requeued at next claim)", mark workers unseen past the lease as lost, and show bootstrap logs ending in `startup exit=<non-zero>`.

### 12. [minor] No cancel command; cancelled jobs are absent from status; `blocked` gives no cause or remedy
- where: `loud/jobqueue.py:123-129` (no CLI calls it); `loud/status.py:45-56` (no cancelled section; the blocked line is at 49-50)
- evidence: grep for `cancel` in the README, submit, status and worker finds nothing. To clear the dead pod's job I had to call `JobQueue._recover_stale` and `cancel` from Python. A dependent of a cancelled job shows `blocked`, and the cancelled job itself appears nowhere.
- fix: add `loud.submit --cancel <id>` and a cancelled section in status, show "blocked by X (failed / check fail / cancelled)", and add a README paragraph "when a job fails or is blocked".

### 13. [minor] A bad `after` entry waits forever with no warning
- where: `loud/run.py:48`; `loud/jobqueue.py:133-150`
- evidence: `after: [t.okk.s0]` (typo) and `after: ['t.ok.s{seed}']` with seed 1 (no such job) both show `waiting` indefinitely; the worker idles out. `after: verify.bp_b8.s0` written as a string becomes 15 one-character dependencies (dry-run shows `['v', 'e', 'r', …]`).
- fix: require a list. At submission, reject ids that are neither queued nor part of this submission. Status could show `waiting (unknown: t.okk.s0)`.

### 14. [minor] Option values are checked only on the GPU, with cryptic errors, and errors don't name the config file
- where: `loud/options.py:34-56`; `loud/common.py:95-101`; `loud/proxy.py:159-166`
- evidence: `sequences: 10m`, `preset: 38M`, `checkpoint_dtype: bf16` and score `splits: [val, tset]` all pass submission. They fail on the GPU as `KeyError: '10m'` and similar. `checkpoint_dtype` fails only after the compile warmup, and `tset` only after val is scored. `unknown proxy options: ['bacth']` does not say which file. YAML syntax errors cite `"<unicode string>"`, because the text is parsed rather than the file.
- fix: check values in `proxy_options`/`score_options`, prefix errors with the config path, and call `yaml.safe_load` on the open file.

### 15. [minor] `submit` freezes `/mnt/disk/loud` (`LOUD_REPO`), not the checkout it runs from, whatever state that tree is in
- where: `loud/paths.py:9`; `loud/snapshot.py:24`; `loud/submit.py:48`
- evidence: submitting from the frozen copy made snapshot `61d5cf…`. Its manifest says `source /mnt/disk/loud`, and its `jobqueue.py` hash is `cc61de…` (the author's in-progress edit, with a new `heartbeat(jid, worker_id, attempt_dir, …)` signature) next to the old `worker.py`. The copy's own `jobqueue.py` is `cd29e7…`. The README never says this.
- fix: snapshot the package's own checkout unless `LOUD_REPO` is set explicitly, and print the source in submit's output. Combined with #6, warn when the snapshot's files are modified or untracked.

### 16. [minor] The end-to-end tests run against any exported `LOUD_QUEUE_DIR`
- where: `tests/test_worker.py:13-14`
- evidence: with `LOUD_QUEUE_DIR=<scratch>/leak-queue` exported, `test.ok/after/bad` and a worker record landed in that queue and the test failed. Inside a worker pod the variable points at the real queue (`worker/gpu_worker.yaml:52-53`), so the test's worker would claim real jobs on a CPU.
- fix: set `LOUD_QUEUE_DIR` in `env_for`.

### 17. [minor] The README's direct `run.py` example uses the run directory of a job that is queued; a crashed direct run blocks its own retry
- where: README:47; `loud/run.py:78`; `loud/worker.py:122-125`; `loud/runlog.py:49-50`
- evidence: `configs/verify/bp_b8.yaml` maps to `verify.bp_b8.s0`, which is queued in the live queue. Reproduced with noop: after a direct run, the worker wrote `job.json` and `stdout.log` into that directory, then failed with `FileExistsError … already holds a run; resubmit under a new variant or with --rerun`. A direct run that crashed (mine, in flex-attention compile) left an empty `metrics.jsonl`, and the retry fails the same way. `run.py` has no `--rerun`.
- fix: default direct runs to a separate path (for example `s<seed>.direct`) or refuse when the job id is in the queue. Make the message fit direct runs (set the directory aside, or give `--out`).

### 18. [minor] The README claims "CPU-tested on a tiny model", but the repo has no way to exercise a proxy variant before a GPU
- where: README:13; `configs/verify/smoke.yaml:1`
- evidence: `run.py` with `{preset: 2m, stop_after: 2, official_eval: false}` under the CPU venv fails after about 4 min with `InductorError: IndexError` in `utils/model.py:31` (flex attention). No test runs the proxy. `smoke.yaml` says "First job on a new worker", but it is one job run once, the README never mentions it, and no verify config waits for it.
- fix: either add a CPU path (eager attention behind a test-only flag) with a pytest, or drop the claim. Make the verify configs depend on `verify.smoke.s0`, fix its comment, and mention it in the README.

### 19. [minor] Jargon is used before it is defined
- where: README:5, 25-27, 57, 103-106, 122-123
- evidence: "stylo", "EIDF", "E1" (never defined in the README), "ANVIL II", "Cut chain", "identity path", "verification 3" and PR numbers appear without a gloss. A newcomer has to read the 30 KB handoff to follow the Status table and the queue section.
- fix: a short glossary, with pointers to the handoff sections.

### 20. [nit] `submit` on a directory skips `.yml`
- where: `loud/submit.py:23`
- evidence: it globs only `*.yaml`, though `load_config` accepts `.yml` and JSON. A directory holding only `x.yml` gives `no configs found`.

### 21. [nit] README commands assume the repo root as working directory
- where: README:38-47
- evidence: `python3 -m loud.worker` from another directory fails with `No module named 'loud'`.
- fix: say "from `/mnt/disk/loud`".

### 22. [nit] `collect.py --experiment` silently skips writing `table.md` and `runs.json`
- where: `loud/collect.py:116`

### 23. [nit] The table cannot show whether verification 1 passed
- where: `loud/collect.py:61`; `configs/verify/score_record.yaml:12`
- evidence: the table prints val to 4 decimals (3.9448), while `score_record`'s tolerance is 1e-5, and there is no check column. Status prints 5 decimals.

### 24. [nit] Small status display oddities
- where: `loud/status.py:31`, `:63`
- evidence: `workers (N)` counts rows it then hides, and jobs without `expect` print `check None`.

## Checked and fine
- `loud/paths.py`: `LOUD_ARTIFACTS` controls runs, code and results; `LOUD_QUEUE_DIR` controls the queue. Nothing I ran wrote under `/mnt/disk/loud/artifacts`, `data` or `.git/index`; the live queue still holds the same 11 jobs and one snapshot.
- README commands on CPU: `loud.status`, `submit --dry-run`, real `submit`, `collect.py --out`, the `--help` of every CLI (`run.py --help` works without torch), and the tests (12 pass in 37 s when `LOUD_QUEUE_DIR` is unset).
- Rejected at submission with a clear message: unknown task, unknown proxy/noop/score/reference option names, `batch % micro_batch`, dots in variant names. A multi-seed config with `run.py` and no `--seed` gives a clear message.
- Worker with `noop`: claims by priority, runs from the snapshot, records done/failed, and status shows the exception's last line (`RuntimeError: noop asked to fail`).
- Dependents of a crashed job and of a failed check both show `blocked`. Resubmitting a failed job without `--rerun` requeues it and its dependents return to `waiting`. `--rerun` on a done job archives the record in `history/` and moves the run to `s0.superseded1`.
- After a pod death, lease recovery on the next claim moves the partial run to `s0.attempt1`. SIGTERM release is covered by `tests/test_worker.py`.
- Run directories hold `config.json` (spec, snapshot digest, external pins, environment), `metrics.jsonl`, `stdout.log`, `job.json`, and `result.json` with per-metric check details. Names agree across config, job id (`exp.variant.s<seed>`), run directory and table.
- `collect` excludes `.attempt`/`.superseded` directories and lists incomplete runs (smoke) separately.
- README hashes (`test.pt cb7a29f0…`, checkpoint `499dff9f…`) match `external.json`; the 20-minute idle exit matches `LOUD_IDLE_EXIT_SECONDS=1200`.

Two things from the review itself, not findings. Other reviewer agents share this scratch directory and its `queue/` and `artifacts/`, so I worked in `scratch/usability/` and removed the jobs I had first queued into the shared `scratch/queue`. The author was editing `/mnt/disk/loud/loud/jobqueue.py` during the review (changed at 15:10), so the live tree no longer matches the frozen copy.
