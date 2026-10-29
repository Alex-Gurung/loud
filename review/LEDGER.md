# Review ledger

One row per finding, triaged by the author after checking it. Findings are grouped by report (`reports/<file>`), and
`#n` is the finding's number there. Status is **fixed** (what changed), **rejected** (the evidence that the finding is
wrong or not worth it) or **deferred** (why, and what it waits on). "Same as" points to the row that holds the fix.

## Round 1: the harness before any GPU run (frozen copy of 2026-10-28 15:05 UTC)

### 20261028T150532Z-code-codex.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | blocker | An expired attempt can overwrite its retry's files, and can finish the retry's job | fixed | Each attempt writes to its own `s<seed>.attempt<k>/`, which is also its lease token: heartbeat, finish and release must name it. A done attempt is renamed to `s<seed>/` under the lock. Test: `test_each_attempt_has_its_own_directory_and_a_done_one_becomes_the_run`, including a same-worker reclaim |
| 2 | major | A crash between write and unlink leaves a job claimable twice | fixed | `_move` rewrites the record in place, then `os.replace`s it. Test: `test_interrupted_transition_leaves_one_record` |
| 3 | major | Failed-check runs enter the averages | fixed | `collect.counts` excludes them; they are listed apart with the reason. Test: `test_table_leaves_out_failed_checks_and_flags_differences` |

### 20261028T150535Z-research-codex.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | The verification gate allows 0.02; the plan stops at 0.01 | fixed | All gates are 0.01, against the ledger's bp38_s0 (val 3.7083, test 3.7014) |
| 2 | major | No code compares the port with the script step by step, and E1 does not wait for it | fixed | New `compare` task. `verify.bp_b8_match` (onefwd against the script) and `verify.bp_b8_ledger` (against the published log) are in every E1 backprop cell's `after` |
| 3 | major | The nine trajectory means are never kept | fixed | `means: true` in every E1 cell, each mean scored on the held-out set at the end (`result.means_heldout`). `bp_b4` saves them (bf16, about 0.7 GB per seed) |
| 4 | major | Failed checks enter the means | fixed | Same as code-codex #3 |
| 5 | minor | H3 uses four forwards, not one | fixed | `HeldoutScorer` scores all held-out rows in one forward; the warmup compiles that shape |

### 20261028T150538Z-usability-codex.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | Failed checks enter the averages | fixed | Same as code-codex #3 |
| 2 | major | Misspelled top-level keys silently drop seeds or gates | fixed | `run.KEYS` whitelist; `seed` with `seeds` rejected; types of `seeds` and `after` checked. Test: `test_bad_configs_fail_at_submission` |
| 3 | major | Resubmitting ignores priority and `after` changes | fixed | A waiting job takes newer code, priority, `after`, `gpu` and config path ("updated"). Test: `test_resubmission` |
| 4 | major | A disk-blocked worker reports idle | fixed | It reports `waiting: shared disk nearly full (… GB free)` and exits as `exited: idle (shared disk …)` |
| 5 | minor | `batch: 0` passes submission | fixed | `options.positive`, plus the range, held-out size, at least one update, preset and dtype checked at submission |

### 20261028T150535Z-code-claude.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | Worker tests inherit `LOUD_QUEUE_DIR`, so they can claim real jobs | fixed | `env_for` and the method tests set it to a temporary queue |
| 2 | major | Resubmission drops `after` and priority changes | fixed | Same as usability-codex #3 |
| 3 | major | Every run records git `3bb0197` as clean, though none of the code is committed | fixed | `git_info` counts modified and untracked files among the snapshot's own files. The table shows the code digest. Committing the code waits on the user |
| 4 | major | `b_simple` can never exceed the batch size | fixed | Uses the unbiased \|G\|² = \|g\|² − tr(Σ)/B; logs `trace_cov`, `mean_sq` and `true_sq` so they can be averaged over updates |
| 5 | major | No test runs the proxy loop | fixed | `tests/method_checks.py`, all on CPU: `proxy_gradient` checks that micro_batch 1 is exactly the float32 mean of per-sequence gradients, micro_batch = batch is exactly one backward, and the warmdown matches. `proxy_run` covers the whole loop and its outputs |
| 6 | minor | A plain resubmission brings back cancelled jobs | fixed | A cancelled job comes back only with `--rerun`. Failed jobs are still requeued (documented: the usual retry after a fix) |
| 7 | minor | `refs --check` passes with data missing | fixed | A missing split raises, so the pod bootstrap and `prepare_data.sh` fail |
| 8 | minor | About 650 kernel launches per update in gradient averaging | fixed | `torch._foreach_zero_/add_/div_`, about 50 launches; still bit-identical (checked by `proxy_gradient`) |
| 9 | minor | micro_batch 1 may be the largest avoidable cost | deferred | `verify.bp_b8` against `verify.bp_b8_onefwd` measures it; decide the experiment default from their step times and losses |
| 10 | minor | `ckpt_100.pt` written for incomplete runs | fixed | Saved only at the last update of a complete run |
| 11 | minor | Disk-full status overwritten | fixed | Same as usability-codex #4 |
| 12 | minor | The `collect.py` docstring is false | fixed | Same as code-codex #3 |
| 13 | minor | Dead code and unused options | fixed | Removed `EXTERNAL`, `data_dir`, `--once`, the `fractions` parameter, and `PerSequenceGrads`' `select`/`dtype`. `state_dicts` now feeds the held-out scoring of the means. The running-job message no longer suggests cancel |
| 14 | minor | The loss loop is copied three times | fixed | One `heldout.mean_loss`, used by H3, the val curve and the test score |
| 15 | minor | Bad option values fail only on the GPU | fixed | Same as usability-codex #5 |
| 16 | minor | Weak and slow tests | fixed | Order test checks a regression constant. New tests for the table, compare, config checks, crash safety and stale attempts. Noop jobs no longer import torch. Lease tests keep their 2 s sleeps |
| 17 | minor | A job can sit in two state directories | fixed | Same as code-codex #2 |
| 18 | minor | A direct run from another checkout snapshots `/mnt/disk/loud` | fixed | Snapshots freeze `paths.SOURCE`, the checkout `loud/` sits in; submit prints it |
| 19 | nit | Warmup comment wrong at batch 4 | fixed | The shared loop warms the scorer up with its own row count |
| 20 | nit | `non_blocking` from pageable memory | fixed | Removed |
| 21 | nit | The final held-out score is logged twice | fixed | The in-loop score skips the last update. Checked in `proxy_run` |
| 22 | nit | `anvil_beta` sits inside `lr`; a local shadows `run()` | fixed | `anvil_beta` is a top-level metrics field. The loop moved to `train.train` |
| 23 | nit | `loud.status` creates directories and rereads done jobs per queued job | fixed | `JobQueue(create=False)`; done and stopped jobs read once |

### 20261028T150535Z-research-claude.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | Gates accept 0.02; the ledger has seed-0 numbers and a per-step log | fixed | Gates: val 3.7083 ± 0.01, test 3.7014 ± 0.01. `verify.bp_b8_ledger` compares the train loss at every logged update (every 61) within 0.01 |
| 2 | major | The step-for-step check never runs | fixed | Same as research-codex #2 |
| 3 | major | `bp_b4` uses boost-cut's batch-8 rates at batch 4 with no anchor or reason | fixed | Reason stated in `configs/e1/bp_b4.yaml` (the plan's H1 is `backprop_preset.py`'s recipe). New cell `e1.bp_b4_pr40` uses PR #40's own recipe, so the choice is measured |
| 4 | major | Failed checks enter the means | fixed | Same as code-codex #3 |
| 5 | minor | `b_simple` biased | fixed | Same as code-claude #4 |
| 6 | minor | Means off by default and discarded | fixed | Same as research-codex #3 |
| 7 | minor | `sequences: 1M` cannot reproduce the published 10M log | fixed (doc) | README: log matching uses `10M` with `stop_after`. The Cut-chain port's check will follow it |
| 8 | minor | GPU and code differences between compared variants go unflagged | fixed | The table flags `GPU differs` and `code differs` across one experiment's cells. A config may require a GPU (`gpu: H100`), and workers honour it |
| 9 | minor | Checkpoints mix bf16 (0–50%) and fp32 (100%) | fixed | `ckpt_100.pt` is its own file in the same dtype; `final.pt` stays exact |
| 10 | minor | The held-out curve summary and plot scripts are missing | fixed / deferred | The table shows `heldout_min`. Plot scripts are listed as not started in the README; they come with each experiment's results |
| 11 | minor | No proxy test | fixed | Same as code-claude #5 |
| 12 | nit | The PR #20 citation (3.7081) cannot be verified | fixed | Configs and README cite ledger bp38_s0 (3.70835) |
| 13 | nit | `ckpt_100` for incomplete runs | fixed | Same as code-claude #10 |
| 14 | nit | PR #40 imports an unpinned eval.py | fixed | `archive/10M/utils/eval.py` pinned |
| 15 | nit | README says every val loss comes from eval.py in a fresh process | fixed | README and `bp_b8_script.yaml` say the reference task's val is the script's own `evaluate()` |
| 16 | nit | E1 configs carry no `expect` | rejected | An `expect` is a gate: a failure takes the run out of the table and blocks dependents. An E1 cell 0.015 off is a finding to report, not an invalid run. The verification gates check correctness |

### 20261028T150535Z-usability-claude.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | Misspelled keys are dropped silently | fixed | Same as usability-codex #2. The dry run prints each job's action against the queue |
| 2 | major | A malformed `expect` kills the run after training | fixed | `options.check_expect` checks the shape and that the metric is in the task's result, at submission |
| 3 | major | A copied config with the old variant name queues under that name and stops halfway | fixed | `submit.plan` rejects duplicate ids and notes a file stem that differs from the variant. Every conflict is checked before anything is written |
| 4 | major | `after`/`priority` changes ignored | fixed | Same as usability-codex #3 |
| 5 | major | A full disk shows as idle; the floor is undocumented | fixed | Same as usability-codex #4; the floor is in the README and `loud.status` |
| 6 | major | The recorded git commit is false | fixed | Same as code-claude #3 |
| 7 | major | The pre-flight check passes with data missing | fixed | Same as code-claude #7 |
| 8 | major | Nothing performs the step-for-step check | fixed | Same as research-codex #2. Every verification job also waits for `verify.smoke` |
| 9 | major | The reference task never gets the seed | fixed | `{seed}` placeholder (the backprop script config passes `--seed {seed}`). Several seeds without it are rejected, as are unknown placeholders |
| 10 | major | Failed checks in means; crashed runs not mentioned | fixed | Same as code-codex #3. Seed directories without `result.json` are listed |
| 11 | minor | A dead pod's job and worker stay "running" | fixed | Status marks `LEASE EXPIRED` jobs and `LOST` workers, and lists pods whose startup failed |
| 12 | minor | No cancel command; cancelled jobs absent; "blocked" gives no cause | fixed | `loud.submit --cancel`; a cancelled section; `dependency_state` returns the cause shown by status |
| 13 | minor | A bad `after` id waits forever | fixed | Unknown ids (in neither the queue nor the submission) stop the submission; `after` must be a list |
| 14 | minor | Option values unchecked; errors don't name the file | fixed | Same as usability-codex #5. Errors are prefixed with the config path; YAML is parsed from the file |
| 15 | minor | submit freezes `/mnt/disk/loud`, not its own checkout | fixed | Same as code-claude #18 |
| 16 | minor | End-to-end tests use an exported queue | fixed | Same as code-claude #1 |
| 17 | minor | A direct run uses the queued job's directory; a crash blocks its retry | fixed | Direct runs work in `s<seed>.direct<k>/` and become `s<seed>/` only if that is free and the queue does not hold the job |
| 18 | minor | No CPU path for the proxy; smoke not a dependency | fixed | Same as code-claude #5; smoke gates the verification jobs, and the README describes it |
| 19 | minor | Jargon before definition | fixed | Terms section at the top of the README |
| 20 | nit | `.yml` skipped | fixed | Both suffixes |
| 21 | nit | Commands assume the repo root | fixed | README says so |
| 22 | nit | `--experiment` silently writes nothing | fixed | Documented in `--help` |
| 23 | nit | The table cannot show that verification 1 passed | fixed | `check` column (`pass n`) |
| 24 | nit | Status count and `check None` | fixed | Counts recent workers only; `-` when there is no check |

## Round 2: round-1 fixes and the H5 identity-path port (frozen copy of 2026-10-28 16:25 UTC)

### 20261028T162505Z-research-codex.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | E6 redraws the signed basis every update, so a cycle's K windows are not complementary rows of one basis | fixed | The probe holds one signed basis per cycle (`s // K`) while still drawing signs as PR #26 does, so every K consumes the same random stream. Test: `identity_window` checks a cycle's windows without rewinding the generator, and that a new cycle draws a new basis |
| 2 | minor | The match gates can pass without the requested updates | fixed | `compare` requires each of the first `first` steps that `b` logs (all of them by default) to be in `a`, and rejects duplicate steps. Tests in `test_compare_matches_two_runs_by_step` |
| 3 | minor | GPU work by the means leaks into the training time | fixed | The paused section ends with `torch.cuda.synchronize()` |
| 4 | minor | Identity and E6 cells keep no H4 checkpoints or means | fixed | Checkpoints and means on in every E1 and E6 cell (disk: 50 GB free, about 0.45 GB per run) |

### 20261028T162510Z-code-codex.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | The comparison gate passes with steps missing | fixed | Same as research-codex #2 above |
| 2 | major | Untimed means leak into the training time | fixed | Same as research-codex #3 above |
| 3 | major | `collect` silently omits failed and crashed attempts | fixed | Attempts of seeds that never finished are listed under "no finished run", marked running when the queue holds them |
| 4 | minor | `stop_after: -1`, `lr_mult: -5` pass submission | fixed | `options.check_types`: each option has its default's type; counts are integers >= 0; `lr_mult` > 0. Tests added |
| 5 | minor | The whole-batch gradient test compares against storage that the trainer rewrites | fixed | `direct()` clears `.grad` first and clones what it returns |

### 20261028T162515Z-usability-codex.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | The README's recovery steps leave a failed-check gate blocked (it is done, so a plain resubmission changes nothing) | fixed | README separates crashed jobs (resubmit) from failed checks (resubmit with `--rerun`, with the command) |
| 2 | major | A cancelled job with a changed config is requeued without `--rerun` | fixed | Cancelled jobs change only with `--rerun`, whatever the config. Test in `test_cancel` |
| 3 | major | Bad option values still reach workers | fixed | Same as code-codex #4 above |

### 20261028T162514Z-code-claude.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | A non-string `gpu` or non-int `priority` passes submission, then crashes every worker's claim and `loud.status` | fixed | `_specs` checks both. Tests in `test_bad_configs_fail_at_submission` |
| 2 | major | `heldout_min` mixes in the nine means' held-out scores | fixed | Rows with `mean_decay` are left out |
| 3 | major | `verify.smoke` does not run every H1–H4 path | fixed | Smoke is a complete 57-update run on the 1M range: H2, H3, all five checkpoints with optimizer state, the nine means (saved) and official val and test scoring. Comment and README corrected (crashes are not retried) |
| 4 | minor | Three SIGTERM releases fail a job | fixed | Releases never use up an attempt; only lease expiries count (`expiries`). The worker checks for a stop signal before starting the child, and catches a lost lease on the stop path. Test: `test_release_requeues_keeps_the_attempt_and_uses_up_nothing` |
| 5 | minor | The crash-safety test cannot catch round 1's bug | fixed | The test now kills only the record's rename and the unlink of a record file (as the reviewer's injection does) |
| 6 | minor | The means test cannot fail; `strict=False` hides key mismatches | fixed | `proxy_run` runs 11 updates (two observations) and asserts every mean's score differs from the final weights'. Loading a mean raises on unexpected keys, or on missing keys other than buffers |
| 7 | minor | No check covers the per-rank draws or the cross-sequence sum | fixed | `identity_step` runs two virtual GPUs against PR #26's FastRelay at ranks 0 and 1 plus Boost.step's sum; the weights must be equal |
| 8 | minor | The worker's dependency-order test passes with dependencies ignored | fixed | The dependent job has the best priority, and the test asserts it was claimed after its dependency finished |
| 9 | minor | The new queue features have no tests | fixed (part) | Tests for the gpu requirement, release, unknown and circular dependencies, cancel/rerun and bad priority/gpu values. Direct runs and the LostLease kill path are still checked by hand only |
| 10 | minor | The identity update syncs once per sequence, and eight estimators' idle hooks run on every forward | fixed | Loss computed without a sync. Each estimator's hooks are attached only while it measures (`identity.attach`). GPU timing to confirm |
| 11 | minor | Training options fail only on the GPU | fixed | Already fixed by `check_types` |
| 12 | minor | A self-dependency (or cycle) waits forever | fixed | `submit.check_dependencies` walks the graph over the submission and the queue. Test added |
| 13 | nit | README test time stale | fixed | |
| 14 | nit | `--dry-run` and direct runs create queue directories | fixed | `create=False` |
| 15 | nit | Release on the stop path does not catch `LostLease` | fixed | Same as #4 |
| 16 | nit | Duplicated done/stopped maps and sort key; the dead `state` field | fixed | `JobQueue.finished()` and `queue_order()`; the field is gone (the directory is the state) |
| 17 | nit | `score.json` points at a path that no longer exists | fixed | Records the checkpoint's file name and hash |

## Round 3: the H6 diagnostic toolkit (frozen copy of 2026-10-28 16:39 UTC)

### 20261028T163926Z-research-codex.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | E5 cannot make its decision: there is no recovered-credit measurement | deferred | E5's ridge solve (512 x 512 per token and site, toward the identity guess) is E5's own diagnostic. It is built on `token_map_svd`'s frozen-attention map and comes with E5, before E5's runs are queued |
| 2 | major | E8 measures gradient directions, not the pending update | fixed | `checkpoint_optimizer` saves ANVIL II's state and next-step beta and fast weight with each checkpoint (on in `bp_b4`). `direction_stats` then takes directions from the pending update `g.lerp(w * fast + (1 - w) * slow, beta)`, and records `directions_from` |
| 3 | major | Linearity is measured on the summed loss, in float32 only | fixed | Per token, as a Cut probe sees it: each token's own loss with attention frozen (no cross-token cancellation). Quantiles of nonlinearity and signal, in float32 and in bf16 |
| 4 | minor | Finite differences replace the HVP without an accuracy check | fixed | The CPU check compares the curvature with an exact Hessian-vector product (double backward through dense attention): within 2%. The relative step is recorded |
| 5 | minor | A one-sequence noise estimate gives NaN | fixed | `direction_stats` needs batch >= 2 (options and toolkit) and records the sequence count. The noise remains a variance over 8 sequences: read it as such |

### 20261028T163931Z-code-codex.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | A one-sequence batch gives NaN noise and reports completion | fixed | Same as research-codex #5 above |
| 2 | major | `token_map_svd` allocates about 10 GiB in reverse mode | fixed | Forward mode (`jacfwd`: 512 inputs against 8 x 512 outputs), chunks of 32 tokens, and the identity map left out |
| 3 | major | Curvature probes compute and clone every block gradient to use one | fixed | `torch.autograd.grad` with respect to the perturbed matrix only |

### 20261028T162514Z-research-claude.md (round 2)

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | `row_offset_per_step` is a no-op (fresh signs every probe), and its test rewinds the generator | fixed | Same as round-2 research-codex #1 (one basis per cycle; the test no longer rewinds). The reviewer measured that holding the basis barely changes what reaches momentum, because each update reads different sequences. So `e6.identity_k4_fresh` (128 rows of a fresh basis each update) is the control that says whether stitching matters. The partial first cycle is documented |
| 2 | minor | The drift check measures a sequence the momentum holds, against fast and slow separately | fixed | It measures a held-out sequence, and also logs the cosine with ANVIL II's blend `w * fast + (1 - w) * slow`. Configs say to compare it across K, not with 0.9 |
| 3 | minor | No test of the multi-rank emulation | fixed | Same as round-2 code-claude #7: `identity_step` at two ranks |
| 4 | minor | Each per-step comparison waits for the final-loss gate it explains | fixed | New `after_done` dependency (done, passed or not). The three compare jobs use it; E1 stays gated on everything. Test in `test_dependencies_wait_and_block_on_failed_check` |
| 5 | minor | `e1.identity` keeps no means; identity and E6 keep no checkpoints | fixed | Already fixed after this copy was frozen (round-2 research-codex #4): checkpoints and means in every E1 and E6 cell |
| 6 | nit | `e6.identity_k1.s0` repeats `e1.identity.s0` | rejected | Kept as a determinism check (about 6 GPU-minutes); the config says so |
| 7 | nit | The smoke test never runs the identity path | fixed | `verify.identity_smoke`: 57 updates through E6's windows, the drift check, H3, H4 and scoring; `verify.identity` waits for it |

## Round 4: E5's diagnostic, multi-GPU jobs and E12, plots (frozen copies of 2026-10-28 21:53 and 22:01 UTC)

### 20261028T215315Z-code-codex.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | Adding `gpus` to every config hash makes unchanged queued jobs look like changed configs | fixed | `gpus` enters the hash only above 1; every queued job's hash checked unchanged; `test_one_gpu_jobs_keep_their_config_hash` |
| 2 | major | The worker counts nvidia-smi's GPUs, ignoring `CUDA_VISIBLE_DEVICES` | fixed | `gpu_info` narrows to the visible indices (or counts UUIDs); `test_gpu_info_follows_cuda_visible_devices` |

### 20261028T220115Z-research-codex.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | E5's diagnostic measures a different credit from PR #40's chain (every MLP unit, autograd target, no own-query) | fixed (partly) | Probes now go through the chain's routed units (`route_k` 512, its ranking), and the target is a finite-difference direct cycle at each site run the same way; autograd is reported beside it. Attention stays frozen, as the plan's diagnostic says; the docstring and configs state that the own-query response is left out |
| 2 | major | E12's re-tunes and baseline have one seed; the record script fixes `SEED = 0` | fixed | `script_seed: true` sets the script's SEED from the job seed (`loud/launch.py`); record and every re-tune run seeds 0-2. Seed 0's reproduction is a separate compare job, `e12.record_match` (PR #46's published log pinned), which the re-tunes wait for |
| 3 | minor | No E12 plot of validation loss over time with the 3.5 target | fixed | `val_time.png` for e12 |

### 20261028T215315Z-code-claude.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | E5's embedding launch is biased by x0's skip paths (recovery 0.83-0.92 at mid0-mid2 on the record checkpoint); a mid0 launch recovers 0.9999 | fixed | Not a code error: that bias is what E5 asks about, now reported for the real checkpoints. The reviewer's alternative is added as a second launch (block 0's direct cycle read at mid1-mid7, x0 held): a two-cycle hybrid. The E5 table shows both, and the method check tests the mid0 launch on a live model |
| 2 | major | `gpus` in every config hash breaks resubmission | fixed | Same as 20261028T215315Z-code-codex.md #1 |
| 3 | major | E12 re-tunes run one seed; the record script fixes SEED | fixed | Same as 20261028T220115Z-research-codex.md #2 |
| 4 | major | The README is stale: five tools, plots not started, one GPU per worker; nothing on E5, E12, `gpus`, `launch.py`, `pr46`, plots, `cpu_env.sh`; PR #46 has no future-draw knob | fixed | README rewritten after the merges (status, the 8-GPU worker, E12's adaptation and its dropped future-draw knob) |
| 5 | minor | The condition number is dominated by the RMS norm's radial direction | fixed | Reported with and without the smallest singular value; the E5 table uses the latter |
| 6 | minor | Ridge 1e-3 x mean diagonal pulls well-conditioned directions to the guess | fixed | Ridges 0, 1e-5, 1e-4, 1e-3; the table reads 1e-4 |
| 7 | minor | The method check cannot catch an MLP or bf16 error (zeroed writers, x0 lambdas zeroed) | fixed | A live random model: the chain arithmetic's direct cycles must match autograd at every site (p10 > 0.999), and the mid0 launch must recover the later sites with bf16 tracking float32. Doing so found and fixed a cancellation: the MLP input now uses PR #40's exact `norm_difference` |
| 8 | minor | `test_launch` runs the live repo's `launch.py` | fixed | `paths.SOURCE` |
| 9 | minor | 36,864 SVDs plus ~110k solves; one SVD per site gives everything | fixed | One SVD per site and chunk gives the condition numbers and every ridge's solution; the site that is the probe itself needs none |
| 10 | minor | Nothing extracts when the 60% run crosses 3.5; E12's final plot mixes merged records with unmerged proxies | fixed | `plots/e12/summary.md`: final, plain-weights val and the time to 3.5, with the merged/plain note |
| 11 | minor | A misspelt `settings` field fails only on the 8-GPU node; JSON lists replace tuple fields | fixed | Fields checked at submission from the script's source (`launch.settings_fields`, no torch); lists become tuples where the default is a tuple |
| nits | nit | jacrev docstring; no progress during reference runs; "queued" wording; `cpu_env.sh` rebuilds on any failed import | fixed (partly) | Docstring and wording fixed; reference progress and a narrower rebuild left as they are (the reference log is copied at the end; a rebuild takes a minute) |

## Round 5: the experiments (frozen copies of 2026-10-28 22:55 to 23:59 UTC, commit d99a223)

### 20261028T225506Z-code-codex.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | blocker | E8 accepts batch 1, whose noise variance is NaN, and completes with NaN weights | fixed | `curvature_damping` and `grad_noise_every` need batch >= 2; and every training loop (shared and population) now stops on a non-finite training loss instead of finishing "complete" |
| 2 | major | The Cut chain crashes when each virtual rank gets one row (rows == split) | fixed | The scalar moves get their own scales; `cut_windows` covers rows == split against one GPU |
| 3 | minor | E9 syncs the GPU once per micro-batch counting tokens | fixed | The count stays a device tensor |

### 20261028T231339Z-research-codex.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | E10 counts agreement among tokens with a nonzero contribution, not all tokens | fixed | `agree_over: all` is the plan's literal reading, checked against brute force; control cell `e10.agree_all_t70`. The active-token reading stays the default: under the literal one, an entry whose input is zero for more than 30% of tokens (common after the ReLU^2) is always dropped |
| 2 | major | E7 can start before the diagnostic that justifies it | fixed | New task `gate` evaluates the plan's rule on H6's credit_dump (median over sites above 0.9 at 20%, below 0.7 at 100%; `expect` takes {min, max}); `e7.precondition` gates the four ramps |
| 3 | major | `cut_prefix_match` allows 0.02 after update 20, twice the plan's seed noise, unmeasured | fixed | 0.01 over all 262 updates (0.003 over the first 20); `verify.cut_prefix_reversed` (the 8-way sum in reversed order, `reverse_sum`) and its compare job measure what rounding alone does, the reference for any relaxing |
| 4 | minor | The end merge's candidate scores bypass H3's log | fixed | Every candidate's score goes through `heldout.jsonl` (reason `merge`), for the Cut chain and the proxy (one shared `checkpoints.end_merge`) |
| 5 | minor | E3 saves no H4 checkpoints or means | fixed (partly) | The trunk's 0/10/20% and each copy's 50% and 100% (`ckpt_<f>.c<k>.pt`); no means, since after the branch there is no single trajectory and the rules are E3's merges (stated in the module) |

### 20261028T235900Z-cut-claude.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | blocker | E7's row counts exceed Dynamo's 64-graph limit, so most of its Cut updates run the chain eagerly, biasing its time test | fixed | The port lists every row count a run can use, raises the limit to twice the graphs they need and fails the job if the warmup reaches it; CPU checks match the plan against the kernel's real calls and every Cut config against its limit |
| 2 | major | No Cut config pins the GPU; E7's baseline is in another experiment, so a mismatch goes unflagged | fixed | `gpu: H100` on every Cut cell (E6, E7, E8, E11) |
| 3 | minor | Turns misses a collective mismatch when the last part has the extra group_sum | fixed | Raised in both directions (and the first failure, not a follower's); tested both ways |
| 4 | minor | E7's plots show neither its criterion nor its baseline | fixed | E7 and E11 get e6.cut_k1 as the baseline row, and `time.md` (val, plain-weights val and training minutes against the baseline) |
| 5 | minor | `e1.cut_1m.s0` and `e6.cut_k1.s0` are the same run | fixed | `configs/e1/cut_1m.yaml` removed; E1's 1M Cut number is `e6.cut_k1.s0` (README) |
| 6 | minor | The rank-by-rank end merge is about 8 of every 35 minutes | fixed | `merge_batch` (one forward of the 32 rows) in every experiment cell; the record's layout stays for verification |
| 7 | minor | `cut_prefix_match`'s bounds are uncalibrated | fixed | Same as 20261028T231339Z-research-codex.md #3 |
| 8-11 | nit | drift blend equals fast; idle `measured`; duplicated end_merge; split=2 windows check | fixed | blend dropped for the Cut chain; `measured` cleared after use; one `checkpoints.end_merge` for both trainers; the numeric windows-average check at split 2 added |

### 20261028T235900Z-proxy-claude.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | `direction_stats` never uses the pending update on a real checkpoint (its matrices include the value-embedding gates, which ANVIL II does not hold); round-3 #2 was not fixed | fixed | It now takes ANVIL II's matrices (`anvil.is_anvil_matrix`); the method check loads a state saved by `FractionCheckpoints.save` and requires 'pending update'; h6 config comments corrected |
| 2 | major | A rejected E4 update rewinds ANVIL II's clock, so its momentum schedule follows accepted updates | fixed | The clock (the schedule, not a record of data) keeps following the loop; weights, rails, energy and Adam's state are still restored. Check updated: the next update equals a fresh trainer's at that clock |
| 3 | major | E12's summary says the proxies have no merge, which they now have; the README omits the proxy's `end_merge` | fixed | The note says to compare merged with merged and plain with plain; the README's proxy paragraph describes `end_merge` (added after this copy was frozen) |
| 4 | minor | E3's held-out curve and `heldout_min` read midpoint scores | fixed | `collect.own_score` skips means, copies, midpoints and the ensemble mean, in collect and the plots |
| 5 | minor | E9 rescales to sqrt(min(m, n)), up to 1.3x what polar() gives | fixed | Scaled to `polar(pending).norm()`, so only the direction differs from ANVIL II; check updated |
| 6 | minor | E3's calibration (rule a near a batch-16 run) has no cell | fixed | `configs/e3/bp_b16.yaml` |
| 7 | minor | E8's sub-options are accepted with E8 off | fixed | `damp_directions`, `damp_every`, `damp_where` need `curvature_damping` |
| 8 | minor | No GPU smoke run gates E2, E4 or E10 | fixed | `verify.blur_smoke`, `step_control_smoke`, `cap_smoke`, `agree_smoke` gate their experiments |
| 9 | minor | E8's "needs the Cut port" is stale, and the hook cannot reach PR #40's BatchedAnvil | fixed (partly) | Comments point at `configs/e8/cut_rows128*.yaml`, which the Cut port's author is building with a BatchedAnvil hook |
| 10 | nit | Fisher dTd cost is 180 GFLOP, not 90 | fixed | |
| 11 | nit | `per_sequence_grads` with E9 is rejected though E9 keeps the plain gradient | fixed | Allowed |
| 12 | nit | Fisher token count syncs per micro-batch | fixed | Same as 20261028T225506Z-code-codex.md #3 |
| 13 | nit | The ensemble rescores the copies at the end, inside the clock | fixed | The last checkpoint's scores are the decision's, timed |

## Round 6: round 5's fixes and the Cut-chain additions (frozen copy of commit 6daad47)

### 20261029T000503Z-code-codex.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | `direction_stats` mixes a token-summed gradient with token-mean rails (2,048x apart) and skips the rails' own step | fixed | Gradients of the token-mean loss, as training; `diagnostics.pending_update` steps the fast and slow rails with g at the betas the checkpoint now saves (`fast_beta`, `slow_beta`) before blending; the method check compares it with the tensor the optimizer's next step hands `polar()` |
| 2 | minor | E8's ShapedBatchedAnvil runs PR #40's compiled update eagerly | rejected | The optimizer step is well under 1% of a Cut update (~16 s), and E8 on the chain is judged on loss, not time |

### 20261029T002425Z-research-codex.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | E8's "real test" uses autograd curvature, where the plan's forward-only version uses +- probes | fixed (scope) | The plan allows autograd as a stand-in for step-control ideas in development and describes the probes as the forward-only version; the cell is labelled a hybrid, and the probe version is listed as built only if the hybrid recovers the loss (README, "Not built, on purpose") |
| 2 | major | H6's pending update mixes gradient scales | fixed | Same as 20261029T000503Z-code-codex.md #1 |
| 3 | minor | E7's self-check runs without its cross-seed condition and on a fixed ramp | fixed | Moved to `configs/followups/e7/`, which a directory submission skips (`loud.submit`, tested); their queued jobs cancelled; the README says when and how to submit them |
| 4 | minor | E10's literal reading lacks 0.6 and 0.8 | fixed | `agree_all_t60`, `agree_all_t80` |

### 20261029T010000Z-round6-claude.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | blocker | On torch 2.13 (the GPU image's) Dynamo's config is per context, so Turns' second part compiles under the default limit of 8 and most Cut graphs end up eager | fixed | Each part runs in `contextvars.copy_context()` of the caller; the warmup fails unless every planned graph compiled. Checked on torch 2.13 (a venv on this pod): the parts see the raised limit, and the reviewer's 7m E7 ramp compiles 36 of its 36 planned graphs (8 before), none added during the run; `test_turns_parts_see_the_callers_dynamo_config` |
| 2 | major | E6's plots crash on Cut runs, which no longer log `blend`, and `plots all` stops there | fixed | The drift plot reads `blend` where there is one, else `fast` (what PR #40 steps along); the plots test has a Cut-style drift record |
| 3 | minor | E7's control variate has coefficient 1 | rejected | The plan's E7 premise is a good guess (cosine > 0.9 early), where coefficient 1 is the standard choice; a fitted coefficient would be a second knob |
| 4 | minor | E7's gate takes the median over nine sites including the embedding | fixed | `gate` values take a site prefix; the precondition reads the eight mid sites |
| 5 | minor | `direction_stats` skips the rails' update with the current gradient; `fast_beta` not saved | fixed | Same as 20261029T000503Z-code-codex.md #1 (after this copy was frozen) |
| 6 | minor | E8's Cut pair runs compiled and eager optimizer code | rejected | Same as 20261029T000503Z-code-codex.md #2: rounding-level differences in a step well under 1% of a Cut update |
| 7 | minor | E7's self-check is queued unconditionally | fixed | Same as 20261029T002425Z-research-codex.md #3 |
| nits | nit | `heldout_min` counts merge candidates; `merge_loss` duplicates `mean_loss`; garbled docstring; E6's warmup repeats an update; NaN weights after the last update | fixed (partly) | `own_score` skips reason `merge`; docstring rewritten; the shared loop raises on non-finite final weights. The duplicate scorer and the repeated warmup update stay (the record's rank-by-rank layout differs from `mean_loss`; one extra untimed update) |

## Round 7: the v2 update (frozen copy of commit a387d8d; Codex's of 3ef4441 and 81e66e0)

### 20261029T125008Z-research-codex.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | E13's diagnostic crashes: `tiers_tool`'s tuple lands in `summary` | fixed | Unpacked; `run_task` writes both files |
| 2 | major | E10's "70% agreement" keeps entries with 40% when contributions are zero | fixed | Kept where (N + \|S\|) / 2 ≥ 0.7 × tokens, N the nonzero counts; the check has the reviewer's 4-of-10 case |
| 3 | major | E1's H8 cells do not wait for `h6.h8_validation` | fixed | All three wait for it |
| 4 | major | E15's baseline never logs the handoff cosine | fixed | Every Boost run logs it; the E15 table has the column |

### 20261029T125003Z-code-codex.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | E13's diagnostic cannot serialize its result | fixed | Same as research-codex #1 |
| 2 | major | Population runs can finish with NaN weights | fixed | The shared loop's finite-weights guard before `final.pt` |
| 3 | major | E6's clock includes its logging | fixed | `h8.Untimed`: the cosine, stable ranks and E14's gradient cosine are timed apart and taken off the clock (reset before it starts) |

### 20261029T125003Z-research-claude.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | blocker | E8's SNR directions come from the update holding the batch's gradient (pure noise: mean r 0.85 against 0.18) | fixed | Every arm, both settings, takes U and V from the momentum before the batch (passed by `ShapedAnvil` and `ShapedBatchedAnvil`); zero rails fall back to the update's own; `snr_scaling` checks the basis and that the step passes the pre-batch blend |
| 2 | major | E16's n-gram tables train at the base Adam rate | fixed | `arch.embedding_rates` puts them in wte's group (300×), record #5's own rule; `h8_arch` checks the groups |
| 3 | major | E15's and E14's baselines lack the handoff cosine and effective ranks | fixed | Logged for every run |
| 4 | major | E14's gap mixes the estimator with a recipe change (`e1.bp_b4` is boost-cut's) | fixed | `e1.bp_b4_pr40` (PR #40's recipe at batch 4, three seeds, priority 50) is the gap's end and E16's fixed-model arm; `e16.bp_fixed` removed |
| 5 | minor | P1's bands delay seed 0 of E2/E3/E4; real-chain checks run before E1 | fixed | One P1 band (55); the chain's checks and the track-3 smoke at 66 |
| 6 | minor | E7: H8 applies a guess at S0 the chain does not; the gate's `S` prefix includes S0 | fixed | The prior acts at the mid sites only; gates take glob patterns, `e7.precondition` reads `S[1-8]` |
| 7 | minor | The own-query key cap (256) is on in `h8_validation` and `synth_vs_real` | fixed | `query_keys: 0` there (and in synth_vs_real's all-units variant) |
| 8 | minor | Signs per sequence: E6's rank-m mechanism holds per sequence only | fixed (doc) | README, Differences; the stable-rank log shows what survives |
| 9 | minor | `cos_to_autograd` has no estimator or per-site output | fixed | `estimator: identity, h8, cut`; h8 adds per-site credit cosines; four configs |
| 10 | minor | Tables miss E7's no-prior arms, E15's arm (d), E14's top cosine; stable ranks mix Boost updates | fixed | All in the tables; stable rank over Cut updates only |
| 11 | nit | Job count; a stale Differences bullet | fixed | |

### 20261029T125003Z-code-claude.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | E16's n-gram tables at Adam ×1 | fixed | Same as research-claude #2 |
| 2 | major | H8's E7 prior adds an S0 guess the chain lacks | fixed | Same as research-claude #6 |
| 3 | minor | Logging inside H8's timed update | fixed | Same as code-codex #3 |
| 4 | minor | E7's gate prefix matches S0 | fixed | Same as research-claude #6 |
| 5 | minor | E13's n-gram model runs on the CPU in the clock | rejected | It is the method's own cost (the chain's arm pays it too); about a minute per run; E13 is judged on loss at fewer rows |
| 6 | minor | `row_budget` is a dead option | fixed | E13's rule on the chain: `row_budget: ngram` with `budget_theta` (`cut.NGramBudget`); the table learns every update's batch; `cut_budget` checks it |
| 7 | minor | H8 accepts a full-cycle prior and `checkpoint_optimizer` | fixed | Both refused |
| 8 | minor | Dead code | fixed | `after_step` hooks, `FisherFactor.close`, `SequenceGrads.__len__`, the one-element loop |
| 9 | minor | E8's plots mix the two settings | fixed | The H8 arms get `final_h8.png` against `e8.h8_exact_control` |
| nits | nit | Claim docstring, job count, gate docstring, dense `diag_embed` | fixed | The diagonal is added in place |

### 20261029T125003Z-usability-claude.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | The first submission queues nothing runnable (cancelled v1 jobs) | fixed | Submissions give each unchanged job's reason; README: the first v2 submission is `--rerun` (done) |
| 2 | major | Jobs blocked two levels up show "waiting" | fixed | Transitive blocking names the chain |
| 3 | major | The one-GPU order contradicts the README | fixed | One P1 band (research-claude #5) |
| 4 | major | The disk budget does not fit | fixed | Finals in bf16 after scoring (~19 GB); README on the floor and what to free |
| 5 | major | Plots fail under plain python3 | fixed | README uses the venv; a one-line hint without matplotlib |
| 6 | major | Four H6 diagnostics have no reader | fixed | `plots h6` tables |
| 7–14 | minor | Dry run on a missing queue; `--cancel` with `--dry-run`, lists and patterns; arg job ids unchecked; `after_done` hidden; stale help; no one-GPU guidance; gate skips read as errors; jargon | fixed | Gates' ids may sit in `after` or `after_done`; `skipped (gate … says skip)` |
| 15 | minor | `followups/e8` queues the H8 arms with a placeholder assignment | deferred | Their comments say to set E6's winner first; E8 waits on its diagnostic anyway |
| nits | nit | Missing seeds, tracebacks, status lines, layout | fixed | |

### 20261029T141200Z-code-codex.md (re-review of round 7's fixes, commit c9e0ac6)

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | E13's chain budget joins a capture's sequences before scoring them | fixed | Each sequence gets its own last target; `cut_budget` checks a two-sequence capture. No queued job used it |
| 2 | minor | E15's logged readouts stay in the clock | deferred | No E15 result reads its time; the readouts every tenth Boost update are the remaining timed logging |

## Round 8: the first queue's results (147 runs, 2026-10-29)

### 20261029T230000Z-results-claude.md

| # | Severity | Claim | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | major | The H8 validation bar was set at the observed value, and float32 refutes its explanation for x0_lambdas, c_v, c_proj | fixed (doc) | Configs and README now say what float32 showed: the query gap is rounding; c_v and c_proj differ systematically but stay at 0.995+; x0_lambdas 0.980 in both, far from autograd for both estimators, cause open. `h8_validation` now reports the matrices' and the scalars' minima apart |
| 2 | major | Boost does nothing in H8, so E15's verdict cannot transfer | fixed (doc) | README Differences and the reading; E15's lambda follow-ups not run; the no-Boost arm now logs the cut cosine at the record's handoff update |
| 3 | major | README says nothing has run; estimates not replaced | fixed | Status paragraph: the first queue, shared-GPU times as bounds |
| 4 | minor | E16's verdict mixes seed sets | fixed | Seed-matched (0-1): backprop +0.1411, H8 +0.1430, beside the leaderboard's +0.127 |
| 5 | minor | E7's skip is borderline | fixed | Its two arms run without the precondition (6 short runs) |
| 6 | minor | E4's explanation is wrong | fixed (doc) | The multiplier rises in the warmdown tail, where the judge accepts nearly every shrunken step |
| 7 | minor | E10 clip and E11 never acted | fixed | The board reads their logs: E11 `null (the rule never acted)`, E10's note gives clip's cosine and agree's kept share |
| 8 | minor | `cosine_to_identity` under bf16 exceeds 1 | fixed | Computed with autocast off |
| 9 | minor | Shared-GPU wall-clocks unflagged | fixed | `collect.py` flags `shared GPU` from each run's job record |
| 10 | minor | E5's float32 rerun tests the method, not the bf16 readout the plan specifies | deferred | A chain-style bf16 replay (differencing the changes) needs the record's kernels; float32 is necessary, not sufficient, and the README says so |
| 11 | nit | Board's 1e-5, score.json's hash after bf16, E13's reading, identity_match's one update, E15 s1's spike | fixed (partly) | Board text; E13's note in the reading; the rest recorded here |
