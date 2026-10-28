# research review: Claude

- source: frozen copy at /var/tmp/loud-review/20261028T150535Z-claude/src (git 3bb0197 plus the working tree, 2026-10-28 15:05 UTC)
- reviewer: Claude (loud-reviewer instructions, general-purpose agent), lens research

## Verdict
The proxy behaves like `backprop_preset.py` and the trainer module it imports. It calls the same `make_model`, `make_optimizer` and `Recipe`. The warmdown is numerically identical, the data order is equal tensor for tensor, the gradient is the batch mean with the 8-GPU cast order, and the untimed warmup is the same. The faults are in the checks and configs that decide whether E1 may start: the verification tolerance is twice the plan's stop rule, the README's step-for-step check is never run, and the batch-4 baseline uses a recipe that has no published number and no stated reason. Fix these before the E1 numbers become the baselines for every later experiment.

### 1. [major] The verification gates accept errors up to 0.02; the plan stops at seed noise (0.01)
- where: `configs/verify/bp_b8.yaml:13-14`, `configs/verify/bp_b8_onefwd.yaml:14-15`, `configs/verify/bp_b8_script.yaml:14-15`; the plan, line 59; E1 waits on this check (`configs/e1/bp_b8.yaml:8`, `configs/e1/bp_b4.yaml:8`)
- evidence:
  - The plan says: "If any of the three disagrees by more than seed noise (0.01), stop." All three verify configs use `val_loss: [3.708, 0.02]`.
  - boost-cut's ledger (`external/boost-cut/ledger/runs.json`) has two seed-0 runs of this exact script: `bp38_s0` at 3.70835 and `bp38_curve` at 3.70846. A correct port should land within about 1e-3.
  - 0.02 is about 2.2 seed SDs (seeds 0–2: 3.7083, 3.7225, 3.7062, sd 0.0089). It equals the success margin of E2, E3 and E9. A port error of that size would pass, unblock E1, and get built into every baseline.
  - Published numbers that no check uses:
    - the test loss (3.70142);
    - train loss every 61 steps (`external/boost-cut/ledger/logs/bp38_s0.log`).
- fix:
  - Set `val_loss: [3.7083, 0.01]` (or 0.005) and add `test_loss: [3.7014, 0.01]`.
  - Compare `verify.bp_b8`'s train loss at steps 61, 122, …, 610 against `bp38_s0.log`.
  - Optionally add `eval_every: 61` to match that log's untimed val curve.

### 2. [major] "`bp_b8_onefwd` must match `bp_b8_script` step for step": no code compares them, and nothing waits on either
- where: `README.md:109-110`, `configs/verify/bp_b8_script.yaml:2-3`, `configs/verify/bp_b8_onefwd.yaml:1-2`
- evidence:
  - Searching for a comparator finds only these comments.
  - `jobqueue.dependency_state` reads only a job's own `check`. Each of the two jobs checks only its final val against 3.708 ± 0.02.
  - Neither job is in any `after:` list, so E1 can finish before either has run. The check that would catch a port bug the final loss hides cannot fail.
  - The pairing is sound: my CPU run (below) shows `micro_batch: 8` gives exactly the gradient of one direct forward of all 8 sequences.
- fix:
  - Add a check that diffs the two `metrics.jsonl` files per step. Use a tolerance of about 1e-3, since the ledger shows seed-0 reruns are not bitwise identical.
  - Add both jobs to E1's `after:`.
  - The same pair tells you whether `micro_batch: 8` (one forward, likely faster on one GPU) can replace the default `micro_batch: 1` for experiments.

### 3. [major] The E1 batch-4 baseline uses boost-cut's batch-8 recipe; no published number exists for it and no reason is given
- where: `configs/e1/bp_b4.yaml:10-13`; `README.md:103-106`; the plan, line 94
- evidence:
  - `bp_b4` runs `recipe: boostcut` at batch 4: lr 0.003, betas (0.7, 0.95), warmdown 0.3 to 0, ANVIL 0.023, 1,212 steps at the same per-step rates.
  - The record that defines "the record's own batch" (PR #40, `external/noise/archive/10M/train/val/run.py:65-85`, `log/config.json`) uses:
    - lr 0.0021, betas (0.5, 0.98);
    - warmdown 0.4 ending at 0.3 of each rate;
    - LM-head init std 0.02, ANVIL 0.026, fast weight 1.0.
  - The ledger has backprop runs only at batch 8, so the plan's "expect about 3.71 ± 0.01" has nothing to anchor it.
  - This cell supplies the diagnostic checkpoints for E5, E7 and E8, and it is the "batch-4 baseline" that E3 and E9 must beat by 0.02. If it is under-tuned, those wins are inflated.
  - The README only says the recipe "is a config choice". It does not say which recipe E1 uses or why.
- fix: Decide, and write the reason in the config. Either add an `e1.bp_b4_pr40` cell (`recipe: pr40`, batch 4, held-out 32: exactly PR #40's recipe, data and schedule under backprop), or state why boost-cut's rates at batch 4 is the intended baseline.

### 4. [major] `collect.py` says runs that fail their check are listed apart; they enter the means
- where: `loud/collect.py:8` (the claim), `loud/collect.py:67` and `loud/collect.py:80` (what happens)
- evidence: I made a fake run root in scratch with one seed passing (3.70) and one failing (3.80). The table showed `n=2, 3.7500 ± 0.0707` with only a "check failed" flag. The "not in the table" list covers only incomplete or unscored runs.
- fix: Exclude `check == 'fail'` runs from the cell and list them with the incomplete ones, or correct the docstring.

### 5. [minor] The H2 noise-scale log (`b_simple`) is biased and saturates at the batch size
- where: `loud/pergrad.py:5-6`, `loud/pergrad.py:56-61`
- evidence:
  - The denominator is |ḡ|² over B sequences, and E|ḡ|² = |G|² + tr(Σ)/B. So the logged value is about B·B_true/(B + B_true).
  - A simulation using the same formula, logged against true B_simple:

    | B | true 8 | true 64 | true 512 | true 4096 |
    |---|---|---|---|---|
    | 8 | 4.0 | 7.1 | 7.9 | 8.0 |
    | 4 | 2.7 | 3.8 | 4.0 | 4.0 |

  - On CPU, step 1 already logs values of 0.7 to 4.0.
- fix: Use the unbiased |G|² estimate, `mean_sq - trace_cov / B`, as the denominator (McCandlish et al.). Smooth numerator and denominator separately before taking the ratio, or log only the two sums.

### 6. [minor] The H4 trajectory means are off by default and thrown away unless saved; the README calls H4 done
- where: `loud/options.py:21-22`, `loud/proxy.py:222-225`, `loud/checkpoints.py:94` (`state_dicts`, never called)
- evidence:
  - The plan (line 71) keeps the nine means "of every run … so merge experiments reuse them".
  - Here they are opt-in. With `means: true` but no `save_means`, they are computed and then discarded: only a count reaches `result.json`. No E1 config enables them.
  - They cost about 1.4 GB of GPU memory, not disk, so the disk reason given for checkpoints does not apply.
- fix: Enable `means` by default, or flag the difference in the README status table. Remove `state_dicts` or use it, e.g. to score each mean on H3 at the end and log the scores.

### 7. [minor] `sequences: 1M` cannot reproduce the published 10M log, which the plan's H5 check needs
- where: `loud/data.py:17`, `loud/data.py:59-67`; the plan, line 76
- evidence:
  - `1M` permutes rows 0–487 with `randperm(488)`, so the first batch is rows [181, 418, 74, 226]. The published 10M log's first batch is rows [4085, 2118, 1064, 3720].
  - The 122-step schedule also differs from the 1,212-step one. "Per-step losses match the published metrics.jsonl for the same steps" is therefore impossible with this slice.
- fix: For log matching, document `sequences: 10M` with `stop_after: N`, or add a "prefix of the 10M order" option before H5 relies on it.

### 8. [minor] Differences in GPU type and code version between compared variants go unflagged
- where: `loud/collect.py:75-80`; `worker/gpu_worker_any.yaml:37`
- evidence:
  - `mixed GPUs` and `mixed code` are checked only inside one variant.
  - Jobs carry no GPU requirement, so an H100-or-H200 worker can pick up an E1 cell while its comparison cell runs on an H100.
  - Every later variant will run newer code than the E1 baselines. Plan items such as E5 and E7 ("equal loss at lower time") compare wall-clocks across variants.
- fix: Flag GPU-type and code-digest differences across the cells of one experiment. Optionally add a `gpu:` field to jobs that the worker honours.

### 9. [minor] The H4 checkpoint set mixes precisions: 0–50% in bf16, 100% in fp32
- where: `loud/checkpoints.py:25`, `loud/checkpoints.py:61-68`
- evidence:
  - `ckpt_100.pt` is a link to the fp32 `final.pt`; the other four store 2-D weights in bf16.
  - Under autocast the forward is identical. H6's Jacobian SVDs and cosines (E5, E7) may run in fp32, though. Then the 20%-versus-100% comparison mixes precision with training progress.
- fix: Also save `ckpt_100.pt` with the intermediate dtype, or use `checkpoint_dtype: float32` for the `bp_b4` diagnostic runs.

### 10. [minor] H7 and §5 pieces are missing and not flagged in the README
- where: `loud/collect.py:42` and `loud/collect.py:64-85`; `README.md` status table
- evidence:
  - `heldout_min` is computed but not shown in the table. The plan's "held-out curve summary" column (line 88) shows only the final held-out loss.
  - The plan's "plot script per experiment" (line 126) does not exist, and the status table does not say so.
- fix: Add the held-out curve summary to the table, and list plot scripts as not started.

### 11. [minor] No test runs `loud/proxy.py`, despite "CPU-tested on a tiny model"
- where: `README.md:13`; `tests/`
- evidence:
  - No test imports `proxy`.
  - I ran `proxy.run` on CPU (preset 2m, dense attention patched in, `scratch/work/cpu_proxy.py`). Step 1, the checkpoints, held-out scoring and `grad_noise` all work.
  - `scratch/work/mb_equiv.py` shows:
    - `micro_batch: 8` gives exactly the gradient of one direct forward;
    - `micro_batch: 1` gives the same loss and a 1.00002× gradient norm (a mean, not a sum);
    - the two differ by 0.24% relative from bf16 rounding.
- fix: Turn `mb_equiv.py` into a test.

### 12. [nit] The 3.7081 seed-0 baseline cited for PR #20 is unverifiable here
- where: `loud/refs.py:43`, `README.md:105`, `configs/verify/bp_b8.yaml:1`
- evidence: The only number in the pinned repos is the ledger's run of this exact script: `bp38_s0` at 3.70835, test 3.70142, with a log.
- fix: Cite `ledger/runs.json` `bp38_s0` as the target.

### 13. [nit] `ckpt_100.pt` is created for incomplete runs
- where: `loud/checkpoints.py:64-69`
- evidence: In a `stop_after` run (e.g. `verify.smoke`, stopped at step 20), `ckpt_100.pt` links to `final.pt` and is recorded as fraction 1.0.
- fix: Skip the link when `last < steps`.

### 14. [nit] The `pr40` trainer imports an unpinned file
- where: `external.json`
- evidence: The PR #40 script imports `archive/10M/utils/eval.py`, which is not pinned. The proxy does not use it.
- fix: Pin it.

### 15. [nit] Not every `result.json` val loss comes from `utils/eval.py` in a fresh process
- where: `README.md:99`
- evidence: The `reference` task's `val_loss` is the script's own in-process `bc.evaluate`, not `utils/eval.py` in a fresh process. It is the same computation, but the README states the fresh-process version for every run.
- fix: Reword the README line.

### 16. [nit] The E1 configs carry no `expect`
- where: `configs/e1/bp_b8.yaml`, `configs/e1/bp_b4.yaml`
- evidence: The plan's "about 3.71 ± 0.01" is not recorded anywhere, so an off cell is never flagged.
- fix: Add the expectation.

## Checked and fine
- **Model, init and optimizer.** The `boostcut` recipe calls the pinned `make_model`, `make_optimizer`, `Recipe` and `POLAR_MARGIN` (1.1) verbatim:
  - ANVIL II on `transformer.h.*` 2-D weights except `ve_gate`;
  - AdamW at ×300 (wte, value_embeds), ×30 (lambdas) and ×1 otherwise;
  - betas (0.7, 0.95), eps 1e-8, no weight decay.
- **The `pr40` recipe.** It uses PR #40's `Settings`, `make_model(projection_std, head_std)`, `BatchedAnvil` and warmdown with a final rate of 0.3. Its single-phase backprop matches PR #40's phase handoff, since both phases use batch 4.
- **Warmdown.** `set_rate` equals backprop_preset's formula at every step for 610, 606 and 1,212 steps (maximum difference 0). Steps are one-based in both. ANVIL's internal schedule is scaled to the real step count.
- **Gradients.**
  - Each micro-batch's gradient is accumulated in fp32, divided by n, then cast to the parameter's dtype, as the 8-GPU all-reduce does.
  - The loss is the batch mean (forward default `'mean'`), which answers the plan's open question correctly.
  - Autocast is bf16, and float32 matmul precision is `'high'`.
- **Data.**
  - The order with `heldout: 0` equals backprop_preset's `load_split(train.pt, 4880)[randperm(4880, seed 1)]`, checked tensor for tensor.
  - Holding out 32 drops only the tail of the order, as PR #40 does.
  - The seed changes only the init; the warmup noise always uses seed 0.
- **Pins.** `refs.check_all` passes on all pinned files, downloads and data hashes. `test.pt` is cb7a29f0 and is the Dust test documents, which `train.pt` excludes.
- **Timer.**
  - The clock starts at the first update.
  - Warmup on random tokens is followed by a fresh model and optimizer.
  - Held-out scoring, checkpoints and `val` monitoring are paused, and `eval_every` only logs.
  - No decision reads `val.pt`, the run is one epoch, and the final weights are what gets scored.
- **Scoring.**
  - `loud/score.py` runs `utils/eval.py`'s own `evaluate` in a fresh process with a strict load.
  - The test loop is identical to it.
  - The held-out scorer matches PR #40's `holdout_loss` (batches of 8, summed loss over predictions).
- **Trajectory means.** The rule (from 60% by first sequence, every 5 updates and at the last update, fp32, the nine decays) matches PR #40.
- **Configs.** The queued snapshot `674a06a81d64f88f` is byte-identical to this review copy. Seeds and held-out settings are consistent across the E1 cells. The `score_record` targets match the ledger's rescore (3.944812 and 3.936956).
