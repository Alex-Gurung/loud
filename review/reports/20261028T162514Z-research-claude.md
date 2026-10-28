# research review, round 2: Claude

- source: frozen copy at /var/tmp/loud-review/20261028T162514Z-claude/src (2026-10-28 16:25 UTC)
- reviewer: Claude (loud-reviewer instructions, general-purpose agent), lens research
- saved by the author: the agent's tooling could not write this file. Experiment scripts:
  /var/tmp/loud-review/20261028T162514Z-claude/scratch/research/work/

## Verdict

The H5 port is faithful to PR #26's 8-GPU run with the same seed:
- per-rank seeding of Capture and Boost, Settings, Recipe and GPTConfig as `main()` builds them;
- the warmdown, gradient averaging and loss;
- the warmup and rebuild, and the data order at seed 0.

On CPU, the init model's loss on the first batch of loud's order is 8.31902, against the published step-1 value of
8.3192. The verification targets trace to their sources, and the round-1 major fixes hold.

The research problem is in E6. In this copy, `row_offset_per_step` does nothing: PR #26 redraws the basis signs at
every probe, so the K windows never form one cycle. The only CPU test of the windows resets the generator before each
window, which training never does. The live tree has since fixed this. My CPU measurement suggests the fix will change
little in what reaches momentum, so E6's numbers say little about "stitching" unless the two are compared. The drift
check and the multi-rank test coverage are weaker than the docs say.

### 1. [major] `row_offset_per_step` is a no-op: every probe draws a fresh signed basis, so consecutive windows never stitch into one cycle, and the test behind the README's claim resets the generator
- where:
  - `loud/identity.py:37-44` and `:106`;
  - `external/noise-pr26/train/speedrun/run.py:268-272` and `:284-285`: `basis()` = Hadamard × `signs()`, drawn from the
    rank's generator on every `probe()` call;
  - `tests/method_checks.py:115-124` (the `set_state` at line 121);
  - `README.md:199-201`;
  - `configs/e6/identity_k4.yaml`, `configs/e6/identity_k16.yaml`.
- evidence:
  - **Algebra.** Take a window R of r = 512/K rows of the Sylvester matrix, with column signs D. It credits
    (1/r)·D·M_R·D·g, where M_R = r·(c_w c_wᵀ ⊗ I_r) and c_w is a ±1 character of the top log₂K index bits. Fresh
    random D absorbs c_w, so every offset has the same error distribution: coordinate j aliases with the K−1
    coordinates sharing j mod r.
  - **Per-window error by offset is flat** (`window_stats.py`, 38M model at init, real rows, 64 tokens, 8 sign draws).
    Relative error against the same-sign full cycle:
    - K=4: 1.729 to 1.737 at every offset (√3 = 1.732);
    - K=16: 3.81 to 3.92 at every offset (√15 = 3.87).
  - **Averaging the K windows of one update:**

    | Signs | K=4 | K=16 |
    | --- | --- | --- |
    | Same signs | 2.3e-7 | 2.6e-7 |
    | Fresh signs, as training draws them | 0.873 | 0.967 |

    Fresh signs land at √((K−1)/K), the value for independent draws. So E6 as queued tests "a random 1/K subset of
    signed rows per update". It does not test "the cycle spread over K updates", which is what the plan,
    `identity.py:10-14` and the configs describe. `rows: 128` with `row_offset_per_step: false` is the same experiment.
  - **The CPU test.** `identity_window` passes only because of the generator reset (`window_test_fresh.py`):
    - with the reset: error 2.2e-7, assertion passes;
    - with signs drawn as in training: error 0.86, assertion fails.
  - **What the stitch would buy.** `stitch.py` uses the tiny preset, 64 tokens, the real IdentityPath, weights held
    after 20 updates, and K consecutive real batches. It compares the K-update average of windowed estimates with the
    full-cycle average, over block matrices:

    | Signs | K=4 | K=16 |
    | --- | --- | --- |
    | Fresh | 0.952 | 1.528 |
    | Held for the cycle (what the live tree now does) | 0.964 | 1.475 |

    Holding the signs barely helps. Each rank reads a different sequence at each update, and a held basis cancels only
    the aliasing of what those K sequences share. Per-sequence variation dominates: the K full-cycle batch estimates
    already spread 1.07 (K=4) and 1.94 (K=16) around their mean. This is one small-model measurement with 3 draws, so
    treat the size as indicative.
- fix:
  - The live change (one signed basis per cycle `step // K`, the generator still advancing every probe) is the right
    mechanism. Replace `identity_window` with a test that drives K consecutive `IdentityPath.update` windows the way
    training draws them.
  - Because the stitch appears to buy little, consider one control cell at K=4 with `row_offset_per_step: false`, so
    E6 can say whether stitching mattered. Until then, describe E6's result as "1/K rows per update".
  - Note that in the live indexing the first cycle (steps 1 to K−1) is partial: it has no window 0.

### 2. [minor] The drift check measures a sequence the momentum already holds, against fast and slow separately rather than the blend ANVIL II steps with
- where: `loud/identity.py:125`, `:128-145`; `external/noise-pr26/train/speedrun/run.py:988-1002`
- evidence:
  - `diagnose` measures `self.last = batch[:1]`, the first sequence of the update just applied. The fast momentum
    therefore contains (1−β)/8 of that sequence's own estimate, about 0.019 at β 0.85. That inflates the cosine by
    roughly 0.019·‖g_seq‖/‖m‖, which is not tracking.
  - ANVIL steps along `weight·fast + (1−weight)·slow`: weight 1 with the rail β before the engage step (261 of 606),
    then 0.4385 and 0.85. The blend's cosine cannot be recovered from the two logged cosines.
  - The cosine is over all block matrices concatenated, so the largest-norm matrices dominate it.
  - On CPU (tiny preset, K=4, 120 updates, `drift_self.py`), every cosine fell between −0.08 and 0.10:
    - in-momentum sequence: mean 0.022 (fast);
    - the next update's unseen sequence: −0.001.

    The self-overlap effect is not resolved at this scale (unverified). A one-sequence cosine is noise-dominated, so
    the plan's follow-up threshold of 0.9 is unreachable, though the comparison across K is still fair.
- fix:
  - Measure the next update's first sequence before its update, or a held-out sequence.
  - Log the cosine with ANVIL's blended momentum, and optionally per matrix type.
  - Note in the E6 configs that the absolute values are not comparable with 0.9.

### 3. [minor] Nothing tests the multi-rank emulation, and `verify.identity_match` cannot see per-rank draws
- where: `tests/method_checks.py:127-143`; `configs/verify/identity_match.yaml`; `README.md:27`, `:196`
- evidence:
  - `identity_step` runs `batch=1`, so it checks one virtual GPU only. The ranks above 0 (seeds `seed+1000·r`) and the
    summation over ranks are never compared with the script's own `Boost.step` across ranks. They are correct by
    inspection: same constructor arguments as `main()`, and the summation equals `all_reduce` then `/ sequences`.
  - The GPU compare cannot catch a seeding error. Two sign draws move the full-cycle credit by only 0.96% relative
    (`window_stats.py`). After the live fix, the windowed path depends on per-rank bases, and no published log or test
    covers them.
  - The README's "one update equals PR #26's own" holds only at batch 1.
- fix: Add a two-process CPU test with gloo. Run the unmodified `FastRelay.step` at world 2 against
  `IdentityPath.update` at batch 2, both with rows below the width and the full cycle, and require identical
  parameters. Otherwise, state "batch 1" in the README.

### 4. [minor] Each per-step comparison waits for the final-loss gate it is meant to explain
- where:
  - `configs/verify/identity_match.yaml:8`;
  - `configs/verify/bp_b8_ledger.yaml:9`;
  - `configs/verify/bp_b8_match.yaml:9`;
  - `loud/jobqueue.py:160-162`.
- evidence:
  - `dependency_state` blocks a job whose dependency "failed its check". If `verify.identity` misses 4.1772 ± 0.01, or
    `bp_b8_script` misses 3.7083 ± 0.01 (for example under torch 2.13), the compare never runs.
  - Yet the compare is what separates a port error from an environment difference. `bp_b8_match` tests the port alone,
    since both sides run on one GPU with the same torch. The plan's "stop and find out why" needs it most exactly then.
- fix: Let a compare task depend on its runs being done rather than passed (for example `after_done`). Alternatively,
  have the compare read the runs regardless and keep E1 gated on both.

### 5. [minor] `e1.identity` keeps no trajectory means, and no identity or E6 run keeps H4 checkpoints; the ledger says otherwise
- where: `configs/e1/identity.yaml:9-13`; `review/LEDGER.md`, research-codex #3; the plan, lines 71 and 126;
  `README.md:190`
- evidence:
  - The ledger's fix note says "`means: true` in every E1 cell". The E1 identity cell has neither `means` nor
    `checkpoints`.
  - The plan lists the H4 checkpoints among the logs that must exist for every run. The README says only that they are
    opt-in, with no reason (presumably disk).
- fix: Add `means: true` to `e1/identity.yaml`, or correct the ledger. State the disk reason for opt-in checkpoints in
  the README.

### 6. [nit] `e6.identity_k1.s0` repeats `e1.identity.s0`
- where: `configs/e6/identity_k1.yaml`, `configs/e1/identity.yaml`
- evidence: The two configs have the same seed, held-out split, batch and full cycle. `drift_every` uses a separate
  estimator and never changes training, so the second run repeats about 6 GPU-minutes.
- fix: Reuse one run as the other's seed-0 cell, or accept the repeat as a determinism check and say so.

### 7. [nit] The smoke test never runs the identity path
- where: `configs/verify/smoke.yaml`, `configs/verify/identity.yaml:8`
- evidence: `verify.smoke` runs only the proxy. The identity path's first GPU contact is therefore the full
  `verify.identity` run: PR #26's compiled kernels under torch 2.13, flex attention, eight Captures, and the windows
  plus drift path. A crash there is retried 3 times.
- fix: Add a short smoke update of the identity path (`stop_after: 5`, `rows: 128`, `drift_every: 1`), and make
  `verify.identity` and E6 wait for it.

## Checked and fine

- **Seeding.**
  - `Estimator(model, settings, seed, rank, batch)` reaches `Boost.__init__` with split 1, so each rank gets Capture
    `seed+1000r(+3/+5)` and Boost `seed+17+1000r`. That is exactly `main()`'s
    `FastRelay(model, settings, args.seed, rank, world)`.
  - The drift estimator's offset of 104729 cannot collide with any training stream.
- **Settings, Recipe and model.** `fc_k` 256, `route_k` 512, `fast_mlp_mode` all, `no_chain`, `skip_mid`, no value path,
  `anvil_lr` 0.023 and `sequence_len` 2049 all match the published `log/config.json` field for field.
- **Schedule and averaging.**
  - `set_rate` is PR #26's: one-based steps, min(1, left/0.3).
  - ANVIL is built with the schedule length.
  - Each sequence's gradient is divided by its valid tokens in float32. The gradients are summed, divided by the batch
    and cast to each parameter's dtype, then `zero_grad` and `step` run, as in `Boost.step`.
  - The logged loss is the mean over ranks, as `average_ranks`.
- **Warmup.** One update on random tokens, then the hooks are closed and the model (`manual_seed`), optimizer and
  estimators are rebuilt. The noise seed (0 rather than `args.seed`) is irrelevant after the rebuild. The timer covers
  the batch transfer and excludes diagnostics, held-out scoring and drift.
- **Data.**
  - The order is `randperm(4880, seed 1)`. The records use the constant `SEED + 1` (boost-cut `run.py:1096`, archive
    `val/run.py:2449`, `backprop_preset.py:56`), so the README's claim holds.
  - PR #26 alone uses `args.seed + 1`, and the README flags it.
  - `train.pt[:4880]` hashes to PR #26's `train.pt`.
  - Step-1 loss on CPU is 8.31902 against the published 8.3192.
- **Window mechanics.**
  - `share()` is overridden only inside `probe()`, so the embedding's norm cycle keeps the full basis.
  - The generator is consumed identically for every K, and the scaling is width/rows.
  - Each update is unbiased over the column signs: the error is zero-mean with relative size √(K−1).
  - `rows = 512` runs PR #26's `probe()` itself.
- **E6 configs.**
  - K = 1, 4, 16 maps to rows None, 128 and 32, with seeds 0–2, held-out 32, batch 8 and `drift_every` 50.
  - The success and kill criteria are as in the plan, and order and held-out split are shared across cells.
- **Verification targets.**
  - bp38_s0: 3.70835 and 3.70142, ± 0.01.
  - PR #26: 4.177219…, ± 0.01. `identity_match` covers 20 updates at 0.005; the 4-decimal rounding of the log is at
    most 5e-5.
  - `score_record`: from boost-cut `ledger/verification.md` and `runs.json` (bc38f_s0).
- **Log parsing.**
  - `compare` skips the ledger log's command, `[modal_app]` and result lines, and the duplicated run collapses to
    identical values.
  - The reference task copies the script's `metrics.jsonl` into the run, and boost-cut's `load_split` takes the first
    4,880 rows of the 30M `train.pt`.
  - All 28 jobs validate in a dry-run submission against a scratch queue.
- **Round-1 major findings marked fixed.**
  - Gates are 0.01, with test loss.
  - The compare tasks exist and are in every E1 backprop cell's `after`.
  - `collect.counts` drops failed checks.
  - `bp_b4` states its reason, and `bp_b4_pr40` exists.
  - Every E1 backprop cell has `means`, and `bp_b4` saves them.
  - H3 scores in one forward.
