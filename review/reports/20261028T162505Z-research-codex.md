# research review: Codex

- source: frozen copy `d26f79e0562383f4` at 20261028T162505Z (git 3bb0197 plus the working tree)
- reviewer: Codex 0.164.0, model gpt-6.1-sol, session 01a18354-5aaa-7e03-a4f6-29b3c4ef1587
- focus: Round 2. Round 1's findings and fixes are in review/LEDGER.md. First check that each blocker and major marked fixed really is. Then review the new H5 identity-path port (loud/identity.py on the shared loop loud/train.py, configs/verify/identity.yaml, configs/verify/identity_match.yaml, configs/e1/identity.yaml, configs/e6/) against PR #26's script, external/noise-pr26/train/speedrun/run.py (its main() and Boost.step). Is it faithful to the 8-GPU run with the same seed (seeding per rank, gradient averaging, settings, schedule, data order)? Is E6's rows / row_offset_per_step what the plan's E6 asks for, and unbiased? Does the drift check measure what E6 asks? Also check the new verification gates and targets in configs/verify/ against their sources (the boost-cut ledger, PR #26's log).

The E6 implementation does not spread a complementary probe cycle across updates: it redraws the signed basis each time, changing the experiment being tested. The Round 1 regression suite passes, but verification coverage, timing, and checkpoint retention still need changes. CUDA behavior and published-loss agreement remain unverified without a GPU.

### 1. [major] E6 redraws directions instead of spreading one cycle across K updates
- where: `loud/identity.py:34`, `loud/identity.py:106`
- evidence: The wrapper rotates row indices but calls PR #26’s `probe()`, whose `basis()` generates fresh column signs on every call. Consequently, successive windows are not complementary directions from one signed basis. A CPU calculation using the reference Hadamard matrix and a fixed linear signal gave relative reconstruction error **2.8e-7** after four windows with fixed signs, versus **0.814** with fresh signs. Individual estimates remain unbiased, but rotating indices now tests fresh partial projections rather than the plan’s cycle spread over time. `tests/method_checks.py:122` conceals this distinction by restoring the generator state before each window.
- fix: Retain the top-probe basis for each K-update cycle, advancing its window between updates. Keep the full-row reference path unchanged, and test successive windows without manually rewinding the generator.

### 2. [minor] The match gates can pass without comparing the requested updates
- where: `loud/compare.py:49`
- evidence: Comparison uses only the intersection of logged steps and never checks its expected size. I exercised the identity-match configuration against the pinned PR #26 log using a scratch run containing only its first matching entry and matching final result. It returned `steps_a: 1`, `steps_b: 610`, `steps_compared: 1`, and **`check.status: pass`**, despite requesting 20 updates. The ledger comparison likewise permits missing reference steps.
- fix: Require every requested reference step to exist on the other side. For `first: 20`, require steps 1–20; for the sparse ledger comparison, require all reference log steps.

### 3. [minor] GPU trajectory-mean work can leak into reported training time
- where: `loud/train.py:110`, `loud/train.py:117`
- evidence: `means.observe()` launches CUDA clones or `lerp_` operations inside the excluded interval, but that interval ends without synchronizing. On updates without another synchronizing diagnostic or checkpoint, outstanding mean work runs before the next update’s synchronization and enters its `step_seconds`. E1 backprop enables nine means while identity does not, introducing variant-dependent timing contamination. This finding follows from the stream ordering in the source; its magnitude was not measured on CUDA.
- fix: Synchronize after excluded GPU work and before ending the pause, or use CUDA events with explicit boundaries for the timed updates.

### 4. [minor] Identity and E6 configs omit the required H4 artifacts
- where: `loud/options.py:55`, `configs/e1/identity.yaml:11`, `configs/e6/identity_k4.yaml:11`
- evidence: Checkpoints and means default to false. The identity baseline and all three E6 variants leave both disabled, so they retain only final weights. The plan’s H4 and reporting requirements call for fraction checkpoints and trajectory means for every run. README documents checkpoints as opt-in but gives no reason for this departure.
- fix: Enable H4 checkpoints and means in the affected experiment configs; save the means where later reuse is required.

## Checked and fine

- All **28 CPU tests passed**, including Round 1 regression tests for attempt isolation, interrupted transitions, failed-check exclusion, config validation, resubmission, and proxy execution.
- PR #26 model construction, initialization, optimizer groups, rates, betas, settings, per-rank generator seeds, gradient normalization, and dtype casts match the port’s source.
- Seed-0 data order and one-based warmdown match PR #26. The different order policy for later seeds is documented and consistent across E6 variants.
- Partial-row scaling preserves unbiasedness in expectation. The defect above concerns temporal coverage.
- The drift diagnostic uses a separate estimator and RNG stream, takes a full-row measurement, and compares block gradients with ANVIL’s fast and slow momenta.
- Verification targets match the pinned boost-cut ledger and PR #26 result.
- Held-out rows are excluded from updates; queued configs make no validation-based training or checkpoint-selection decision.
- Official scoring uses the reference evaluator in a fresh process; test scoring follows its reduction, batching, and autocast behavior.