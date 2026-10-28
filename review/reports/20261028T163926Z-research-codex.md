# research review: Codex

- source: frozen copy `930d0742b12acfb4` at 20261028T163926Z (git 3bb0197 plus the working tree)
- reviewer: Codex 0.164.0, model gpt-6.1-sol, session 01a18361-7c34-7630-89bf-ffebf800487e
- focus: Round 3: the new H6 diagnostic toolkit, loud/diagnostics.py (task 'diagnose', options in loud/options.py, configs in configs/h6/, CPU check 'diagnostics' in tests/method_checks.py). Check each of the five tools against the plan's H6 items 1-5 and against what E5, E7 and E8 need from them (their decision rules). Check the sites (raw embedding; each block's stream after attention, before the MLP: see utils/model.py in external/noise-pr26), the identity-path credit guess against PR #26's FastRelay.estimate in external/noise-pr26/train/speedrun/run.py, the frozen-attention per-token map in token_map_svd, the finite-difference curvature and noise in direction_stats (the plan says autograd Hessian-vector product: is the substitution sound?), and the linearity measure. Also check whether the measurements are statistically meaningful at one sequence, and whether the stated deviations are documented.

The toolkit does not yet support the planned E5 and E8 decisions: E5 lacks the recovered-credit measurement, and E8 measures different directions from those specified. The supplied CPU check passes, but it does not validate curvature accuracy or probe-size calibration. The reference-site definitions and identity-path recurrence match the pinned model and estimator; CUDA behavior remains untested.

### 1. [major] E5 cannot make its implementation decision from the diagnostic outputs
- where: `loud/diagnostics.py:181`
- evidence: H6 item 3 requests Jacobians and singular values; E5 additionally requires recovered-credit cosines and a 9 × 2 table at the 20% and 100% checkpoints. `token_map_svd` computes Jacobians, discards them, and returns only singular values and conditioning summaries. None of the five tools performs the ridge solve, measures recovered-credit cosine, or produces the ≥0.95 gate. Singular values alone cannot determine recovery accuracy.
- fix: Add the offline recovery diagnostic using measured embedding-probe changes and loss changes, ridge toward the identity guess, and compare recovered credit with autograd at every site. Emit the required table, including the embedding site.

### 2. [major] E8 measures fresh-gradient directions instead of pending-update directions
- where: `loud/diagnostics.py:211`
- evidence: The code averages gradients from eight held-out sequences and takes their SVD. E8 specifies the top 16 singular directions of the pending update, before ANVIL’s orthogonalization. The pinned boost-cut optimizer constructs that input from the current gradient and its fast/slow momentum state; its `Anvil.step` uses `g.lerp(weight * fast + (1 - weight) * slow, beta)`. Weights alone cannot reconstruct it. The module docstring acknowledges the substitution because H4 keeps no optimizer state, but README still says H6 is “not started” and explains neither this deviation nor its implications. Consequently, the resulting h²/c distribution does not characterize the directions E8 intends to damp.
- fix: Save the pending pre-orthogonalization updates or their top singular directions alongside H4 checkpoints, and let `direction_stats` accept those directions. Label fresh-gradient measurements separately if retained.

### 3. [major] The linearity statistic cannot reliably calibrate the intended per-token probes
- where: `loud/diagnostics.py:240`
- evidence: Each draw perturbs every token, runs the full attention graph, and reduces cross-entropy with `loss_reduction='sum'` before computing the ratio. This measures aggregate sequence response rather than the per-token response used by the frozen-attention diagnostic and Cut probes. Linear terms can cancel: two token responses `+aσ + bσ²` and `−aσ + bσ²` yield a zero aggregate denominator despite finite, small per-token nonlinearity. Four draws on one sequence do not resolve this aggregation problem. All evaluations also use float32; casting the reduced scalar to double cannot recover lost precision, and the measurement does not test the bf16 rounding explicitly identified by the plan as a probe-size constraint.
- fix: Preserve token losses and report per-token linearity distributions under the intended frozen-attention probe path. Evaluate the actual bf16 probe path alongside the float32 reference, and retain the signed loss changes so cancellation and rounding can be inspected.

### 4. [minor] Finite-difference curvature replaces HVP without an accuracy check
- where: `loud/diagnostics.py:216`
- evidence: H6 item 4 and E8 request autograd Hessian-vector products. The implementation instead uses a central gradient difference with `e = 1e-3 * max(||W||, 1)`, without step convergence checks or an error estimate. The substitution is mathematically consistent as `e → 0`, but its finite-step accuracy is not established by `tests/method_checks.py:204`, which checks only direction count and nonnegative noise. In a CPU model with nonzero writers, reducing the step tenfold changed the seven measured curvatures by less than 0.7%; that supports this small case, not trained full-size checkpoints.
- fix: Use HVP where supported, or validate the fallback against HVP on a CPU model and check step convergence on each measured direction. Document the backend limitation motivating the substitution and record the effective step and uncertainty.

### 5. [minor] A one-sequence curvature request completes with undefined noise
- where: `loud/options.py:210`
- evidence: `diagnose_options` accepts `tool='direction_stats', batch=1`. I exercised that setting on a small CPU model: the unbiased `.var()` at `loud/diagnostics.py:229` warned about insufficient degrees of freedom and returned `noise: nan`, with `h2_over_c: None`. The task otherwise writes its results and reports `complete=True`; Python’s JSON encoder emits the nonstandard `NaN` value. Even the default eight-sequence variance is a noisy estimate: under independent Gaussian projections, its relative standard deviation is approximately `sqrt(2/7) = 53%`, before squaring uncertainty into downstream decisions.
- fix: Require at least two sequences for `direction_stats`, reject nonfinite results, and report sample count and uncertainty. Use repeated held-out batches before treating h²/c differences as a decision signal.

## Checked and fine

- The supplied `diagnostics` CPU check passed with its documented dense-attention patch.
- Raw embedding hooks precede RMS normalization. Attention-output hooks receive the same credit as the post-attention residual stream; `stream_values` correctly includes the block’s residual mixing.
- Mid-site identity guesses follow PR #26’s residual-scalar recurrence. The embedding guess captures `fast_norm_cycle`, including the reference’s embedding-skip contributions.
- The frozen-attention token map includes embedding normalization, residual and embedding-skip mixing, and intervening MLPs.
- `cos_to_autograd` compares summed-loss float32 reference gradients against the pinned estimator and reports matrix-type, block, and parameter-group cosines.
- Queued H6 configs consistently use seed-0 E1 batch-4 checkpoints and the same held-out order. Credit dumps cover all five fractions; E5 tools cover 20% and 100%.
- The queued checkpoints exclude the diagnostic rows from updates. Diagnostics read no `val.pt`, perform no training updates, and choose no validation checkpoint.
- One-sequence credit and conditioning summaries follow the plan’s exploratory specification; their token quantiles do not establish uncertainty across sequences or seeds.