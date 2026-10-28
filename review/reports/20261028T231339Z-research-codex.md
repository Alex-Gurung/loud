# research review: Codex

- source: frozen copy `fbeb08e8f2d69406` at 20261028T231339Z (git 1139bd6 plus the working tree)
- reviewer: Codex 0.165.0, model gpt-6.1-sol, session 01a184ca-770c-7fb1-a76f-272d1236140a
- focus: Round 5: fidelity to the plan ("Noise Forward-Only Research Harness Engineering Handoff.md") and to PR #40 (external/noise/archive/10M/train/val/run.py) of the experiment code merged since commit 7eab8cb: the Cut-chain port loud/cut.py (H5; E6 and E7 on its chain; the verification configs configs/verify/cut_*.yaml and configs/e1/cut_1m.yaml), E2/E4/E10 in loud/variants.py and configs/e2, e4, e10, E3 in loud/population.py and configs/e3, E8 in loud/curvature.py and configs/e8 (it damps after polar() by default and keeps the plan's literal placement as a control; is that justified?), E9 in loud/fisher.py and configs/e9. Do the configs answer each experiment's question with one change per variant, the right baseline, seeds and logs the plan names? Is anything the plan asks for missing?

The experiment set still has deviations that prevent interpreting every cell as the plan’s specified test. E10 changes the agreement criterion, E7 can run before its prerequisite diagnostic, and the Cut verification gate permits twice the stated seed-noise tolerance. CPU checks support E8’s placement rationale, but GPU fidelity remains unverified.

### 1. [major] E10 measures agreement among active tokens rather than across the batch
- where: `loud/variants.py:217`, `loud/variants.py:224`
- evidence: The plan requires keeping components where at least the threshold fraction “of tokens agree in sign.” The implementation divides by the number of **nonzero contributions**: `total.mul_(majority >= nonzero.mul_(2 * self.threshold))`. I exercised `TokenShaping` with ten tokens, four positive contributions and six zeros. At threshold 0.7, it retained the gradient, returning `4.0`, although only 40% of tokens supported that sign. ReLU-squared makes zero contributions a normal case. README describes the active-token interpretation but gives no reason for changing the experiment.
- fix: Use the total batch-token count as the denominator. If active-token agreement is intentional, identify it as a separate variant and retain the literal plan control.

### 2. [major] E7 starts before the diagnostic that is supposed to justify running it
- where: `configs/e7/ramp128_f30.yaml:11` and the other three E7 configs
- evidence: E7 requires the H6 credit diagnostic first, with median cosine above 0.9 at 20% and below 0.7 at 100%. Its configs mention that prerequisite only in comments; their `after` lists contain no H6 jobs. Moreover, E7 has priority 65 while both prerequisite diagnostic configs have priority 70, so ready E7 jobs run first. The documented bulk submission therefore launches the conditional experiments before establishing their premise.
- fix: Make the diagnostics dependencies and add a concrete eligibility check, or exclude E7 from automatic bulk submission until the diagnostic results have been reviewed.

### 3. [major] The Cut verification gate accepts discrepancies above the plan’s tolerance
- where: `configs/verify/cut_prefix_match.yaml:19`
- evidence: The gate specifies `train_loss_max_diff_all: [0, 0.02]`; only the first 20 updates receive the tighter 0.003 check. Thus step 100 and every checked Cut update can disagree by almost 0.02 and still release E1, E6 and E7. The plan asks for per-step agreement to a few decimals and sets seed noise at 0.01. The config attributes larger differences to accumulated rounding, but README states that no GPU run has occurred, so that allowance has no measured justification.
- fix: Tighten the later-step gate to at most 0.01. Relax it only after measuring and documenting the discrepancy against an otherwise identical reference execution.

### 4. [minor] Cut’s end-merge selection calls are absent from the held-out log
- where: `loud/cut.py:259`, `loud/cut.py:295`, `loud/cut.py:450`
- evidence: Cut replaces the reference’s `holdout_loss` with `virtual_holdout_loss`, which returns a scalar without logging. The reference’s `joint_parts()` calls that function for candidate selection throughout the merge. Only the plain score, winning score and final choices survive in the result. H3 explicitly requires logging every scoring call, and README claims every call appears in `heldout.jsonl`; these decision-making calls bypass it.
- fix: Wrap the existing virtual-rank scorer with logging, preserving its reduction order. Record each candidate score and enough candidate metadata to reconstruct the selection.

### 5. [minor] E3 omits the required fraction checkpoints and trajectory means
- where: `loud/population.py:220`, `loud/options.py:116`, `README.md:302`
- evidence: Population training creates only `FractionCheckpoints(..., enabled=False).save_final(model)` and never creates trajectory means. Its option restrictions also prevent enabling those facilities. The plan requires H4 artifacts for every run. README acknowledges the omission because “there is no single trajectory” after branching, but that does not prevent saving the trunk before branching and the four copies afterward.
- fix: Save the trunk at applicable fractions and per-copy checkpoints and means after branching, with copy identifiers. Document that representation of H4.

## Checked and fine

- Focused CPU checks passed for Cut full-cycle equivalence and partial-window decoding, E4 rollback, E8 damping, and E9 Fisher factors.
- E8’s post-`polar()` placement has a concrete justification: the CPU check demonstrates that pre-polar scaling is largely undone. The literal pending-update control remains configured.
- E9’s sampled-label backward leaves the ordinary training gradient unchanged; its factor matches an independent token-by-token calculation.
- Main comparison cells use matching training ranges, held-out sizes and seeds; E7 uses the plan’s two seeds.
- Cut delegates model initialization, optimizer construction, schedule and end merge to pinned PR #40 code. Partial rows affect the chain while retaining the other probe budgets.
- Inspected selection paths use training-held-out data rather than `val.pt`; official scoring follows final-weight selection.
- CUDA kernels, compiled execution and published-log numerical agreement were not exercised.