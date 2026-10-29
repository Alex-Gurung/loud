# Results review (lens: results), 2026-10-29, Claude

Every number in the table traces to a `result.json`, and every mean and sd recomputes exactly. All 147 runs are complete, finite, on H100s, with clean code, and none read `val.pt`. The verdicts for E2, E3, E4, E6, E10, E14, E15 and E16 follow from the table and survive seed noise. Two of the author's readings do not hold up, though. First, the float32 control refutes the stated reason for the 0.98 bar on the H8 validation, for `x0_lambdas` and for the value and projection gaps; only the query gap is rounding. Second, in H8, Boost is worth nothing, so E15's "not met" cannot speak to the real chain. Smaller problems: E16's board numbers mix seed sets, the E4 explanation is wrong, two "not met" arms never acted, and the README still says nothing has run.

### 1. [major] The H8 validation bar was set at the observed value, and the float32 control refutes its explanation for `x0_lambdas`, `c_v` and `attn.c_proj`
- where: `configs/h6/h8_validation.yaml:4-8,19`, `configs/h6/h8_validation_fp32.yaml:1-3`, `README.md:540`
- evidence:
  - `h6/h8_validation/s0.superseded1` failed at the plan's 0.99 (min 0.98005). Commit dfd941a then set `min_cosine: {min: 0.98}`. The rerun passes by 5e-5.
  - The config says the gap is the record's "bf16 finite differences through the softmax". The fp32 control (`h6/h8_validation_fp32/s0/h8_validation.json`) contradicts this:
    - Query matrices move from 0.9856–0.9937 to 0.9922–0.9981, so the query gap is rounding.
    - `x0_lambdas` stays at 0.98005 → 0.98004.
    - `attn.c_v` (0.9946–0.9975) and `attn.c_proj` (0.9949–0.9986) are unchanged to four decimals.
  - The fp32 config's own rule is "if not, H8's … credit differs from the record's and E6/E7's H8 results need that caveat". So by that rule the remaining gap is not rounding.
  - The comment "keys, MLPs, head and embeddings at 0.9995 or more" is false. Below 0.9995 are `c_v` 0.9946, `attn.c_proj` 0.9949, `mlp.c_proj` 0.9981–0.9995 and `value_embeds` 0.9988–0.9993.
  - On the trained checkpoint (`h6/cos_to_autograd_h8_c100`), H8's `x0_lambdas` gradient has cosine −0.069 to autograd. PR #40's Cut has +0.311 (`cos_to_autograd_cut_c100`), though it has its patches on.
  - So README:540's "synthesis mode follows PR #40 exactly" is not what the validation shows.
- fix:
  - Rewrite the two config comments and the README line: queries are rounding; `x0_lambdas`, `c_v` and `attn.c_proj` differ systematically, cause unknown.
  - Compare the eight `x0_lambdas` entries one by one (H8 synthesis against the record, float32) to find the cause. This settles it; unverified here (needs a GPU).
  - Either justify a scalar-specific bar or restore 0.99 for block matrices.

### 2. [major] Boost does nothing in H8, so E15 has nothing to test there and E1's "cut-only higher" fails
- where: `artifacts/runs/e1/h8_cut*/s*/metrics.jsonl`; `loud/h8.py:760-761` (the handoff cosine is logged only for Boost runs)
- evidence:
  - At update 242, on the same batch (data order is fixed across seeds), the no-Boost run's training loss is lower in all three seeds: 4.686/4.684/4.685, against lens Boost's 4.723/4.721/4.740.
  - Final val: `h8_cut_noboost` 3.8021 ± 0.0105 against `h8_cut` 3.7993 ± 0.0032. Paired differences are −0.0053, +0.0044, +0.0094, so Boost's gain is 0.003 ± 0.007, not a result.
  - E15 asks whether Boost's lens is a crutch for a weak early Cut gradient (the plan: cut cosine 0.68 at initialization on the real chain). In H8 the handoff cosine is already 0.992, and no Boost does as well as Boost. E15's arms can only tie or lose to a baseline that equals no Boost.
  - `h8_cut_noboost` logs no `handoff_cut_cosine`. Its handoff is update 1, and `diagnose` is not called at step 0. So E15's "raises the handoff cosine" criterion has no no-Boost reference.
- fix:
  - In E1's and E15's readings, state that H8 shows no Boost effect, so E15's "not met" does not transfer to the chain. Do not run E15's λ follow-ups in H8.
  - Log `cut_cosine` at update 242 (and at update 1) in the no-Boost arm.

### 3. [major] The README says nothing has run on a GPU, and the time estimates were never replaced
- where: `README.md:66-73`
- evidence: line 66 says "Nothing has run on a GPU yet. The 161 jobs are about 21 GPU-hours … by v2's single-GPU estimates". In fact 147 runs finished between 14:16 and 22:41 on 2026-10-29. Plan §2 says to "measure and replace after E1".
- fix: State the current state, and the measured times with the co-tenancy caveat in #8.

### 4. [minor] E16's board verdict compares three-seed references with two-seed arms
- where: `loud/board.py:177-183`
- evidence:
  - The board (and the author's reading) gives track-3 gains of +0.1416 (backprop) and +0.1448 (H8). These use the three-seed means of `e1.bp_b4_pr40` (3.6396) and `e1.h8_cut` (3.7993).
  - README:204 says seeds 0–1 are the fixed model's arms. Seed-matched (as `loud.plots e16` prints), the gains are +0.1411 and +0.1430.
  - The verdict is unchanged. It is a sign test on one pair of architectures.
- fix: In `verdict`, restrict the e16 references to the arms' seeds (as `plots.py` does), and report the leaderboard's +0.127 next to it.

### 5. [minor] E7's skip is borderline, and it depends on how the eight sites are reduced
- where: `configs/e7/precondition.yaml:1-3`, `loud/gate.py:24-27`
- evidence:
  - Per site, the variance ratio at 50% runs from 0.841 (S1) down to 0.111 (S8). The median is 0.538 (the gate's reduction); the mean would be 0.498.
  - At 100% the median is 0.561 and the mean 0.535.
  - S6–S8 sit at 0.11–0.31 at both checkpoints, and S5 is at 0.503/0.521. Every site is below 1, so the prior lowers variance everywhere.
  - The whole call rests on one held-out sequence (`row: 0`).
  - The comment says "median over the nine sites"; the gate reads `S[1-8]`.
- fix: Report E7 as "borderline (0.54/0.56 against 0.5; S5–S8 pass)", not as a clean skip, and correct the comment. The two arms are 6 H8 runs of about 100 s, so running them is a cheap way to settle it.

### 6. [minor] The E4 explanation is wrong: the multiplier grew in the warmdown tail, not from steady acceptance
- where: `artifacts/runs/e4/adapt/s*/metrics.jsonl`
- evidence:
  - `adapt` seed 0's multiplier: 2.5 at update 50, then 1.5–3 through update 500, 4.15 at 550, 6.9 at 575, 23.1 at 600, and 40.9 at 606.
  - Acceptance over seeds 0–2: 0.60–0.61 over updates 1–500, 0.65–0.67 over 501–575, and 0.77–0.84 over 576–606.
  - Mechanism: once the hand warmdown shrinks the steps, the paired 4-row check accepts almost every one, so v2 undoes the last ~10% of the warmdown.
  - Final multipliers are 41/52/32 across seeds. `adapt_nowd` ends at 1.2–1.5.
- fix: Correct the reading. Show the multiplier per 50 updates in E4's table.

### 7. [minor] E10's clip and E11's runway never acted, and E10's agree kept at most 0.5% of entries; "not met" there tests nothing
- where: `loud/board.py:135-138`; the runs `e11/runway`, `e10/clip`, `e10/agree`
- evidence:
  - **E11:** at every logged site and update, 99.98% of tokens (minimum 99.17%) pick d = 0.08, so α ≡ 1. Paired val against `h8_synth` differs by +3e-5, −1.9e-4 and −8e-5.
  - **E10 clip:** `cosine_to_identity` stays at 1.001–1.003 throughout, and paired val is within 7e-4.
  - **E10 agree:** `agree_kept` is 5.4e-5 at update 250 and 5.4e-3 at update 1210 (val 4.49).
  - The author already calls E11 "null by construction"; the board still says "not met".
- fix: Give these a separate board verdict, "null (rule inactive)" for E11 and E10 clip, and state E10 agree's kept fraction in its note.

### 8. [minor] `cosine_to_identity` is computed under bf16 autocast, so it exceeds 1
- where: `loud/h8.py:595`, which runs inside the autocast opened at `loud/h8.py:691`
- evidence:
  - `e10/clip/s0` logs 1.0028, 1.0018, 1.0013 and 1.0012.
  - Under autocast, `a @ b` of two float32 vectors returns bfloat16 (checked on CPU torch: `(a@b).dtype == torch.bfloat16`), so the numerator carries about 2^-9 relative rounding.
  - E6's values (0.34–0.73) are not materially affected.
- fix: Compute it in `torch.autocast(..., enabled=False)`, as `h8.py:355` and `:440` already do.

### 9. [minor] `train_seconds` were measured with several jobs sharing a GPU, and nothing flags this
- where: `collect.py` table; `result.json` (no co-tenancy field)
- evidence:
  - `e1.h8_cut` took 36.7/58.9/72.6 s across seeds; `e1.bp_b4_pr40` took 211–298 s.
  - Jobs carry a `slot` (`…-gpu0`), and workers `gpu0-0` and `gpu0-2` ran on one GPU at the same time.
  - The timer excludes the warmup and compile, `diagnose`, held-out monitoring, checkpoints and official scoring. It includes E4's held-out judge and E3's end-of-run rule scoring; that accounts for E3's 0.03–1.06 s gap between `metrics.jsonl` and `result.json`.
  - No experiment is judged on time.
- fix: Record the slot and the number of concurrent jobs in `result.json`, and have `collect.py` flag shared-GPU timings. Do not use them to replace the plan's estimates.

### 10. [minor] E5's float32 rerun tests the method without rounding; it does not test the bf16 readout the plan specifies
- where: `loud/diagnostics.py:188-200` (`precision`); queued `e5.readout_fp32_c20/c100`
- evidence:
  - The attribution holds as far as it goes. At σ = 0.01, `linearity_check`'s bf16 nonlinearity is 0.88–1.29 at every site, against 0.006–0.017 in float32. Rounding alone therefore swamps the solve, which explains cosines of 0.006–0.11.
  - But median condition numbers rise from 913 (S2) to 13,730 (S8) at 20%, and from 1,195 to 43,264 at 100%. That amplifies even float32's ~1% forward-difference error.
  - The plan's test is "from bf16 forward differences", because the chain runs in bf16 and differences site changes, not whole losses.
  - unverified: the fp32 reruns settle the method question only.
- fix: Treat an fp32 pass as necessary but not sufficient. Add a bf16 replay that differences the changes the way the chain's kernels do before taking E5's decision.

### 11. [nit] Small inaccuracies
- `loud/board.py:36` still says "rescoring 1e-5"; the gate is 1e-4.
- `score.json` names `final.pt` with the sha256 of the exact weights, which were then overwritten in bf16 (for `e6.token_k4.s0` the score hash is `019d8136…`, the stored file `c20a738e…`).
- E13 reading: at θ = 4 the low tier holds enough tokens (44%) but 27% of the energy; at θ = 2 the energy passes (6.8%) but the tokens do not (21%). It is not only "too few tokens".
- `verify.identity_match` has `train_loss_max_diff_all` 0.103 from a single update (272: 4.605 against 4.502; neighbours within 0.007).
- `e15.latent_only.s1` spikes at update 631 (4.56 → 7.25) and only partly recovers. It is not the worst seed (s2, 4.067).

## Checked and fine
- **Table:** every row traced to `result.json`; means and sample sds recomputed for all arms, matching `collect.py`. `val_loss`/`test_loss` equal `score.json` in 112 runs. The final `heldout_loss` equals the `final: true` line of `heldout.jsonl` in 110 runs.
- **Logs agree:** in `metrics.jsonl`, step count, contiguity, last train loss and `train_seconds` match `result.json` for every training run (E3 differs by ≤1.1 s, see #9). No NaN or Infinity in any JSON or JSONL, and no early stops; `verify.cut_prefix*` stop at 262 by design.
- **No `val.pt` in training:** `eval_every` is 0 in every experiment cell. Diagnostics and gates read held-out row 0, or all 32 held-out rows for E13.
- **Provenance:**
  - All runs ran on an H100 80GB HBM3, torch 2.13, with clean git.
  - The three snapshots behind experiment cells (`dfe1b7c0`, `f24d0669`, `7e9fd479`) differ only in `diagnostics.py`/`options.py` (diagnose options) and `run.py` (memory cap). The training code is identical, so the E2/E3 "mixed code" flags and E1's two commits are benign.
  - The verification snapshot `ac849c9f` differs only by `donated_buffer = False`.
- **Comparable cells:** compared arms share seeds 0–2 (E16 0–1), 10M, held-out 32 and recipe/batch.
  - The E6, E10, E11 and E14 arms reproduce their reference's Boost phase exactly: training loss at updates 242–243 is identical per seed.
  - E3's trunk equals `e1.bp_b8` through update 121.
- **No warmup leak:** the first-update loss is identical across trainers sharing an init (8.40347 at seed 0), and it matches a fresh model (held-out loss at step 0 is 8.407).
- **Seeds and spikes:** data order is fixed across seeds (`ORDER_SEED` 1, as `backprop_preset.py`), so sds measure init only. The ~+0.8 jump at update 981, seen in most runs and every seed, is one batch, not an instability.
- **Verifications:**
  - Record rescore 3.944756 (1e-4).
  - `bp_b8` 3.70762 against 3.70835. Ledger: 9 of 10 logged updates within 0.004, one at −0.0112 (update 366), mean |d| 0.0034, mean signed −0.0021.
  - The unmodified script and `bp_b8_onefwd` are bit-identical.
  - Identity 4.1771 against 4.1772, first 20 updates within 0.0031.
  - Cut prefix within 0.0008 over 20 updates and 0.009 over 262.
- **H8 validation (queries and partial rows):**
  - The query gap is rounding (fp32 ≥ 0.992).
  - `synth_vs_real` medians are 0.998–1.000 (p10 ≥ 0.994 with record routes, ≥ 0.9996 with all routes).
  - `synth_vs_exact` is 0.50, which is √(128/512) as expected.
- **Verdicts recomputed against the plan:**
  - E2 +0.034, E3 +0.038, E4 +0.355, E10 +0.0003, E11 −0.0001, E14 −0.0008, E15 +0.010: all not met, none fragile.
  - E6: token beats time by 0.034/0.050/0.068 in every seed.
  - E16: order agrees.
  - E1: arms as quoted; `h8_synth` lies between 3.777 and 3.896.
- **E3 rules behave as specified:** select acts 9 times; the ensemble keeps the best copy (mean 4.291 against 4.218); barriers end positive for the ensemble and negative for the averaged and selected arms.
- **E13's skip is robust:** no θ meets both conditions at either checkpoint.
- **E8 numbers as quoted:** `mlp.c_fc` 0.659 → 0.841 at 100%. Note that exact gradients already put 0.66–0.95 of the energy below SNR 1 at 100%, which cuts against the plan's prediction of no difference with exact gradients.
- **E11's explanation holds** (#7).
