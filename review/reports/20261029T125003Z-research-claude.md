# Research review (Claude), round 7: v2 fidelity, frozen copy of a387d8d

Verdict: H8's core (cut-mode detaches, Boost's readouts, synthesis's block-local site mapping, the Scheme, reforward and the validation settings) matches v2 and PR #40, and most experiment cells answer their questions. Four things would still give wrong or unreadable answers. E8's SNR scaling takes its directions from the update that already holds the batch's own gradient, so its noise estimate is biased toward keeping noise. E16's track-3 model trains its n-gram tables at 1/750 of the record's rate. E15's handoff-cosine criterion has no baseline value. E14's gap fraction uses a backprop cell trained with a different recipe. The run order also delays the cheap proxy experiments behind every H8 cell, which works against "fast iteration first".

### 1. [blocker] E8's SNR scaling picks its directions from the update that holds the measured batch, so on pure noise it keeps r ≈ 0.85 instead of ≈ 0.18
- where: `loud/snr.py:103-113` (`direction` takes the SVD of `pending`), `loud/anvil.py:40` and `loud/anvil.py:79-85` (`pending = g.lerp(w * fast + (1 - w) * slow, beta)`, after the rails were stepped with `g`); the claim at `loud/snr.py:17`
- evidence: `pending` holds the batch gradient `g` at weight (1 − β) + β·[w(1 − fast_beta) + (1 − w)(1 − slow_beta)]. That is 0.21 with PR #40's rails (the H8 setting, w = 1) and 0.14 with boost-cut's (the proxy, w = 0.4385). The docstring says "about a tenth". In noise directions this is roughly half of `pending`'s noise energy, so the singular directions line up with this batch's noise, m_i² is inflated, and r_i = max(0, m_i² − s_i²/8)/m_i² moves toward 1: the bias the estimator exists to remove. I simulated it on pure noise (μ = 0, 128 × 128, B = 8, the rails' betas), in `scratchpad/review-research-claude/scripts/e8_bias2.py`. With directions from `pending`, mean r was 0.847, top-16 r 0.994 and Frobenius ratio 0.895. With directions from the momentum before the batch, they were 0.182, 0.178 and 0.343. boost-cut's rails gave 0.856 against 0.186. So the `snr` arm will look almost like `control`, and the `rate` arm takes nearly full steps, in exactly the low-SNR directions E8 is about. The README's own "Differences" item rejects same-sequence directions for the diagnostic for this reason ("SNR ~100 along them"). Both settings use this code (the `configs/followups/e8/` proxy and `h8_*` cells).
- fix: take (u_i, v_i) from the momentum before this batch's gradient enters, w·fast + (1 − w)·slow before the lerp (as `snr_spectrum` does), in all three arms, so the control stays matched. Fix the docstring's weight.

### 2. [major] E16's track-3 model trains its hashed n-gram tables at Adam rate 0.0021, 1/750 of record #5's
- where: `loud/h8.py:596` and `loud/h8.py:614`, `loud/proxy.py:56`, both using PR #40's `make_optimizer` (`external/noise/archive/10M/train/val/run.py:2191`: `scale = 300 if ('wte' in name or 'value_embeds' in name) else 30 if 'lambdas' in name else 1`)
- evidence: PR #40 has no n-gram tables, so `ngram_embeds.*` falls into the 1× group. I built a tiny track-3 model through `arch.make_model` and PR #40's `make_optimizer` (`scratchpad/review-research-claude/scripts/e16_groups.py`). The Adam groups were ×30 for the lambdas, ×300 for `wte` and `value_embeds`, and ×1 for `lm_head`, `ngram_embeds.{2,3,s}`, `ngram_proj` and the blocks. Record #5's recipe (`train/arch/run.py:2468-2470`, `train/arch/log/config.json`: `emb_scale` 750, `ngram_scale` 0, so tables at 750×) trains the tables at 1.575. The tables start at N(0, 1) and are stored in bf16 on the GPU. At a step of 0.0021, 32% of N(0, 1) bf16 entries cannot change at all (I measured the share that rounds back), and the rest move about 0.2% per hit. So both E16 trainers train a model whose n-gram embeddings stay at their random init: not "track-3 record #5's design", and not comparable with the leaderboard's 3.211 against 3.338. The README does not flag this.
- fix: put `ngram_embeds` in the embedding group (300× under PR #40's scales) for `arch: ngram` in both `H8.make_optimizer` and `Reference`, for example with a wrapper that moves those parameters before `make_optimizer` builds the groups. State the choice in the README.

### 3. [major] E15's second success criterion ("raises the handoff cosine at equal final loss") cannot be evaluated
- where: `loud/h8.py:729` (`handoff_cut_cosine` only `if self.targets is not None`, i.e. only with `latent_boost`); `loud/plots.py:281-297` (the e15 table has val and gap only)
- evidence: arm (a) is `e1.h8_cut`, which has `latent_boost: none`, so it never logs `handoff_cut_cosine`. No table or plot reads the cosine from arms (b) and (c) either. The same gating hides E14's comparator: `effective_rank` is logged only with `nextlat` (`loud/h8.py:726`), so `e1.h8_cut` has no ranks to compare the arms with.
- fix: in `diagnose`, log `handoff_cut_cosine` whenever the run has a Boost phase, and `effective_rank` at 20% and 100% for every cut-mode run (both are untimed and on 4 held-out rows). Add a handoff-cosine column (with `e1.h8_cut` as arm (a)) to the e15 table, and a rank table to e14.

### 4. [major] E14's "fraction of the gap closed" mixes the estimator with a recipe change
- where: `loud/plots.py:281` (`low=('e1', 'bp_b4')`), `configs/e1/bp_b4.yaml` (`recipe: boostcut`)
- evidence: the gap's top end, `e1.h8_cut`, runs PR #40's recipe: lr 0.0021, betas (0.5, 0.98), warmdown 0.4 down to 30% of each rate with no end merge in H8 (no `finish`), head std 0.02, ANVIL 0.026, fast weight 1. The bottom end, `e1.bp_b4`, runs boost-cut's: lr 0.003, betas (0.7, 0.95), warmdown 0.3 to zero, head std 0.001, ANVIL 0.023, fast weight 0.4385 (`loud/refs.py:41-48`). The rate floor alone is worth about what PR #40's merge bought (3.703 to 3.676). The README says for E16 that matching the recipe is "so the comparison isolates the trainer", and `e16.bp_fixed` (PR #40's recipe, batch 4, no merge) is that matched cell. But it has seeds 0 and 1 only, runs at P2 priority 75, and the gap table does not use it. So "closes at least half the gap" can pass or fail because of the recipe.
- fix: use `e16.bp_fixed` as the low end of E14's (and E15's) gap. Give it seed 2 and priority 50 with E1. Keep `e1.bp_b4` as a second column, since it is the plan's literal cell.

### 5. [minor] The run order puts every H8 seed of P1 ahead of the cheap proxy experiments' first seed, and runs P2-only real-chain gates before E1
- where: priorities in `configs/` (E6/E7 55, E14/E15 60, E2/E4 62, E3 64); `README.md:102` and `README.md:122`
- evidence: the queue claims by priority, then seed, so the first-seed-first rule holds only within a band. On one GPU, by the README's estimates, seed 0 of E2, E3 and E4 (5 to 12 min each) waits for all three seeds of E6 and E7 (24 × ~10 min) and E14 and E15 (15 × ~6 min): about 5.5 GPU-hours after E1. Before E1, priorities 3 to 11 also run `verify.cut_smoke` (~20 min), `verify.cut_prefix` and `cut_prefix_reversed` (~30 min) and `verify.track3_smoke`. These gate only the P2 real-chain confirmations and E16. Separately, the README says "Every H8 experiment waits for" `h6.h8_validation`, but `e1.h8_synth`, `e1.h8_cut` and `e1.h8_cut_noboost` (9 runs, about 1.1 h) and the H6 diagnostics on them do not. Counting the configs outside `followups/` gives 156 jobs, not the README's 154.
- fix: give every P1 cell one priority, so seed 0 of each P1 arm runs before any seed 1. Move the real-chain verification and `track3_smoke` after P1's seed 0. Either gate E1's H8 cells on `h6.h8_validation` or correct the README.

### 6. [minor] E7's prior differs between H8 and the real chain at S0, and its gate reads nine sites although the ledger says eight
- where: `loud/h8.py:318-322` (an x0-skip guess at S0, applied by `Scheme` at every site); `loud/cut.py:30` and `loud/cut.py:283` (the chain has no guess at S0); `configs/e7/precondition.yaml` (prefix `S`)
- evidence: H8's E7 arms get a control variate at S0 (wte's credit) that the chain they are confirmed on does not have. PR #26, which the plan's guess follows, measures S0 with its own cycle. The gate's prefix `S` matches S0 to S8. LEDGER round 6 #4 records the fix as "the precondition reads the eight mid sites", which stopped being true when the sites were renamed.
- fix: drop the S0 guess in H8, or add it to the chain, so both implement one E7. Name the sites the gate reads (S1 to S8, to match the chain) and update the config comment.

### 7. [minor] The CPU evidence for H8's validation bar cannot see the chain's own-query key truncation, and the GPU gate keeps it on
- where: `loud/diagnostics.py:215-229` (`h8_validation`), `loud/diagnostics.py:240` and `loud/diagnostics.py:268-269` (`synth_vs_real`); PR #40's `query_keys: 256` (`run.py:110`), used at `loud/cut.py:215-216`
- evidence: `strong_keys` moves only the top `min(256, length)` logits per head. The CPU checks run 64-token sequences (`tests/method_checks.py:184`), where the cap does nothing, so the README's "cosine 0.996 to 1.0 per matrix" never exercised it. On the GPU (2,048 tokens, windows of 1,024 and 2,048) it is active for most tokens. So a `h6.h8_validation` failure, which blocks every H8 experiment, could come from this cap rather than from H8. `synth_vs_real`'s "all routes" variant does not isolate finite differences and bf16 as its docstring says.
- fix: run `h8_validation` with `query_keys = 0` (the plan's list turns off every other cross-token or truncating piece), and add a `query_keys 0` variant to `synth_vs_real`.

### 8. [minor] E6's `time` arm cannot show the plan's mechanism, rank at most m across an update
- where: `loud/h8.py:365-379` (signs per sequence, rows shared); the plan's E6 rationale, repeated in no README caveat
- evidence: the column signs are drawn per sequence, as PR #40 does (`Cut.generator` seeded per sequence), so each sequence's credits lie in a different m-dimensional subspace. I measured the rank of the scheme's output (`scripts/e6_rank.py`, B = 4, T = 256). For m = 128, 64 and 32 it is 128, 64 and 32 within one sequence, and 413, 252 and 128 across the batch. So at K = 4 the update's writer gradient is nearly full rank, and the time-versus-token gap the plan predicts may not appear for this reason.
- fix: say in the E6 configs and README that the rank bound is per sequence (about batch × m per update). If the mechanism matters, add one H8-only arm with signs shared across the batch.

### 9. [minor] H6 item 1 has no estimator argument and no per-site output
- where: `loud/diagnostics.py:115-133`
- evidence: v2 specifies `cos_to_autograd(estimator, ckpt)`, "per matrix type and site". The tool always runs PR #26's FastRelay and reports per matrix type, block and group only. There is no offline cosine to autograd for the Cut chain or for H8 synthesis, the estimators the experiments use.
- fix: take the estimator (identity, Cut via `cut.py`, H8 synthesis) as an option and add per-site credit cosines. Or note the reduced scope in the README.

### 10. [minor] Several plan comparisons are missing from the tables
- where: `loud/plots.py:363-370`, `loud/plots.py:483-492`
- evidence:
  - The E7 table compares only with the full cycle, not with its matching no-prior arms (`e6.token_k8`, `e6.token_k4`), which the plan names.
  - E15's table omits arm (d) (`e1.h8_cut_noboost`).
  - E14's `top_cosine` log is never tabulated.
  - `stable_rank_mean` averages Boost-phase updates too (`rank_every` acts in every phase, `loud/h8.py:702`). Those updates are identical across E6's arms, so they dilute the difference.
- fix: add those rows and columns, and average the stable rank over Cut-phase records only.

### 11. [nit] Stale README text
- where: `README.md:54`, `README.md:437-441`
- evidence: "154 jobs" (the configs give 156). The first "Differences from the handoff" bullet (3.338, PR #44, PR #52) is now v2's own text, not a difference.
- fix: update both.

## Checked and fine
- **H8 cut mode** (`loud/h8.py:91-106`, `109-129`): the normed key/value input and the `ve_gate` input are detached, the query, the value-embedding tables and every parameter path stay live, and the token's own key and value are held too, consistent with Replay. `freeze_queries` detaches the query input only.
- **Boost** (`boost_loss`) against PR #40's `Boost.estimate`:
  - the head learns from the final readout plus both sublayer readouts of blocks 0 to L−2;
  - the mixing scalars learn from their block's attention readout with its inputs held;
  - wte learns from block 0's own-path readout;
  - each sublayer's input is detached.
- **Synthesis** (`synthesis_gradients`):
  - attention and `value_embeds[i]` take the credit at S(i+1);
  - block i's MLP takes λ_{i+1} times the credit at S(i+2), and the last MLP the top-stream credit;
  - wte takes S0's credit, and the head and scalars come from the cut graph;
  - there is one `autograd.grad` per sublayer, so nothing leaks.
- **Scheme**: 1/m scaling with random column signs is unbiased for any row choice independent of the signs. The time and token assignments match v2, and E13's tiers pick rows by loss, not by sign. The decoding is in float32.
- **Replay and the validation bar**:
  - Replay moves x0 for S0, and for mid sites runs the site's MLP and then the later blocks with every key and value held.
  - `h8_validation`'s settings match v2's list (value path, path scalars, key and gate probes and future draws off, route_k 2,048).
  - The six H8/H6 CPU checks I ran (`h8_gradients`, `h8_boost`, `h8_schemes`, `h8_arch`, `h8_snr`, `h6_synth_vs_real`) pass (132 s).
- **synth_vs_real** reads the chain's own bases and row index, in site order (emb, mid0 to mid7), so the rows and signs are the same.
- **snr_spectrum**:
  - its directions come from the checkpoint's rails, independent of the 64 sequences;
  - it uses four estimators (backprop, synthesis at K = 1, time and token at K = 4);
  - it reads the next 64 training sequences, wrapping at 100%;
  - its SNR (m² − s²/S)/(s²/8) is at batch 8.
- **E8's H8 setting leaving Boost on the record's optimizer**: defensible. All four H8 arms share an identical, exact Boost phase, so the within-setting comparison is unconfounded. The SVD polar's norm matches ANVIL II's renormalized update, so the handoff has no step-size jump, and `h8_exact_control` gives the partial-row cost. The cost is that the two settings differ in more than the estimator over the first 20%, so read "little difference with exact gradients" within each setting.
- **E13**:
  - Witten–Bell interpolation is a proper distribution;
  - scoring is causal (score, then insert);
  - the held-out rows are never inserted;
  - the tiers rule max(n-gram, model) ≤ θ is right;
  - there is a gate per θ on both checkpoints.
- **E14** follows `models/model_nextlat.py@3770be6` (fetched to scratch), apart from v2's one hidden layer:
  - smooth-L1 with β = 1, the mean over unmasked elements;
  - the next token's embedding gets gradient;
  - the mask is applied where token t+1 is EOT (id 0, which `prepare_data` prepends);
  - the EMA teacher's decay runs from 0.996 to 0.9995.
- **E2, E3, E4**:
  - E2 draws r = s·RMS·√W·u at nine sites (an antithetic pair, a clean loss logged);
  - E3 branches at update 121, sends 2 sequences to each copy, and implements the average, select and ensemble rules and the midpoint barrier log;
  - E4 judges on 4 rotating held-out rows (paired), restores the weights and optimizer state, and settles at acceptance 0.594.
- **E16 scoring and forward**: E16 scores through `utils/eval.py --model` with the pinned `train/arch/model.py`, and the cut forward equals the track-3 forward (test).
- **Data order and schedules**: `randperm(seed 1)` for every seed is also what `backprop_preset.py` does, so the 0.009 seed sd is comparable. PR #40's `stage_schedule` numbers updates continuously, so the proxy's single PR #40 schedule matches H8's.
