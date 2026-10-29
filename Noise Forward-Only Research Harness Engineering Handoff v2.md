# Noise Forward-Only Research Harness: Engineering Handoff v2

Oct 29, 2026 · @Alex

## 1. Purpose and ground rules

Build a one-GPU research harness for the Noise benchmark (pretraining a fixed 37.7M-parameter transformer without backpropagation) and run the experiments in Section 5. Every experiment answers one yes/no question with at most one tunable knob, so that what works and what fails can be read from the logs. Engineering complexity is acceptable; hyperparameter sprawl and conceptually muddy changes are not.

**The benchmark.** Fixed model ([`utils/model.py`](https://github.com/qlabs-eng/noise/blob/main/utils/model.py)), fixed data (FineWeb, BPE-4096 tokenizer, sequences of 2,048 tokens), one epoch, score = mean next-token cross-entropy on `data/val.pt` (1,114,112 predictions) via `utils/eval.py`. Three tracks: lowest validation loss (no time limit), fastest time to validation loss 3.5 on the 30M-token set, and open architecture. Full rules and records: [qlabs-eng/noise](https://github.com/qlabs-eng/noise).

**Rules every submitted run obeys** (paraphrased from the README; read the original before submitting):

1. No autograd in any mode and no hand-written backward pass. Every error signal is measured against a loss by running the model forward. A measured signal may take at most one step through a nonlinear part of the model; steps through linear maps do not count; a signal that has taken a step cannot take another. Credit cannot be chained back block by block. Entries are judged by the spirit of this rule.
2. Final weights load into `utils/model.py` with a strict state-dict match (tracks 1 and 2).
3. Train only on `data/train.pt`, at most one epoch: each training sequence feeds at most one update. Batch size and order are free. Sequences that feed no update may be used to score decisions (the current record holds out 32 for its end merge).
4. No checkpoint selection on validation loss. Score the final weights only.
5. Timing (speedrun) runs from the first update to the last, excluding validation; an untimed warmup on random tokens may compile kernels; seed 0.

**Autograd in development.** Allowed for three things, never in a submitted script: (a) as a stand-in for the estimator when testing what to do with a gradient (H1); (b) as an exact emulation of the forward-only estimator, Boost and Cut, including synthesized partial-probe noise (H8); (c) as an offline diagnostic on checkpoints (H6).

**Why the stand-ins are faithful.** On the archived 10M set the record's forward-only estimator trained to 3.703 before its end merge; backprop with the same model, optimizer, schedule and data order reached 3.708 (seed 0; 3.712 ± 0.009 over three seeds). What to do with a gradient therefore transfers from the backprop proxy. H8 reproduces the estimator's core exactly (the cut, with each token's own query live and other tokens' keys and values held), but not the record's cross-token patches (value path, input-key and gate probes, path scalars, future-token probes), which together close most of the remaining gap to backprop.

**State of the leaderboard (29 October 2026).** 30M validation track: 3.338 (record #3, a forward readout of cross-token credit along each token's path), level with backprop with the same optimizer at 3.340. 30M speedrun to 3.5: 22.9 s (an identity-path phase, then Cut with two single roots and fused kernels). Open architecture: 3.211 (a 12 × 1,024 GPT with hashed n-gram value embeddings). The full chain still costs roughly 400 forward-pass equivalents per update against backprop's 3, so the chain is the bill on the speedrun; on the validation track, changes to the update rule or the objective that help backprop are likely to help here too. [Source](https://github.com/qlabs-eng/noise).

**Convention.** Development runs use the archived 10M set: its 4,880 sequences permuted with seed SEED + 1 exactly as the record does, the last 32 held out (H3), 4,848 training sequences. Backprop-proxy experiments run at 8 sequences per update (606 updates). H8 experiments follow the 10M record's recipe (PR #40): 4 sequences per update (1,212 updates), Boost for the first 20%, then Cut. Each experiment compares arms within one harness and one batch size.

## 2. Environment and setup (one GPU)

Target: one H100 80 GB. Linux, bf16, [uv](https://docs.astral.sh/uv/). The model is 75 MB in bf16, so memory is never the constraint; wall-clock is.

**Repositories.**

| Repo | What you need from it |
| --- | --- |
| [qlabs-eng/noise](https://github.com/qlabs-eng/noise) | `utils/model.py`, `utils/eval.py`, `utils/prepare_data.py`, tokenizer; current scripts `train/val/run.py` (track 1), `train/speedrun/run.py` (track 2), `train/arch/` (track 3), `train/val/dust.py`; the 10M-era scripts and records in `archive/10M/`. |
| [spicylemonade/boost-cut](https://github.com/spicylemonade/boost-cut) | `noise/analysis/backprop_preset.py` (commit `539386e`): backprop with the record's model, init, ANVIL II + Adam, rates, schedule and data order, 610 updates of 8 sequences. Also `noise/analysis/prepare_test.py` (a held-out test split) and `ledger/` (configs and results of every Boost/Cut run). |
| [erodola/noise, release `relay-track1-v1`](https://github.com/erodola/noise/releases/tag/relay-track1-v1) | `check_relay.py`: cosine between an estimator's weight gradient and autograd, per matrix type, on one sequence. Starting point for H6 and for validating H8. |
| [KellerJordan/modded-nanogpt PR #360](https://github.com/KellerJordan/modded-nanogpt/pull/360) | Reference ANVIL II implementation. |

**Data.** `uv sync && uv run utils/prepare_data.py` writes `data/train.pt` (30M tokens: 14,648 sequences of 2,048 predictions) and `data/val.pt`, both hash-checked. For development, also run `archive/10M/utils/prepare_data.py --output archive/10M/data`, which writes the archived 10M set (the first 4,880 sequences of the 30M set) in the layout the 10M scripts require. The first 488 sequences form a 1M slice, used only for timing confirmations on the real chain.

**Model facts you will need.** 8 blocks, width 512, 8 heads of 64, MLP 2,048 units with ReLU-squared, RMSNorm, rotary positions, sliding attention windows (pattern SSSL), vocabulary 4,096, logit softcap 15 tanh(z/15). Each block's input is λ\_r × (previous block's output) + λ\_0 × x0, where x0 is the normalized embedding, so the embedding reaches every block directly, not only through block 1. Value embeddings on every other block behind a sigmoid gate. The nine measurement sites are the raw embedding (S0) and each block's mid stream after attention and before the MLP (S1 to S8). Read `utils/model.py` before touching anything.

**Expected single-GPU times** (published 8×H100 numbers multiplied by 8 where they exist; measure and replace after E1):

| Run | 8×H100 (published) | One H100 (estimate) |
| --- | --- | --- |
| Backprop proxy, 10M | 38 s | about 5 min |
| H8 emulation, cut or boost mode, 10M | none | about 6 min |
| H8 synthesis mode, 10M | none | about 10 min |
| Identity-path forward-only, 10M full epoch (PR #26) | 47 s | about 6 min |
| Record Cut chain, 1M slice | none | about 30 min |
| Record Cut chain, 10M | 35 min | about 4.5 h |

**Verification before any experiment.**

1. `uv run utils/eval.py` on boost-cut's released `track1_boostcut_final.pt` (sha256 `499dff9f...`) must print 3.94481. This proves the data and the scorer match.
2. The unmodified `backprop_preset.py` (before H1's hold-out change) on the 10M set, seed 0, one process: final validation loss within 0.02 of 3.708. Then rerun with H1's hold-out and record the new number; that is the baseline from here on.
3. The identity-path script for a full 10M epoch on one GPU: within 0.02 of 4.18.

If any check misses its tolerance, find the cause before building on it.

## 3. Shared infrastructure

Build H1 to H4, H7 and H8 first (two to three days); H5 and H6 alongside the first experiments.

**H1. Single-GPU proxy trainer.** Start from `backprop_preset.py`: model, init, ANVIL II on the block matrices, Adam elsewhere, rates, warmdown and data order. Two changes. Hold out the last 32 sequences after the SEED + 1 permutation, exactly as the record does: the preset sets `steps = len(training) // batch` and trains on all 4,880, so H3 would otherwise score sequences the model trained on. Add `--sequences` and `--holdout` flags (the output flag is `--output`; at one process no gradient accumulation is needed). Prepare data in the 10M layout with `archive/10M/utils/prepare_data.py --output archive/10M/data`; the archive loader requires exactly 4,880 rows and the 10M hash, so pointing it at the 30M `data/train.pt` raises an error. Writes `config.json`, `metrics.jsonl` (per update: step, train loss, step time, effective rates) and `result.json` (final val loss from `utils/eval.py`, wall-clock). The model's forward averages the loss by default; `utils/eval.py` passes `'sum'` explicitly.

**H2. Per-sequence gradients.** Keep each sequence's gradient before averaging (bf16, discarded after use). Used by E8.

**H3. Held-out scorer.** The 32 held-out sequences never feed an update. `score(weights, rows=None)` returns the mean loss over all of them or a given subset, in one forward pass. Every in-run decision uses it, never `val.pt`; log every call.

**H4. Checkpoints and averages.** Save weights at 0, 10, 20, 50 and 100% of every run. Provide `EMAWeights(decay_schedule)`, used for the EMA teacher in E14 and E15, with decay rising from 0.996 to 0.9995 over training as in ProteinJEPA.

**H5. Forward-only ports, single GPU.**

- The 10M record chain: `archive/10M/train/val/run.py`, byte-identical to PR #40's commit `4ef285e`. Its defaults: `switch` 0.2, `boost_batch` and `cut_batch` 4, `chain_sigma` 0.01 times each token's RMS, `route_k` 512, `own_query`, `value_path` and `path_scalars` on, `future_draws` 128. Process its two-GPUs-per-sequence layout sequentially.
- The identity-path estimator: `archive/10M/train/speedrun/run.py`, or PR #26's simpler PyTorch version. Pin the commit in `config.json`; the live `train/speedrun/run.py` is now a different, 30M record.
- For reference only: the live `train/val/run.py` is 30M record #3 (3.338), the base for any eventual submission.

Verify the port with a full-schedule run stopped early (`--stop-after-step`), comparing per-step losses at steps 2, 10 and 100 against the published `metrics.jsonl`. A 1M-slice run cannot be compared this way, because its Boost phase ends near update 23 instead of 242.

Flags the probe-budget experiments need, leaving the estimator code path otherwise identical: `--rows m` (rows per token per site in the Cut phase; Boost keeps full cycles; estimates scaled by 512/m); `--row-assign {time,token}` (`time`: all tokens in an update use the same m rows, rotating across updates; `token`: token t uses rows offset by (t mod 512/m)·m, so every update covers the full basis across tokens); `--prior {none,identity}` (E7); a per-token row-count callback (E13). Also the probe primitive `reforward(site, delta) → per-token losses`, which re-runs from a site with other tokens' keys and values held at their clean values, as the chain does. It is shared with H8.

**H6. Diagnostics** (offline, autograd allowed, on H4 checkpoints, one sequence each unless stated):

1. `cos_to_autograd(estimator, ckpt)`: per matrix type and site; start from `check_relay.py`.
2. `credit_dump(ckpt)`: the exact per-token credit at each site (H8), the identity-path guess, their per-token cosine, and the variance ratio ‖g − p‖² / ‖g‖² per site (E7).
3. `deep_readout_check(ckpt, k)`: per token, 512 bf16 forward-difference probes launched at S1 through `reforward` at `chain_sigma`; the per-site changes, the loss changes, the condition number of each 512 × 512 system, and the cosine between its solution and the exact credit at Sk (E5). Use forward differences, not `jacrev`: with exact Jacobians the check is consistent by construction.
4. `snr_spectrum(ckpt, estimator)`: per block matrix, the momentum's singular directions and, along each, the signal-to-noise ratio of per-sequence gradients over 64 sequences, for exact credit and for synthesized partial-row estimates (E8).
5. `linearity_check(ckpt, site, sigma)`: the loss change under +v and −v at every site, and its distance from linear.
6. `synth_vs_real(ckpt)`: H8 synthesis against H5's real partial-cycle estimates with the same rows and signs; cosine per site.

**H7. Runner and results table.** `run.py --config configs/<name>.yaml` launches one run; `collect.py` gathers every `result.json` into one table (experiment, arm, seed, final val loss, wall-clock, git commit) with mean ± sd per arm. Nothing is reported that is not in this table.

**H8. Autograd emulation of Boost and Cut** (development only).

The fact it rests on: with other tokens' keys and values held at their clean values and each token's own query live (the record's `own_query`), every path from a site to a token's loss stays inside that token's row. So the gradient of the same network with every stream-derived input to the keys, the values and the value-embedding gate detached, queries and all parameter paths left live, is the quantity the chain measures at every site, computed for all sites at once. Attention's own parameters get their exact gradient from the credit at its output.

- **Cut mode.** In every block, detach the normed input to the key and value projections and the gate input (`ve_gate` reads the block input's first 32 channels); keep the query input, the value-embedding tables and every parameter path; one ordinary backward. A `--freeze-queries` flag also detaches the query input and reproduces record #9's Cut, without own queries.
- **Boost mode.** Detach each sublayer's input; sum the shared head's readout losses on the stream after every sublayer; one backward. Match the record's Boost class for which readouts train the head.
- **Synthesis mode.** Take the exact per-site credits g\_k (autograd.grad with respect to the nine site activations, on the cut graph); transform them by a probing scheme (partial rows with H5's sign and row assignment, control variates, per-token budgets, shaping); then form parameter gradients block-locally, with one autograd.grad call per block whose inputs are only that block's parameters, so transformed credit never leaks between blocks. In code indices (blocks 0 to 7; block i's mid stream is site S\_{i+1}): block i's attention parameters and `value_embeds[str(i)]` (which lives on the GPT module, not the block) from the credit at S\_{i+1}; block i's MLP from `resid_lambdas[i+1]` times the credit at S\_{i+2}, and block 7's MLP from the top-stream credit; the embedding table from S0; the head and mixing scalars from the cut graph.
- **`reforward`**, shared with H5, for anything that needs losses at finite distances (E5, E11).

Synthesis mode with the identity transform reproduces PR #40's core estimate. Cut mode's single backward differs in one place: block i's MLP also receives the own-query path through block i+1, which PR #40's out-credit `lam * chain` leaves out and 30M record #3 partly adds back with its "in-roots". Treat cut mode as a slightly more complete variant. Experiments that compare probing schemes use synthesis mode with the identity transform as their baseline, so baseline and arms share one code path; experiments that change the objective use cut mode, which is cheaper.

To first order a probe with pattern v at site k reads v · g\_k, so synthesized estimates match the chain's up to finite-difference, bf16 and route-truncation error (H6 item 6). H8 does not emulate the record's cross-token patches or its top-512 MLP routes (H8 uses all 2,048 units; the top-128 version cost 0.006 in the boost-cut ablation). H8 and full backprop therefore bracket the cross-token credit, and the record's patches close most of that gap (record #14 before its merge 3.703, backprop 3.708).

Validation before use: on a fixed checkpoint, per-matrix cosine of at least 0.99 between synthesis mode (identity) and H5 run with `future_draws` 0, `value_path`, `path_scalars`, input-key and gate probes off, and `route_k` 2,048. There is no published number for H8's exact configuration, so E1's runs become the reference.

## 4. Run plan

P1 is about 18 GPU-hours on one H100, P2 about 7.5 plus confirmations on the real chain; all estimates, to be replaced with measured times after E1. Diagnostics gate their experiments: where a diagnostic says skip, skip.

| Priority | Item | Needs | Harness | New runs | Est. GPU-h |
| --- | --- | --- | --- | --- | --- |
| P0 | H1 to H4, H7, H8, verification | none | none | none | build |
| P1 | E1 baselines | H1, H8 | proxy, H8 | 15 + 2 single | 2 |
| P1 | E2 smoothing | E1 | proxy | 9 | 1.5 |
| P1 | E3 population as local SGD | E1 | proxy | 15 | 2 |
| P1 | E4 measured step control | E1 | proxy | 9 | 1 |
| P1 | E6 where to put the missing rows | E1 | H8 synthesis | 18 | 3.5 |
| P1 | E7 identity-path control variate | E6, H6 item 2 | H8 synthesis | 6 | 1 |
| P1 | E8 signal-to-noise along singular directions | E6, H6 item 4 | proxy, H8 synthesis | 18 if built | 0.5 + 3 |
| P1 | E14 latent consistency (NextLat) | E1 | H8 cut | 18 | 2.5 |
| P1 | E15 Boost with a latent target | E1 | H8 cut | 6 | 1 |
| P2 | E10 per-token shaping | E1 | H8 synthesis | 6 | 1 |
| P2 | E11 per-token runway weights | E1 | H8 synthesis, `reforward` | 3 | 1 |
| P2 | E13 n-gram and copy budgets | E6 | H8 synthesis | 9 if built | 2 |
| P2 | E5 deep readout, diagnostic | H6 item 3 | offline | none | 0.5 |
| P2 | E16 does H8 rank architectures | E1 | proxy, H8 cut | 8 | 3 |
| P2 | Confirm each winner on the real chain | H5 | 1M slice | 2 to 3 per winner | 1 to 1.5 per winner |
| P3 | E9 logit readout, true Fisher | E8 | proxy | 6 | 1 |
| P3 | E5 deep readout, implementation | E5 diagnostic, maintainers' answer | H5 | 3 | 1.5 |
| P3 | E12 30M runs | winners | rented 8×H100 | as needed | rented |

## 5. Experiments

Each experiment states its question, the one change, its arms, its logs and its decision rule. Three seeds per arm (0, 1, 2), the same seeds across arms. A difference under 0.02 nats is not a result without all three seeds.

**E1. Baselines.** Arms, three seeds each: backprop proxy at 8 sequences per update (the anchor for 3.708) and at 4 (the upper end of the bracket for H8 experiments, same rates); H8 Boost → Cut in synthesis mode with the identity transform (PR #40's core) and in cut mode; H8 cut mode from step 0 with no Boost. Single runs: the identity-path estimator for a full 10M epoch, and the H5 early-stopped verification. Save H4 checkpoints from backprop at batch 8 and from H8 synthesis (identity). Expected: backprop about 3.71 (slightly above 3.708, because 32 sequences are now held out); H8 Boost → Cut between record #10's 3.896 and record #11's 3.777, since it has own queries and exact values but not #11's value path; cut-only higher; identity-path about 4.18. These runs are the references for every later H8 experiment.

**E2. Smoothing with exact antithetic pairs.** Question: does the gradient of a blurred loss train better than the gradient of the loss? Change, per update in the backprop proxy: for every token and site draw r\[t\] = s · RMS(h\[t\]) · √512 · u, with u uniform on the unit sphere, so each coordinate's RMS displacement is s times the token's activation RMS (the units of `chain_sigma`). Run forward and backward with +r added at all nine sites, again with −r, and average the two gradients. Odd-order terms cancel, leaving the gradient of the loss averaged over activation noise, to second order. Knob: s ∈ {0.03, 0.1, 0.3}; the record's probe step is 0.01, chosen to stay in the linear regime, so smaller values would be null by construction. Arms: 3 × 3 seeds. This is close to activation-noise training, a known regularizer; the question is whether it helps in this one-epoch, data-limited regime. Logs: train loss, final val loss and the gap between them. Success: some s beats E1's backprop at batch 8 by 0.02. Kill: none does, or train loss rises while val stays flat. If it succeeds, the forward-only version (each site's cycle run around its own displaced point) follows in H5.

**E3. Population as local SGD.** Question: does any rule over several weight copies beat one trajectory at equal steps and equal data? Setup in the backprop proxy: train to 20% (update 121 of 606), then branch into K = 4 copies. Each update's 8 sequences go 2 to each copy, so every copy makes all 485 remaining updates at local batch 2 with the baseline's rates. Forward and backward compute equal one run; optimizer compute and optimizer state are 4×. Each copy keeps its own optimizer state. Arms: (a) average weights and optimizer state every H updates, H ∈ {1, 10, 50}; (b) no averaging; every 50 updates, score the copies with H3 and overwrite the worst with the best; (c) no averaging until the end, then keep the average only if H3 scores it below the best copy. H = 1 is a sanity anchor near the baseline, though not identical to it because ANVIL's orthogonalization is nonlinear. Arms: 5 × 3 seeds. Logs: per-copy loss curves; every 10 updates, the H3 loss at each pairwise midpoint (a midpoint worse than both ends means the copies have left one connected basin). Calibration: [Lin, Stich, Patel and Jaggi](https://arxiv.org/abs/1808.07217) found that local SGD started after the early phase generalizes better than large-batch SGD at equal steps; our batch (8 × 2,048 tokens) is small, so the effect may not appear here. Success: any arm beats E1's backprop at batch 8 by 0.02. Follow-ups if so: K = 8 at local batch 1; per-copy rate multipliers under rule (b), with one knob, their spread.

**E4. Measured step control.** Question: can accept-or-reject on measured loss replace the hand-tuned warmdown? Change: after forming each update, evaluate 4 of H3's 32 held-out sequences (rotating) under the old and the new weights, a paired comparison costing 2 forwards on 4 sequences. Judge on held-out rows: not on the next training batch, which would then arguably feed two updates, and never on the batch that produced the step, which almost always improves. Arms: (v1, zero knobs) if the held-out loss rose, revert weights and optimizer state; (v2) v1 plus a global step multiplier that rises by a factor of 1.1 after an accept and falls by 0.87 after a reject, on top of the hand warmdown; (v2b) v2 with the warmdown removed, at a constant base rate. Both factors follow from one choice, a target acceptance of 0.6 (ln(1/0.87) / ln(1.1/0.87) = 0.59). If acceptance falls below the target late in training, the multiplier decays geometrically, and that decay is the measured warmdown. Arms: 3 × 3 seeds. Logs: acceptance rate per 50 updates; the effective rate against the hand schedule. Success: v2b matches E1's backprop at batch 8 within noise, meaning a recipe that retunes itself when the token count changes.

**E5. One launch, eight readings** (diagnostic P2, implementation P3). Goal: the credit at S1 to S8 from one 512-probe cycle launched at S1, instead of eight separate cycles. Why it can work: with other tokens' keys and values held, everything above site Sk depends only on the token's activations at Sk, on x0, and on held quantities. A probe at S1 leaves x0 untouched, so to first order its loss change is d\_j = Δ\_k^j · g\_k at every later site k, where Δ\_k^j is the change the probe has become at Sk, which the chain already computes in order to continue. For each token and site, 512 probes give a 512 × 512 linear system. S0 keeps its own cycle, because probes there also move x0, which enters every block directly. Saving on the stream cycles: per probe pattern, 16 MLP passes and 2 head evaluations instead of 44 and 9, about 2.75×. The key, gate, query and future probes are unchanged, so the saving on the whole chain is smaller; measure it. Diagnostic (H6 item 3, 20% and 100% checkpoints): per site, the median condition number and the median cosine of the solved credit to the exact credit, from bf16 forward differences. Decision: all sites at 0.95 or above → ask the maintainers, then implement; failures only at the top two or three sites → a hybrid with direct cycles there; most sites failing at 100% → drop. Rules: 30M record #3 already scores root moves at later measured hubs, under its own convention that "a decoded credit feeds only linear consumers". E5's solved credits would also feed queries and keys (through the softmax) and the MLP hidden units (through relu²). Either restrict them to linear consumers, which shrinks the saving, or ask (Section 7). Implementation in H5: tokens in chunks of 512, each system stored in bf16, solved with a ridge toward the identity-path guess at a fixed weight of 1e-3 times the mean diagonal of AᵀA.

**E6. Where to put the missing rows.** Question: with m = 512/K rows per token per site in the Cut phase, which assignment of rows works better? Arms in H8 synthesis mode: (a) `time`: all tokens in an update share the same m rows, rotating across updates; (b) `token`: token t uses rows offset by (t mod K)·m, so every update covers the full basis across tokens. K ∈ {4, 8, 16}, so m ∈ {128, 64, 32}. Baseline: E1's synthesis run with the identity transform (K = 1). Arms: 6 × 3 seeds. Why (a) is expected to lose: when every token's estimate lies in one m-dimensional subspace, each writer matrix's gradient has rank at most m on its output side; scaled by K it dominates the momentum, and the orthogonalization then inflates that low-rank term. Log the stable rank of the orthogonalization's input to see it directly. Both arms are unbiased: with random column signs, E\[v vᵀ\] = I for any row chosen independently of the signs, so any subset of rows scaled by 512/m is unbiased. The K updates of arm (a) do not add up to one exact cycle, since they see different sequences and different weights. Logs: per update, the cosine of the summed weight gradient to the identity-transform one. Success: an arm at K = 4 within 0.02 of K = 1, and at K = 16 within 0.05. Time the winner on the real chain (1M slice).

**E7. Measure only what the cheap guess gets wrong.** Question: does a control variate built from the identity-path guess cut the rows needed? The guess p at site k is the top-stream credit (one cycle through the head only, about 2% of the chain) handed down the residual path by the mixing scalars, as the speedrun records use it. Estimator: ĝ = p + (1/m) Σ\_{j∈S} (v\_j · (g − p)) v\_j, where v\_j · g is the probe reading and v\_j · p is computed, not measured. It is unbiased, with variance proportional to ‖g − p‖² instead of ‖g‖². Diagnostic first (H6 item 2, 0, 10, 20, 50 and 100% checkpoints): the variance ratio ‖g − p‖² / ‖g‖² per site; if it is above 0.5 from 50% on, skip the runs. Arms in H8 synthesis with token-assigned rows: prior on at m ∈ {64, 128}; the matching no-prior arms are E6's (b). Arms: 2 new × 3 seeds. Success: the prior at m = 64 within 0.02 of the full cycle. Precedent: record #7 used a credit memory as a control variate, with the probes still measuring the residual.

**E8. Signal-to-noise along singular directions** (diagnostic first). Question: does an orthogonalizing optimizer amplify estimator noise, and does scaling its steps by measured signal-to-noise fix that? The noisy quadratic model ([Dang, Wen and Malladi 2026](https://arxiv.org/abs/2610.08975)) says preconditioners that equalize directions amplify noise where the signal is weak; partial-row probing adds near-isotropic noise, which lands hardest there. Diagnostic (H6 item 4, 20% and 100% checkpoints): per block matrix, the signal-to-noise spectrum along the momentum's singular directions over 64 sequences, for exact credit and for E6's best assignment at K = 4. Build the intervention only if partial rows push a large share of the orthogonalization input's energy into directions with signal-to-noise below 1. Intervention: each update, take the momentum's singular directions (u\_i, v\_i) and the 8 per-sequence projections p\_{s,i} = u\_iᵀ G\_s v\_i (H2); with mean m\_i and variance s\_i² over the 8, set r\_i = max(0, m\_i² − s\_i²/8) / m\_i², an estimate of μ\_i² / (μ\_i² + σ\_i²/8) that removes the noise bias in m\_i²; replace the polar factor U Vᵀ with U diag(r) Vᵀ. ANVIL II applies a per-lane energy gain after the polar factor and renormalizes the Frobenius norm, which would undo the scaling, so all arms run with the lane gain off and without renormalization. Arms: control (r\_i = 1); SNR (r\_i as above); and a rate control (a uniform r equal to the SNR arm's Frobenius ratio in that update), so that a gain cannot come from a smaller step alone. Settings: the backprop proxy with exact gradients, and H8 synthesis at 8 sequences per update with E6's best assignment at K = 4. Arms: 3 × 2 settings × 3 seeds. Prediction: little difference with exact gradients, a clear gain with partial rows. Success: with partial rows, SNR scaling recovers at least half the loss the partial rows cost and beats the rate control. Cost: one SVD per block matrix per update, in development.

**E9. Logit readout and the true Fisher** (P3). Every probe already computes the change dz in all 4,096 logits. Keeping two more numbers per probe per token, the curvature term dzᵀ(diag(p) − ppᵀ)dz and the loss change for one target sampled from p, gives curvature and true-Fisher directions at no extra probe cost (a kernel change in H5; dz is never stored). Build only if E8's intervention shows that per-direction scaling matters with exact gradients. Proxy test: left-precondition each block matrix's momentum by (F + δI)^(−1/2) before orthogonalization, F the output-side Fisher from sampled-label gradients, δ fixed at 1e-3 · tr(F)/512. Arms: 2 × 3 seeds.

**E10. Per-token credit shaping.** Two fixed rules in H8 synthesis mode, no sweeps: (a) clip each token's credit norm at each site to 3 times that site's batch median; (b) keep only weight-update entries where at least 70% of tokens agree in sign, computed as sign(X)ᵀ sign(G) in one matrix multiply. Arms: 2 × 3 seeds, against E1's synthesis run with the identity transform. These are cheap controls for E3 and E11 as much as ideas in their own right.

**E11. Per-token runway weights.** Question: under the cut, each token's loss can be evaluated at any distance along its credit; does weighting tokens by how far their loss keeps falling help? Change, in H8 synthesis with `reforward`: for each token t and site k, move the token's activation along −g\_t / ‖g\_t‖ by d · RMS(h\_t) per coordinate (a move of length d · RMS(h\_t) · √512), for d ∈ {0.01, 0.02, 0.04, 0.08}; set α\_{t,k} to the d with the lowest loss, or 0 if none is below the clean loss; normalize α to mean 1 per site; weight token t's credit at site k by α\_{t,k} before forming weight gradients. No knobs beyond the fixed grid. Cost on the chain: 4 probes per site per token against 512. Arms: 1 × 3 seeds. Success: beats E1's synthesis run by 0.02.

**E12. 30M runs on a rented 8×H100 node** (P3). The submission base is now 30M record #3 (3.338, the live `train/val/run.py`), level with backprop (3.340). Run backprop with that record's exact optimizer and schedule on 30M (the README's 3.340 used "a few learning rates"), then port the proxy winners onto record #3's script. For the speedrun (22.9 s), port only what E6, E7 or E13 show cuts chain cost.

**E13. N-gram and copy-aware probe budgets.** Read speedrun record #3 first: its "difficulty-balanced row order" may already cover part of this. Build a streaming n-gram table, orders 1 to 4, with Witten–Bell interpolation (a proper probability with no free parameters), updated causally: for each batch, score every token's true next token under the counts so far, then insert the batch's n-grams; exclude the held-out sequences. A token is a copy if its preceding 3-gram occurred earlier in the same sequence followed by the same token. About fifty lines of numpy, milliseconds per update. Diagnostic first (H6, 20% and 100% checkpoints): credit energy per tier. Build only if the low tier holds under 10% of the energy with over 40% of the tokens. Rule: 512 rows if max(n-gram loss, model loss) > θ nats, else 64, each estimate scaled by 512/m. Knob: θ ∈ {2, 3, 4}. Arms: 3 × 3 seeds in H8 synthesis. On the real chain, two separate arms: the row rule alone, and future-token draws only on copy tokens alone. Using n-gram statistics is established: hashed n-gram tables are in track-3 records #4 and #5 and in modded-nanogpt ([PR #367](https://github.com/KellerJordan/modded-nanogpt/pull/367), [PR #380](https://github.com/KellerJordan/modded-nanogpt/pull/380)); here nothing enters the model.

**E14. Latent-consistency auxiliary objective (NextLat).** Following the [NextLat code](https://github.com/JaydenTeoh/NextLat): the latent is the final-norm output; a dynamics model (LayerNorm over the concatenation of the next token's embedding and the current latent, then a GELU MLP whose output is added to the current latent) predicts the next position's latent; the loss is smooth-L1 with β = 1, averaged over batch, positions and dimensions, masked at document boundaries. One departure: a single hidden layer of width 1,024 instead of NextLat's three layers, because a forward-only submission must train this MLP too, and one hidden layer needs one nonlinear step where more layers would need chaining. Use the same design in development. Fixed: horizon 1, no KL term. One knob, λ. Two structural binaries: the target (the live network, detached, as published; or the H4 EMA teacher, as BYOL, I-JEPA and [ProteinJEPA](https://arxiv.org/abs/2605.07554) use) and the onset (step 0, or the Boost → Cut handoff at 20%).

Why both binaries: self-predictive objectives train from random initialization in vision and RL, usually as the main signal and with an EMA teacher. Where a dense token loss exists from step 0, the evidence favors late onset: ProteinJEPA from scratch won 50, lost 41 and tied 23 against step-matched MLM (p = 0.20), so it starts the latent loss at step 20,000 with an 8,000-step ramp, while from an MLM-warmed model it won 76, lost 15 and tied 23 (p < 0.0001); NextLat's latent loss rises during training. Noise sits between: its token loss exists from step 0, but the measured gradient is poor early (cut cosine 0.68 at initialization).

Procedure in H8 cut mode: sweep λ ∈ {0.3, 1, 3} once (EMA target, onset 20%); then the 2 × 2 of target × onset at the best λ, which adds three cells. 18 runs. Report each result as the fraction of the gap between E1's cut-mode Boost → Cut and backprop at batch 4 that it closes. Logs: the cosine between the auxiliary gradient and the next-token gradient at the top state every 50 updates; the effective rank of top-layer states and site credits at 20% and 100%. Confirm the best arm on the real chain with and without future probes. Cost on the chain: the dynamics MLP runs once per probe at every site, about 13% more chain compute, plus one teacher forward per update. Prior against: at 1.3B parameters on 100B tokens NextLat's perplexity was slightly worse (10.88 against 10.52 on FineWeb-Edu), and the per-element mean makes the term weak by construction. Success: closes at least half the gap, or beats the matching baseline by 0.02.

**E15. Boost with a latent target.** Question: is Boost's logit-lens objective a crutch? Change: a per-sublayer predictor (one GELU hidden layer, 512 → 1,024 → 512, discarded at the end; one nonlinear step, so a forward-only version can train it) maps the stream after the sublayer to the EMA teacher's final latent for the same token, with loss 1 − cosine (cosine beat MSE in ProteinJEPA); skip the last sublayer. The target is soft and rich where the next-token label is one-hot, and the token loss anchors it so it cannot collapse. [Greedy InfoMax](https://arxiv.org/abs/1905.11786) trained stacks of modules with only local predictive losses and no gradient between them; JEPA-only training collapses downstream in ProteinJEPA, and Boost alone stalls at 4.062, so this concerns the warm start only. Arms in H8 (Boost for the first 20%, then cut mode): (a) logit lens, from E1; (b) logit lens plus the latent target, λ = 1; (c) the latent target alone, which tests whether a self-bootstrapped target can carry the early phase as it does in vision; (d) no Boost, from E1. New: 2 arms × 3 seeds; sweep λ ∈ {0.3, 1, 3} only if (b) beats (a). Logs: per-sublayer readout and latent losses through Boost; the cut-gradient cosine to the true gradient at the handoff, the quantity Boost exists to raise. Success: (b) or (c) beats (a) by 0.02, or raises the handoff cosine at equal final loss.

**E16. Does H8 rank architectures the way the real chain does?** (P2). H8 is ordinary autograd with detaches, so it runs on any architecture. If it ranks architectures as the forward-only chain does, track-3 ideas can be screened at backprop speed. Arms on the 10M set: the fixed model and track-3 record #5's shape (12 × 1,024 with hashed n-gram value embeddings), each under backprop and under H8 cut-mode Boost → Cut; 4 arms × 2 seeds. Compare H8's ranking and gap with the leaderboard's (3.211 against 3.338 on 30M). Success: H8 reproduces the order; then use it to screen track-3 changes before any real-chain run.

## 6. Reporting conventions and pitfalls

**Statistics.** Seed-to-seed standard deviation on the 10M set is about 0.006 (Relay), 0.009 (backprop) and 0.013 (Boost → Cut). Three seeds per arm, the same seeds across arms, mean ± sd in the results table. Report final losses from complete runs only; the warmdown moves the loss in the last 40%.

**Scoring.** The number that matters is `utils/eval.py` on the final weights. In-run decisions use H3. A run that reads `val.pt` for any decision is discarded, not footnoted.

**Attribution.** One change per arm. The H5 flags and H8's modes exist so that the estimator code path is otherwise identical across arms.

**Logs every run must have.** `config.json` with the git commit and any pinned upstream commit; `metrics.jsonl`; `result.json`; the H4 checkpoints; the experiment-specific series named in Section 5; and one plot script per experiment that reads only these files.

**Pitfalls.**

1. Held-out leakage: `backprop_preset.py` trains on all 4,880 sequences. Hold out 32 (H1), or H3 scores what the model trained on.
2. The live `train/val/run.py` and `train/speedrun/run.py` are 30M records. Develop from `archive/10M/`; submit on 30M.
3. The 10M record runs Boost for 20% of updates at 4 sequences per update; the 30% figure in older write-ups is record #9's.
4. Chaining: the README now says credit cannot be chained back through the network block by block, and the chained entries (CutBI, PR #17; Relay, PR #20) were never merged. E5 reuses one measurement at several sites; ask first.
5. A learned predictor of credit is not a measurement; use one only as a control variate with probes measuring the residual (record #7, E7). Training-time auxiliary networks (E14, E15) must themselves be trainable with one nonlinear step.
6. The embedding reaches every block through x0. Any argument that "everything above site k depends only on site k" fails for probes at S0.
7. Partial cycles are unbiased because of the random column signs. Rows shared by all tokens make writer gradients low-rank, which orthogonalization inflates (E6).
8. Never judge a step on the batch that produced it; judge on held-out rows, paired (E4).
9. ANVIL II's lane gain and Frobenius renormalization re-equalize any per-direction scaling applied before them (E8).
10. H8: synthesis mode with the identity transform is PR #40's core; cut mode also sends the own-query path into the MLP writers. Validate (H6 item 6) before trusting fine differences.
11. bf16 finite differences: the chain's probe step (1% of each token's RMS) balances linearity against rounding. E5 and E9 push on this; run H6 item 5 first.
12. Identity-path estimators plateau near 4.18 on 10M; they are a development proxy for speed ideas, not a route to the validation record.
13. PR descriptions report seed 0, and research copies and official logs of one configuration differ by 0.01 to 0.02. Treat anything under 0.02 as noise until three seeds say otherwise.

## 7. Questions for the maintainers, and references

**One GitHub issue on qlabs-eng/noise, posted before E5's implementation or any submission that uses E14 or E15.** Suggested text:

- Deep readout. A probe launched at a block's mid stream passes through every later block, and the chain already computes what the perturbation has become at each later site in order to continue. Is it within the rules to read the single loss change at the top as a first-order measurement at every later site, against the perturbation as it stood there, and solve per site for the credit? No signal takes a step through a nonlinearity; the per-site perturbations are computed forward. 30M record #3 already scores root moves at later hubs under the convention that a decoded credit feeds only linear consumers; may a credit decoded this way also take the one allowed nonlinear step (into queries, keys or MLP hidden units)?
- Training-time auxiliary state. A one-hidden-layer dynamics MLP (E14), per-block predictors and an EMA teacher (E15), all discarded before scoring and all trained with one nonlinear step. Acceptable, as record #7's associative credit memory was?

**Local questions.**

- [ ] Which readouts train the shared head during Boost in the 10M record? Match it in H8's boost mode.
- [ ] Does processing the two-GPUs-per-sequence layout sequentially in the H5 port change any reduction order enough to move per-step losses? The H5 verification at steps 2, 10 and 100 answers it.

**References.**

| Source | What it supplies |
| --- | --- |
| [Noise README](https://github.com/qlabs-eng/noise) and [10M archive](https://github.com/qlabs-eng/noise/blob/main/archive/10M/README.md) | Rules, tracks, records on 30M and 10M, the backprop baselines |
| [PR #40](https://github.com/qlabs-eng/noise/pull/40), code in `archive/10M/train/val/run.py` | The 10M record: own queries, value path, future-token probes, end merge, ablations |
| [PR #11](https://github.com/qlabs-eng/noise/pull/11) | Boost and Cut; per-site cosine to autograd of 0.9996 to 0.9999 |
| [PR #16](https://github.com/qlabs-eng/noise/pull/16), [PR #26](https://github.com/qlabs-eng/noise/pull/26) | Identity-path credit; exact values |
| [PR #20](https://github.com/qlabs-eng/noise/pull/20) | Backprop with the record's optimizer (3.708; 3.712 ± 0.009); `check_relay.py`; the chained estimator that is no longer allowed |
| [30M record #3 script](https://github.com/qlabs-eng/noise/blob/6f0c450ed52d52dd2fefd2a8ba485d92f2b37e92/train/val/run.py) | The submission base; the true-path readout and the linear-consumer convention |
| [spicylemonade/boost-cut](https://github.com/spicylemonade/boost-cut) | `backprop_preset.py`, test split, ledgers |
| [Dust paper](https://qlabs.sh/research/dust) | The original estimator and the smoothing hypothesis |
| [Dang, Wen and Malladi 2026](https://arxiv.org/abs/2610.08975); [Zhang et al. 2019](https://arxiv.org/abs/1907.04164) | The noisy quadratic model: curvature-to-noise per direction decides how optimizers rank (E8) |
| [Lin, Stich, Patel and Jaggi 2020](https://arxiv.org/abs/1808.07217) | Local SGD after the early phase (E3) |
| [NextLat paper](https://arxiv.org/abs/2511.05963) and [code](https://github.com/JaydenTeoh/NextLat) | E14's objective and implementation |
| [ProteinJEPA](https://arxiv.org/abs/2605.07554); [LLM-JEPA](https://arxiv.org/abs/2509.14252) | Latent prediction beside a token loss: onset, EMA teacher, cosine loss (E14, E15) |
| [Greedy InfoMax](https://arxiv.org/abs/1905.11786); [Tian, Chen and Ganguli 2021](https://arxiv.org/abs/2102.06810) | Local predictive losses without end-to-end gradients; why self-prediction avoids collapse (E15) |
| [modded-nanogpt PR #360](https://github.com/KellerJordan/modded-nanogpt/pull/360), [#367](https://github.com/KellerJordan/modded-nanogpt/pull/367), [#380](https://github.com/KellerJordan/modded-nanogpt/pull/380) | ANVIL II; retrieval hints and n-gram chains (E13) |
| [KronZO](https://www.gerad.ca/en/papers/G-2025-44) | Pretraining gradients spread across many directions: test before assuming low rank |

## 8. Changes from v1

For an agent already working from v1. Experiment numbers keep their meaning; E16 is new.

| Area | v1 | v2 | Why |
| --- | --- | --- | --- |
| Held-out data | Last 32 sequences held out | Held out after the SEED + 1 permutation, and removed from the proxy's training set | The preset trains on all 4,880, so H3 scored training data |
| Data and sources | Current scripts and 30M data | `archive/10M/` scripts and data layout; live scripts are 30M records | The live scripts moved on; the archive loader rejects 30M data |
| Record recipe | Boost 30% | Boost 20%, 4 sequences per update in both phases | That is PR #40's configuration; 30% was record #9's |
| H8 (new) | none | Autograd emulation of Boost and Cut, with a synthesis mode | Most experiments now run at backprop speed |
| H8 cut definition | not applicable | Detach key, value and gate inputs; keep queries live | The record lets each token's own query reach later attention |
| Bracket | Records #11 and #12 as the bound | H8 against backprop; the record's patches close most of the gap | #11 already includes the value path |
| E2 | τ of 1 to 10 times the probe step | Per-coordinate std of 0.03 to 0.3 of each token's RMS | v1's grid stayed in the linear regime and would be null |
| E3 | Copies at batch 4, a quarter of the updates each | Local SGD: copies at local batch 2, every copy makes every update | Equal steps and equal data; the averaging period is the only knob |
| E4 | Judged on the producing batch; factors 1.05 and 0.8 | Judged on held-out rows, paired; factors 1.1 and 0.87 | Same-batch judging nearly always accepts; v1's factors give acceptance 0.82, not 0.6 |
| E5 | Launch at the embedding; 5.5× | Launch at S1, S0 keeps its cycle; 2.75× on the stream cycles only | Probes at S0 also move x0, which enters every block directly |
| E5 rules | One question | Also the linear-consumer convention of 30M record #3 | Solved credits would feed nonlinear consumers |
| E6 | Signs redrawn every step; "exact over K steps" | Time against token assignment, K of 4, 8 and 16; log stable rank | Different updates see different data; shared rows make low-rank gradients |
| E7 | A row ramp that never used the guess | A control variate on the identity-path guess | Variance then scales with what the guess gets wrong |
| E8 | An ad hoc formula applied before orthogonalization | Diagnostic first; per-update SNR with the lane gain off and a rate control | Basis-indexed averages were ill-defined; ANVIL's later stages undo earlier scaling |
| E10, E11 | Sweeps; E11 used a least-squares fit | Single fixed rules; E11 weights tokens only | One change per arm |
| E13 | Stupid backoff | Witten–Bell; row rule and future-draw rule as separate arms; read speedrun #3 first | Backoff scores are not probabilities; v1 changed two things at once |
| E14 | Three-layer dynamics MLP; 3% cost; 21 runs | One hidden layer; about 13%; 18 runs | A submission must train the MLP with one nonlinear step |
| E16 (new) | none | Does H8 rank architectures like the real chain? | Would let track-3 ideas be screened at backprop speed |
| Leaderboard | 30M record 3.409 | 3.338 (level with backprop), speedrun 22.9 s, track 3 at 3.211 | The leaderboard moved this week |
