# Noise Forward-Only Research Harness: Engineering Handoff

Oct 28, 2026 · @Alex

## 1. Purpose and ground rules

Build a one-GPU research harness for the Noise benchmark (pretraining a fixed 37.7M-parameter transformer without backpropagation), then run the experiments in Section 4. Every experiment is a yes/no question with at most one tunable knob, so that what works and what fails is readable from the logs. Engineering complexity is acceptable; hyperparameter sprawl is not.

**The benchmark.** Fixed model ([`utils/model.py`](https://github.com/qlabs-eng/noise/blob/main/utils/model.py)), fixed data (FineWeb, BPE-4096 tokenizer, sequences of 2,048 tokens), one epoch, score = mean next-token cross-entropy on `data/val.pt` (1,114,112 predictions) via `utils/eval.py`. Three tracks: lowest validation loss (no time limit), fastest time to validation loss 3.5 on the 30M-token set, and open architecture. Full rules and records: [qlabs-eng/noise](https://github.com/qlabs-eng/noise).

**Rules that constrain every submitted run** (paraphrased from the README; read the original before submitting anything):

1. No autograd in any mode and no hand-written backward pass. Every error signal is measured against a loss by running the model forward. A measured signal may take at most one step through a nonlinear part of the model; steps through linear maps do not count; a signal that has taken a step cannot take another. Credit cannot be chained back block by block. Entries are judged by the spirit of this rule.
2. Final weights must load into `utils/model.py` with a strict state-dict match (tracks 1 and 2).
3. Train only on `data/train.pt`, at most one epoch: each training sequence feeds at most one update. Batch size and order are free. Sequences that feed no update may be used for scoring decisions (the current record holds out 32 for its end merge).
4. No checkpoint selection on validation loss. Score the final weights only.
5. Timing (speedrun) runs from the first update to the last, excluding validation; an untimed warmup on random tokens may compile kernels; seed 0.

**What is legal in development but not in a submission.** Autograd is allowed for two things here: as a stand-in for the forward-only estimator while testing ideas that are not about the estimator (populations, smoothing, step control, loss shaping), and as an offline diagnostic (cosine between an estimator and the true gradient on a checkpoint). The Relay PR author published [`check_relay.py`](https://github.com/erodola/noise/releases/download/relay-track1-v1/check_relay.py) for the second use. Nothing autograd touches may appear in a submitted training script.

**Why the backprop stand-in is faithful.** On the archived 10M-token set, the record's forward-only estimator trained to 3.703 before its end merge; backprop with the same model, optimizer, schedule and data order reached 3.708 (seed 0; 3.712 ± 0.009 over three seeds). The estimator already matches the true gradient's training outcome, so experiments about what to do with a gradient transfer from the proxy to the real estimator.

**Two facts that set priorities.** First, the cost of the forward-only estimator is per token probed, not per step: the record's Cut chain costs roughly 400 forward-pass equivalents per update against backprop's 3. Second, on 30M tokens the record (3.409) trails backprop with the same optimizer (3.340) by 0.07 nats after tying on 10M; nobody has retuned the record for 30M, and nobody has rescaled it to stop at the 3.5 speedrun target.

## 2. Environment and setup (one GPU)

Target: one H100 80 GB (a 40 GB card works with batch 4). Linux, bf16, [uv](https://docs.astral.sh/uv/). The model is small (75 MB in bf16), so memory is never the constraint; wall-clock is.

**Repositories to clone.**

| Repo | What you need from it |
| --- | --- |
| [qlabs-eng/noise](https://github.com/qlabs-eng/noise) | Benchmark: `utils/model.py`, `utils/eval.py`, `utils/prepare_data.py`, tokenizer, current scripts `train/val/run.py` (track 1), `train/speedrun/run.py` (track 2), `train/arch/run.py` + `model.py` (track 3), `train/val/dust.py`. The 10M-era scripts and records are in `archive/10M/`. |
| [spicylemonade/boost-cut](https://github.com/spicylemonade/boost-cut) | `noise/analysis/backprop_preset.py` (commit `539386e`): backprop with the record's model, init, ANVIL II + Adam, rates, schedule and data order. This is the proxy baseline. Also `noise/analysis/prepare_test.py` (a held-out test split, the next 544 rows of the FineWeb stream after the validation pool) and `ledger/` with every run's config and result. |
| [erodola/noise release `relay-track1-v1`](https://github.com/erodola/noise/releases/tag/relay-track1-v1) | `check_relay.py`: cosine between an estimator's weight gradient and autograd, per matrix type, on one sequence. Starting point for the diagnostic toolkit. |
| [KellerJordan/modded-nanogpt PR #360](https://github.com/KellerJordan/modded-nanogpt/pull/360) | Reference implementation of ANVIL II if the Noise copy needs to be understood or modified. |

**Data.** `uv sync && uv run utils/prepare_data.py` in the Noise repo streams FineWeb `sample-10BT`, tokenizes, and writes `data/train.pt` (30M tokens: 14,648 sequences of 2,048 predictions) and `data/val.pt`, both hash-checked. The first 4,880 sequences of `train.pt` are the archived 10M set; the first 488 are a 1M slice used for fast forward-only development. All proxy experiments run on the 10M set (sequences 0 to 4,879) unless stated.

**Model facts you will need.** 8 blocks, width 512, 8 heads of 64, MLP 2,048 units with ReLU-squared, RMSNorm, rotary positions, sliding attention windows (pattern SSSL), vocabulary 4,096, logit softcap 15 tanh(z/15), two learned mixing scalars per block, value embeddings on every other block behind a sigmoid gate. Read `utils/model.py` before touching anything; the probing sites are the raw embedding and each block's mid stream (after attention, before the MLP).

**Expected single-GPU times** (extrapolated from published 8×H100 numbers by multiplying by 8; measure and replace):

| Run | 8×H100 (published) | One H100 (estimate) |
| --- | --- | --- |
| Backprop proxy, 10M, 610 updates of 8 sequences | 38 s | about 5 min |
| Identity-path forward-only, 10M full epoch (PR #26 code) | 47 s | about 6 min |
| Identity-path with the later fused kernels (PR #39) | 2.5 s to reach 4.2 | under 1 min |
| Record Cut chain, 10M full epoch (PR #40) | 35 min | about 4.5 h |
| Record Cut chain, 1M slice (488 sequences) | - | about 30 min |
| Record Cut chain, 30M | 1 h 43 min | about 14 h (do not plan on this) |

**Verification before any experiment.**

1. `uv run utils/eval.py` on the published record checkpoint from the boost-cut release (`track1_boostcut_final.pt`, sha256 `499dff9f...`) must print 3.94481. This proves data and scorer match.
2. Run `backprop_preset.py` on the 10M set, seed 0, single GPU. Expect final validation loss within 0.02 of 3.708. If the script assumes `torchrun` with 8 processes, adapt to one process with gradient accumulation so the effective batch stays 8 sequences; keep the data order.
3. Run the identity-path speedrun script for a full 10M epoch on one GPU. Expect about 4.18. The 8-GPU scripts put one sequence per GPU; on one GPU, loop over sequences and accumulate. Record the wall-clock.

If any of the three disagrees by more than seed noise (0.01), stop and find out why before building on it.

## 3. Shared infrastructure (build first)

Everything in Section 4 runs on these seven pieces. Build them in this order; H1 to H4 and H7 are a day, H5 and H6 are two or three more.

**H1. Single-GPU proxy trainer.** `backprop_preset.py` adapted to one process: same model, init, optimizer (ANVIL II on block matrices, Adam elsewhere), rates, warmdown, and the benchmark's data order; effective batch 8 sequences by accumulation. Flags: `--seed`, `--sequences` (how many of the 10M set to use), `--batch` (sequences per update), `--log-dir`. It writes `config.json`, `metrics.jsonl` (per update: step, train loss, step time, effective learning rates) and `result.json` (final official validation loss from `utils/eval.py`, wall-clock), matching the benchmark's log layout so results are comparable with the ledgers.

**H2. Per-sequence gradients.** A hook that keeps each sequence's gradient separately before averaging into the update (four or eight tensors per matrix per update, in bf16, discarded after use). Several experiments need the across-sequence spread of the gradient along a direction, which is the noise estimate in E8.

**H3. Held-out scorer.** Reserve the last 32 sequences of whatever training range is in use; they never feed an update. `score(weights)` returns their mean loss in one forward pass. Every in-run decision (which copy to keep, whether a merge helped, whether to undo a step) uses this scorer and never `val.pt`. Log every call.

**H4. Checkpoints at fixed fractions.** Save weights at 0%, 10%, 20%, 50% and 100% of every run (names `ckpt_00.pt` ... `ckpt_100.pt`). The diagnostics in H6 run on these. Also keep nine exponential moving averages of the weights from 60% onward, every 5 updates, with decay weights 0.5 down to 0.01 on the newest, exactly as the record's end merge does, so merge experiments reuse them.

**H5. Single-GPU ports of the forward-only estimators.** Two scripts, both with `--sequences` and `--batch`:

- The identity-path estimator (track 2 lineage; start from the PR #26 PyTorch version, not the JAX one). One Hadamard cycle of 512 probes at the top stream per token, credit handed down the residual mixing scalars, one local step per block. Fast enough to iterate on in minutes.
- The record's Cut chain (track 1, `archive/10M/` script of PR #40, or the current `train/val/run.py`): Boost phase, then Cut with the full 512-row cycle at nine sites, future-token probes, ANVIL II, end merge. Port the two-GPUs-per-sequence layout to one GPU by processing the two halves sequentially. Verify on the 1M slice that the per-step losses match the published `metrics.jsonl` for the same steps (the data order is fixed, so they should agree to a few decimals).

Two flags both scripts must expose because E6 and E7 use them: `--rows` (Hadamard rows per token per site, default 512; per the proposal the existing `probe()` already accepts a row count and `chain_credits` always uses the full basis, so the change is in the chain) and `--row-offset-per-step` (rotate which rows are used each update). Keep the estimator code path identical otherwise, so a change in loss is attributable to the row count alone.

**H6. Diagnostic toolkit** (offline, autograd allowed, runs on H4 checkpoints, one sequence each):

1. `cos_to_autograd(estimator, ckpt)`: per matrix type and per site, the cosine between the estimator's weight gradient and autograd's, starting from `check_relay.py`.
2. `credit_dump(ckpt, site)`: the true per-token credit at a site (autograd), shape 2,048 × 512, plus the identity-path guess at that site, plus the per-token cosine between them. Used by E7 to decide whether a cheap prior exists.
3. `token_map_svd(ckpt, k)`: for each token, with attention outputs frozen at their clean values, the Jacobian of the map from the embedding to site k (a 512 × 512 matrix per token; compute with `torch.func.jacrev` on the per-token function), and its singular values. Used by E5.
4. `direction_stats(ckpt, directions)`: for each given weight-space direction, the curvature along it (autograd Hessian-vector product) and the across-sequence variance of the gradient along it (from H2). Used by E8.
5. `linearity_check(ckpt, site, sigma)`: loss change under +v and −v for a probe of size sigma, at every site, reporting how far the response is from linear. Sets the probe size for E5 and E9.

**H7. Experiment runner and results table.** `run.py --config configs/<name>.yaml` launches one run; `collect.py` reads every `result.json` into one table (experiment, variant, seed, final val loss, held-out curve summary, wall-clock, git commit) and prints mean ± standard deviation per variant. Every experiment below is a set of config files. Never report a number that is not in this table.

## 4. Experiments

Dependencies: E1 first. E2, E3, E4 and E10 need only H1 to H4 and H7 and can start on day two. E5 and E8 need H6. E6, E7, E9 and E11 need H5. Run them in parallel on the one GPU by queueing configs; nothing here waits on anything except its listed dependency. Seeds 0, 1, 2 on every reported cell; a difference under 0.02 nats with fewer than three seeds is not a result.

**E1. Baselines.** Backprop proxy on the 10M set at batch 8 (the record's reference) and batch 4 (the record's own batch), three seeds each. Identity-path forward-only, full 10M epoch, seed 0. Cut chain on the 1M slice, seed 0, with its `metrics.jsonl` checked against the published log. Save H4 checkpoints from the batch-4 backprop run; they are the diagnostic checkpoints for E5, E7 and E8. Report: the six backprop numbers (expect about 3.71 ± 0.01), the identity-path number (about 4.18), wall-clocks.

**E2. Smoothing with exact antithetic pairs.** Goal: does the gradient of a blurred loss train better than the gradient of the loss? Procedure, per update: draw for every token t and every probing site a random direction r\[t\] of fixed length τ · RMS(activation at that site); run the forward with +r added at all sites and backprop; run it again with −r and backprop; average the two gradients. The ± pair cancels odd-order error, so the average is the gradient of the loss blurred at width τ. Knob: τ ∈ {1, 3, 10} in units of the record's probe step (about 1% of RMS; read `chain_sigma` from the record script). Runs: 3 values × 3 seeds, each about twice a baseline (two forwards). Logs: train loss, final val loss, and the gap between them. Success: any τ beats the batch-8 baseline by more than 0.02 on final val loss. Kill: none does, or train loss rises while val does not fall (a pure regularizer). If success, the follow-up is the cheap version (token 2i gets +r, token 2i+1 gets −r, one forward) and then the forward-only version (run the cycle around the displaced point, E7's harness).

**E3. Population on split data.** Goal: does any exploration rule over several weight copies beat one trajectory at equal compute? Procedure: run the batch-4 baseline to 20% of the epoch (the branch point; copies branched from initialization do not average well). Branch into K = 4 copies. Each copy receives every fourth sequence of the remaining data in the benchmark's order, so the four together use each sequence once and total compute equals one run. Copies take turns on the GPU. Three rules, one per run: (a) average all copies every N updates and continue from the average; (b) every N updates, score all copies with H3, overwrite the worst with the best; (c) run independently to the end, then average the four and keep the average only if H3 scores it below the best copy. Knobs: N ∈ {10, 50} for (a) and (b); none for (c). Runs: 5 variants × 3 seeds. Logs per copy: loss curve; every N updates, the H3 score of each pairwise midpoint between copies (a midpoint worse than both ends means the copies have left one connected basin). Calibration: (a) should land near a single run at batch 16, which the records found worse than batch 4. Success: any variant beats the batch-4 baseline by 0.02. Kill: none does and midpoints show no barrier, meaning the copies never explored anything the single run did not. Follow-up if success: K = 8, selection over learning-rate multipliers per copy (one knob, the multiplier spread).

**E4. Accept-or-reject steps.** Goal: can measured step control replace the hand-tuned warmdown? Version one, zero knobs: after forming each update, evaluate the batch loss at the new weights with one forward; if it rose, undo the step. Version two, one knob: scale a global step multiplier up by 1.05 after an accept and down by 0.8 after a reject, targeting an acceptance rate near 0.6; the hand warmdown stays in place underneath. Runs: 2 versions × 3 seeds plus one run of version two with the warmdown removed (constant base rate). Logs: acceptance rate per 50 updates, effective rate over time plotted against the hand schedule. Success: version two without the warmdown matches the tuned baseline within noise. That is the point: a recipe that retunes itself when the organizers change the token count.

**E5. One launch, nine readings: diagnostic, then implementation.** Goal: recover the credit at all nine sites from one 512-probe cycle launched at the embedding. Diagnostic first (H6 items 3 and 5, no training): on the 20% and 100% checkpoints, one sequence, for each site k and token t, the singular values of the embedding-to-site-k Jacobian with attention frozen, and the cosine between the true credit at site k and the credit recovered by solving the 512 × 512 system (rows = the probe's change at site k, right side = the loss changes), with a ridge toward the identity-path guess. Report a 9 × 2 table of median cosine and median condition number. Decision rule: if cosine ≥ 0.95 at all sites, implement the full version; if it fails only at the top two or three sites, implement the hybrid (direct cycles there, solve below); if it fails at most sites on the 100% checkpoint, drop it. Implementation (in the H5 Cut port): process tokens in chunks of 512, accumulate `AᵀA` and `Aᵀd` per token per site in bf16 during the chain's recomputation (the chain already computes the per-site changes; do not store them), solve per chunk, discard. Memory per chunk about 2.5 GB. Then one 1M-slice run against the unchanged chain: equal loss at lower time is success. Before the implementation, post the one-paragraph rules question in Section 6 to the maintainers.

**E6. The cycle spread over time.** Goal: K× cheaper probing by rotating through the 512 Hadamard rows across K updates and letting momentum stitch the readings. Procedure: `--rows 512/K --row-offset-per-step`, so update s uses rows \[(s mod K)·512/K, …); scale the estimate by K so it is unbiased; momentum unchanged. Start on the identity-path script (minutes per run), then the Cut chain on the 1M slice. Knob: K ∈ {1, 4, 16}. Runs: 3 × 3 seeds per script. Logs: every 50 updates, one full 512-row measurement on one sequence and its cosine with the current momentum estimate (the drift check). Success: K = 4 within 0.02 of K = 1 on final loss; K = 16 within 0.05. Kill: K = 4 already loses more than 0.02. The adaptive follow-up (shrink K when the drift cosine drops below 0.9) is one knob and only worth building if K = 4 passes and K = 16 fails.

**E7. Fewer rows early, all rows late.** Goal: cut probes while the identity-path guess is good. Diagnostic first (H6 item 2): per-token cosine between the identity-path guess and the true credit at each site on the 0%, 10%, 20%, 50% and 100% checkpoints. If the median cosine is above 0.9 at 20% and below 0.7 at 100%, there is a saving to take. Procedure: a linear ramp of `--rows` from m₀ at the start to 512 at a fraction f of training, with each update's rows chosen complementary across tokens (token t uses rows offset by t mod (512/m)) so every row is covered in every update. Knobs: m₀ ∈ {64, 128}, f ∈ {0.3, 0.6}. Runs: 4 × 2 seeds on the 1M-slice Cut chain. Success: equal loss at lower time. Only if the best ramp differs between seeds, add the self-check: 8 held-out rows per token per site, compare their measured loss changes with the prediction from the current estimate, and raise `--rows` when the relative error exceeds a threshold (one knob).

**E8. Steps by curvature-to-noise.** Goal: set each direction's step from its measured curvature h and noise c, the quantity the noisy quadratic model says decides which optimizer wins (Dang, Wen & Malladi 2026). Diagnostic first (H6 item 4 on the E1 checkpoints): for the top 16 singular directions of each matrix's pending update, h by Hessian-vector product and c as the across-sequence variance of the gradient's projection (H2). Report the distribution of h²/c per matrix type. Implementation, proxy first: before ANVIL's orthogonalization, scale the update's component along direction u by 1 / (1 + c\_u / (B h\_u² λ)) with λ the one knob; directions outside the tracked 16 are left alone. Then the real test: E7's 128-row partial cycle with ANVIL unchanged versus with this damping, on the 1M slice. Success: the damped version recovers most of the loss the partial cycle gave up. In the forward-only version, h along u comes from a ± probe pair per direction per token (about 6% more probes for 16 directions per matrix).

**E9. Read the logits, not the loss.** Goal: get curvature and a true-Fisher preconditioner from the probes already taken. Kernel change in the Cut port: where each probe computes the loss from the perturbed logits z + Δz, also emit the curvature term Δzᵀ(diag(p) − ppᵀ)Δz with p = softmax(z) from the clean pass, and the loss change for one target sampled from p (seeded per token so it is the same across the 512 probes). Three scalars per probe per token instead of one; `Δz` is never stored. Proxy test first, since it needs no kernel: replace ANVIL's implicit whitening on the block matrices with a damped true-Fisher left factor computed from sampled-label gradients via autograd (the K-FAC output-side factor), one knob (damping), three seeds. Success: beats the batch-4 baseline by 0.02. If the proxy test fails, skip the kernel work.

**E10. Per-token shaping.** Two one-line changes to how per-token gradients are summed, proxy only. (a) Cap each token's credit norm at the 95th percentile across the batch before the outer product. (b) Keep only update components where at least 70% of tokens agree in sign; sweep the threshold over {0.6, 0.7, 0.8}. Three seeds each. These are cheap controls for E3 and E11 as much as ideas in their own right.

**E11. Per-token targets instead of per-token slopes.** Goal: use the fact that under the cut each token's loss can be evaluated at any distance, not just infinitesimally. Procedure, in the Cut port: after the cycle gives token t's credit g\[t\] at a site, run four extra probes along −g\[t\] at 2, 4, 8 and 16 times the probe step and record where the token's loss stops falling (its runway). Set the token's target as the activation at that distance. Replace the outer-product sum with a damped least-squares fit of the writing matrix to the targets (one small solve per matrix per update). Knob: the damping. Runs on the 1M slice, three seeds, against the unchanged chain. Success: lower loss at equal probes. Cost: 4 extra probes against 512 per site.

**E12. When 8 GPUs are available.** Three things nobody has done on 30M: backprop with the record's exact optimizer and schedule (the fair baseline; the README's 3.340 used a few learning rates); the record re-tuned for 30M (rates, future-draw count, Boost fraction); and the record's warmdown defined on about 60% of the sequences to stop at the 3.5 speedrun target, which is probably an hour and a record for a config change. Whichever of E2 to E11 won on the proxy ports to the chain here.

## 5. Reporting conventions and pitfalls

**Statistics.** Seed-to-seed standard deviation on the 10M set is about 0.006 (Relay), 0.009 (backprop), 0.013 (Boost → Cut). Three seeds per cell, the same seeds across variants, mean ± standard deviation in the results table. Report final losses from complete runs only: the warmdown lowers the loss in the last 40%, so a mid-run checkpoint of one variant against the end of another says nothing.

**Scoring.** The number that matters is `utils/eval.py` on the final weights. Decisions inside a run use the H3 held-out training sequences. If a decision ever reads `val.pt`, the run is invalid under rule 4 and must be discarded, not reported with a footnote.

**Attribution.** One change per variant. If a run changes two things, it goes in the table as two things and tells you nothing about either. The row-count flags in H5 exist so that the estimator code path is byte-identical apart from the count.

**Logs that must exist for every run.** `config.json` with the git commit; `metrics.jsonl` with per-update train loss, step time and effective rates; `result.json` with final val loss and wall-clock; the H4 checkpoints; and for E3 to E7, the experiment-specific series named in Section 4 (midpoint scores, acceptance rate, drift cosine, held-out row error). A plot script per experiment that reads only these files.

**Pitfalls seen in this benchmark's history.**

1. The 10M tracks are archived and no longer accept submissions; the live tracks are 30M. Develop on 10M, submit on 30M.
2. Chained credit (CutBI, Relay) was disallowed on 22 October 2026 after reaching 3.704 in 147 s. Any idea that reuses a measured credit from one block to train another is out; E5 reuses one measurement at several sites without stepping, which is a different thing, but ask first (Section 6).
3. A learned predictor of credit is not a measurement. The one record that used one (#7, associative credit memory) used it as a control variate with the draws still measuring the residual. E7's self-check keeps that structure.
4. The record's rates, floor and draw count were tuned on 10M and reused on 30M, where the record trails backprop by 0.07. Do not tune on 10M and call it done.
5. bf16 finite differences: the chain's probe step is about 1% of activation RMS, chosen to stay linear while staying above rounding. E5's deeper propagation and E9's curvature term both push on this; run H6 item 5 before trusting either.
6. Identity-path estimators plateau near 4.18 regardless of kernels. They are a development proxy for speed ideas, not a route to 3.5.
7. Published single-run numbers in PR descriptions are seed 0; several PRs report 0.01 to 0.02 differences between the research copy and the official log of the same config. Treat anything under 0.02 as noise until seeds say otherwise.

## 6. Open questions and references

**Question to post as a GitHub issue on qlabs-eng/noise before building E5.** Suggested text: "A probe launched at the embedding passes through every block; the chain already computes the perturbation at each later site in order to continue. Is it within the rules to read the single loss change at the top as a first-order measurement at every site along the way (against the perturbation as it stood there), and solve per site for the credit? No signal takes a step through any nonlinearity; the per-site perturbations are forward-computed. We ask because it reuses one measurement across sites, which is adjacent to the chaining rule." Do not build the implementation until there is an answer; the diagnostic is fine to run regardless.

**Smaller questions to settle locally.**

- [ ] Does `backprop_preset.py` accumulate per-sequence losses as a mean or a sum? The benchmark's `forward` returns the summed loss; the gradient scale affects the rates.
- [ ] What exact value of the probe step (`chain_sigma`) does the PR #40 script use, and is it relative to per-token RMS or per-site RMS? E2's τ is defined in those units.
- [ ] On one GPU, does the Cut port's sequential two-halves layout change any floating-point reduction order enough to move the per-step loss? Check against the published log at steps 2, 10 and 100.

**References used in this document.**

| Source | What it supplies |
| --- | --- |
| [Noise README](https://github.com/qlabs-eng/noise) | Rules, tracks, 30M records (3.409 record; backprop 3.419 Adam, 3.340 ANVIL II), data layout |
| [Noise 10M archive](https://github.com/qlabs-eng/noise/blob/main/archive/10M/README.md) | The 10M record table used for the proxy |
| [PR #11, Boost → Cut](https://github.com/qlabs-eng/noise/pull/11) | Boost and Cut phases, exact-credit checks (cosine 0.9996 to 0.9999 per site), seeds |
| [PR #16](https://github.com/qlabs-eng/noise/pull/16) | Identity-path credit as the track-2 estimator; exact values and gates |
| [PR #20, Relay](https://github.com/qlabs-eng/noise/pull/20) | Backprop baseline with the record's optimizer (3.708 / 3.712 ± 0.009, 38 s); `check_relay.py`; what chaining looked like before it was disallowed |
| [PR #26](https://github.com/qlabs-eng/noise/pull/26) | Identity-path full epoch in PyTorch, 4.1772 in 46.96 s; the H5 starting point |
| [PR #40, current record](https://github.com/qlabs-eng/noise/pull/40) | Record #14 recipe: 128 future draws, rates, warmdown to 30%, end merge details, ablations (64/128/256/512 draws: 3.681/3.674/3.668/3.667) |
| [spicylemonade/boost-cut](https://github.com/spicylemonade/boost-cut) | `backprop_preset.py`, test split, ledgers, Boost-alone 4.062 and Cut-alone 3.984 |
| [Dust paper](https://qlabs.sh/research/dust) | Estimator derivation, the smoothing hypothesis (Section 5), model-size results |
| [Dang, Wen & Malladi 2026](https://arxiv.org/abs/2610.08975) | Noisy quadratic model: curvature-to-noise ratio decides optimizer ranking; preconditioning amplifies flat-direction noise; sharp directions at small batch (E8) |
| [Zhang et al. 2019](https://arxiv.org/abs/1907.04164) | The noisy quadratic model |
| [modded-nanogpt PR #360](https://github.com/KellerJordan/modded-nanogpt/pull/360) | ANVIL II reference implementation |
| [Vuckovic 2018](https://arxiv.org/abs/1810.12273), [Davtyan et al. 2022](https://arxiv.org/pdf/2107.03331.pdf) | Filtering view of a drifting gradient (background for E7's self-check) |
| [Qiu & Tong 2024](https://proceedings.mlr.press/v235/qiu24g.html), [Cai et al.](https://arxiv.org/abs/2003.13001) | Sparse-gradient recovery with few probes; only relevant if H6 item 2 shows sparse credit |
| [Allaire et al. 2026, KronZO](https://www.gerad.ca/en/papers/G-2025-44) | Caution: pretraining gradients are spread across many directions; test before assuming low rank |
