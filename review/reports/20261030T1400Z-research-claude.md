# research review: Claude (loud-reviewer)

- source: git 8077df4 (`git diff 1114022..8077df4`), read-only
- reviewer: Claude loud-reviewer agent, research lens
- focus: the autonomous follow-ups: loud/cut.py deep_readout (E5), key_blocks, key_probes, key_transport; loud/h8.py
  ReadCredit, key_draws/key_assign, lr_mult; loud/diagnostics.py key_paths; board verdicts; README claims for E17,
  E19, E20, E5.

The new estimators are faithful to the record and to v2: `key_transport` agrees with the record's own probes to
3-4e-4 on the CPU test models and about 1e-3 on a trained checkpoint, at one or two GPUs per sequence; `deep_readout`
matches v2's E5, and its S0, S1 and mixing-scalar credits equal the record's. The problems are in how the results are
read.

### 1. [major] The board compares single-seed chain arms with chain_record's two-seed mean, labelled "seed 0"
- where: loud/board.py (E5 and E17 verdicts)
- evidence: `val_mean` averages every counted seed; chain_record has seeds 0 and 1, the other chain arms seed 0. The
  arms are paired with chain_record s0 draw for draw (key_transport, key_blocks and key_probes: false draw no random
  numbers; key_input_credits works on a private generator copy, run.py:1175), so their s0 difference is the change
  itself; a two-seed mean adds seed 1's noise (v2 pitfall 13: 0.01-0.02).
- fix: compare on the shared seeds (seed_mean, as E16).

### 2. [major] E5 as built saves 1-2% of a Cut update on the GPU; the stream cycles look like a small share of it
- where: configs/e5/chain_deep.yaml, loud/cut.py deep path
- evidence (step_seconds at matched load, Boost medians 3.94-4.02 s): chain_record s1 55.3 s; e5.chain_deep 54.3 s;
  e5.chain_deep_smoke 54.7 s; chain_nokeys 48.6 s; chain_key_transport 48.2 s. verify.cut_smoke (128 rows per site)
  41.7 s per Cut update against 37.1 s for verify.cut_prefix (512): 4x fewer chain moves did not shorten an update.
  The deep path adds 28 batched [512, 512, 512] LU solves per sequence on part 0 and builds the own-query context
  three times per sequence.
- fix: profile e5.time_deep by part; one pass per sequence; Cholesky (the ridge makes the system positive definite).

### 3. [major] Giving S1's whole cycle to part 0 is hidden by the serial timing and would not carry to 8 GPUs
- where: loud/cut.py deep path
- evidence: per Cut update and sequence (moves x blocks), the record gives each of two parts 11,264 (+144 scalar on
  part 0); deep readout gives part 0 6,288 plus all solves and part 1 2,048: serially 22.7k -> 8.3k (2.7x), on 8
  GPUs 11.4k -> 6.3k plus solves (1.8x at best), part 1 idle about 2/3 of the chain. The loss is unaffected.
- fix: split S1 by tokens: each part runs all 516 moves and the solves for its half of each chunk.

### 4. [major] E5 was built and is running before the maintainers' answer; the README misstates v2's rule
- where: README (status, E5 bullet), board E5 sentence, configs/e5/readout_* comments; plan v2 lines 149, 192, 205
- evidence: v2: "all sites at 0.95 or above -> ask the maintainers, then implement"; Section 7 asks for the issue
  before E5's implementation. The solved credits feed queries, keys, gate and MLP units (nonlinear consumers).
- fix: state why it was built (a measurement, not a submission), mark its chain results not admissible until the
  question is answered, correct the status row and board, add v2's linear-consumers-only variant.

### 5. [major] "Cut updates already take about 18% less time" is not what the runs show
- evidence: at matched load removing every key probe saves 12-13% (chain_nokeys 48.6 s against 55.3 s); restricted
  arms on lighter-loaded GPUs, normalised by their Boost time, give -6% (b357 s0), -11% (b5), +16% (b357 s1).
- fix: remove the number or say "at most 13% (all probes removed, matched load)".

### 6. [major] H8's key_draws share one set of rows across the batch; the record draws per sequence
- where: loud/h8.py key_rows, cut_forward
- evidence: one [m, W] (or permuted basis) per block per update for the whole [B, T] batch; the record draws input-key
  rows from each sequence's generator (seed + 17 + 1000 sequence, run.py:778), so its 4 sequences read independent
  rows. Shared rows correlate the sequences' read errors, likely overstating the cost of partial key reads.
- fix: draw rows per sequence and read per sequence.

### 7. [major] "Boost's first 20% ... costs about 0.019" is seed 0 alone
- evidence: live_both - live_both_noboost per seed +0.0184, +0.0116, +0.0122; mean +0.0141.

### 8. [minor] The deep path draws 2 bases where the record draws 9, so chain_deep is not seed-paired with chain_record
- fix: draw and discard the skipped sites' bases.

### 9. [minor] The board's E5 verdict quotes the unridged solve; the build uses the ridge
- evidence: ridged medians 0.9853 (S5), 0.9874 (S4), 0.9884 (S6) at 20%; p10 0.90-0.93 at S3-S6; the ridge lifts the
  S7-S8 tail to 0.97-0.99. Still over 0.95.

### 10. [nit] "on the CPU test model to 1e-4": 3-4e-4 on the CPU models, 0.7-1.2e-3 on e1.h8_synth.s0:ckpt_100.pt.

### 11. [nit] Stale seed counts: E20 three seeds give +0.1466; E19's K = 4 arms have three seeds (1.4x: +0.010).

## Checked and fine
- The K branch is c_k(norm(d.blocks[i])), so transport's norm_move(d.blocks[i], raw @ W_k) targets what the probes
  move; the GPU paths (raw_key_moves + fused_key_response; decode_key_pairs len(rows)/total_rows) agree with the CPU
  paths; the two-GPU split is consistent; the c_k key errors transport passes on are bit-equal to the record's; it
  stays within PR #23's norm_move convention; the key options draw no random numbers.
- key_blocks, key_probes false and the option checks; cut_key_blocks and cut_deep_readout pass on 8077df4.
- sites_kernel against chain_chunk_eager for site >= 0; deep_readout against v2's E5 (launch at S1, S0's cycle kept,
  ridge 1e-3 x mean diag toward guesses(), bf16 changes, 512-token chunks, scalar moves and lam as the record's,
  group_sum order). On e1.h8_synth.s0:ckpt_20.pt, bf16 A and a TF32 Gram move each token's ridged cosine by <= 4e-4.
- README E17 numbers reproduce from result.json; key_paths shares match; live_both_noboost equals bp_b4_pr40 to
  <= 1.1e-4 per seed. E19 and E20 configs are one change each; chain configs one change against chain_record;
  final_dtype and compare_every/rank_every do not bias comparisons.
