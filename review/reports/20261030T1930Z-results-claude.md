# results review: Claude (loud-reviewer)

- source: git 4acfaa7, read-only; README "Follow-ups (2026-10-30)" and the board
- reviewer: Claude loud-reviewer agent, results lens

Every mean, share and seed count traced reproduces from the result.json files to the stated precision; the problems
are in what is drawn from them and in the board. No chain-run loss is quoted anywhere.

### 1. [major] The board can never give E16 a verdict
The general "every arm has a result" check runs before E16's branch; the `official_eval: false` pair
(`e16.*_track3_plain`) has no val_loss, so E16 stays "waiting". The four-arm reading is met (seeds 0-1: backprop
+0.1411, H8 +0.1431); the closer pair agrees so far (the n-gram tables gain 0.040 held-out under backprop, two
seeds; 0.043 under H8, seed 0).

### 2. [major] E18's two-seed differences under 0.02 are stated as results
v2 (line 139): a difference under 0.02 is not a result without three seeds. NextLat at 30M (per seed s0/s1): lambda
0.3 -0.0045/-0.0050, 1 -0.0086/-0.0074, 3 -0.0082/-0.0050 (one shared h8_cut per seed). Blocks 3, 5, 7 vs all keys
0.0087/0.0073. "Holds or grows": +0.008 and +0.004 over 10M against time_k4's own 0.012 sd. What clears the bar:
lambda 3's +0.048 at 10M vanishing at 30M, and the gap to backprop 0.160 -> 0.123.

### 3. [major] "Waiting, on purpose" says E8's intervention and E14's follow-ups wait; both ran
E8: 21 runs done; its build condition held in part (at 100% exact credit has 0.61-0.98 of its energy below SNR 1;
K = 4 adds mainly in the MLP matrices, c_fc 0.61 -> 0.84). E14: the second stage (9 runs) done and null.

### 4. [minor] E4's +0.337 mixes seed sets; three seeds give +0.328 (per seed against adapt_nowd: -0.008, -0.013, -0.061)
### 5. [minor] The key_paths explanation leaves out the diagnostics against it
Backprop's weights: block 3 0.9% (yet 60% in training); handoff: values 54% vs keys 41%, block 1 15% (yet 1.6% in
training). Call the value-embedding mechanism a conjecture.
### 6. [minor] E19's lr_mult changes Boost too
`cut_lr2` runs Boost at twice the rates; `cut_lr0p5.s0`'s Boost loss differs from h8_cut's by 0.10. The Cut-step
reading does not follow; add a Cut-only arm.
### 7. [minor] Shared-GPU timings
Boost medians are not similar (nokeys 3.98 s, transport 4.00 s, record 3.01/3.21 s); like-for-like pairs are runs
sharing a GPU at once: chain_nokeys 48.55 vs chain_record.s1 56.04 (13.4%); e5.chain_deep 54.34 vs chain_record.s0
55.71 (2.5%, old code). cut_smoke vs cut_prefix ran under different load.
### 8. [minor] E5's ridged readout: S3 0.9924, S4 0.9874, S5 0.9853, S6 0.9884; ridged 10th percentile 0.923, 0.903, 0.908, 0.931 at S3-S6
### 9. [minor] The board shows e5.chain_deep as paired; the chain's noise floor is unmeasured
chain_record.s0 and chain_keys_b357.s0 (same code, seed, Boost) differ by 0.0009 on average in Boost loss (0.0086 at
most); H8 is identical across versions. Treat one-seed chain differences under 0.02 as noise, or repeat chain_record.s0.
### 10. [minor] e10.agree_055 spikes in every seed (5.98 at 472; 5.74 at 381 and 8.36 at 801; 7.02 at 854); finals 4.549/4.637/4.549
### 11. [minor] Hard-coded verdicts and a stale table
E9/E12 unconditional (E9's gate is the exact-gradient setting: snr 3.9126 vs control 3.7820, so the outcome holds);
E18's fixed 10M figures; E20's "+0.150"; table.md from 2026-10-29.
### 12. [nit] E11 per-site averages 83-95%; "1-2%" is 0.5-1.6%; 1.4x is +0.0105; float32 "the same to 3 decimals" fails at S7-S8 at 20%; the superseded key-draw caveat omits the batch-shared rows; chain_keys_b1357 unlisted.

## Checked and fine
E17 (gap 0.1597; every share), the last 9% (live_both_noboost on backprop to 1e-4; Boost +0.014), key_paths shares,
superseded key-draw arms kept out of the board's means, E18 pairs and run lengths, E19/E20/E10/E11/E7/E8/E5 numbers,
provenance (metrics.jsonl matches result.json in all 272 scored runs; eval_every 0; all on H100 80GB), no unfinished
run's loss quoted, the record's key probes on a private generator copy, the repo's board JSON matches a rebuild.
