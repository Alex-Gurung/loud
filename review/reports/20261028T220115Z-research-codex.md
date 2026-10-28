# research review: Codex

- source: frozen copy `53d8094dc1a13665` at 20261028T220115Z (git 7eab8cb plus the working tree)
- reviewer: Codex 0.165.0, model gpt-6.1-sol, session 01a18488-3f14-7432-acee-6449ef7e7afe
- focus: Round 4: review only E5's diagnostic (loud/diagnostics.py credit_recovery, configs/e5/) against the plan's E5 text and PR #40's chain (external/noise/archive/10M/train/val/run.py chain_chunk_eager, chain_credits), and E12 (configs/e12/, loud/launch.py, the proxy recipe pr46 in loud/proxy.py and loud/refs.py) against the plan's E12 text and the current Noise records (external/noise/README.md: track 1 is now PR #46 at 3.338; track 2 PR #44 at 8 min 20 s). Is each faithful to the plan, and where the plan's premises have moved, is the adaptation sound? Also loud/plots.py against the plan's 'a plot script per experiment that reads only these files'.

E12’s switch to PR #46 and its revised interpretation of the 60% run follow the changed records. E5’s diagnostic cannot yet support a decision about the unchanged PR #40 chain, and E12’s retunes lack the seed replication required by the plan. The plotting code reads run artifacts, but omits E12’s target-crossing curve.

### 1. [major] E5’s recovery gate measures a different credit from PR #40’s chain
- where: `loud/diagnostics.py:345`, `loud/diagnostics.py:367`, `loud/diagnostics.py:404`
- evidence: The diagnostic propagates perturbations through every MLP unit and compares recovered credits with autograd through that full MLP (`paths`, lines 198–215). PR #40’s `external/noise/archive/10M/train/val/run.py:1338` instead gathers the routed units; `chain_chunk_eager` at line 558 propagates changes through those selected units. Its defaults also enable own-query responses (`own_query=True`), which the diagnostic freezes. Finally, `direct` is an infinitesimal autograd credit, not the finite-step direct cycle described by the diagnostic and E5 config comments. Freezing attention follows the plan’s diagnostic text, but these additional differences mean passing its 0.95 gate does not establish recovery of the unchanged chain’s credit. README does not explain this limitation.
- fix: Retain the frozen-attention diagnostic, explicitly label its scope, and add a chain-faithful comparison using PR #40’s routed units, own-query behavior and direct finite-difference cycles before using the result to choose the chain implementation.

### 2. [major] E12’s retunes cannot establish the plan’s replicated comparisons
- where: `configs/e12/record_lr085.yaml:6`, `configs/e12/record_lr115.yaml:6`, `configs/e12/record_boost033.yaml:6`, `configs/e12/record_boost134.yaml:6`, `loud/launch.py:19`
- evidence: All four retunes and their record baseline specify only `seeds: [0]`. The plan requires seeds 0, 1 and 2 for reported cells and explicitly rejects small differences without three seeds. Adding seeds to YAML alone would not fix this: `loud/launch.py` replaces only Settings fields and the row loader, while the pinned reference defines `SEED = 0` and uses it for initialization, data permutation and estimator construction. No config argument passes the job seed into that module.
- fix: Add an explicit reference-seed override, record its effective value, and run the baseline and retunes with matching seeds 0, 1 and 2. Keep the published seed-0 reproduction separately identifiable.

### 3. [minor] E12 plots omit the validation-versus-time curve needed for the 60% experiment
- where: `loud/plots.py:90`, `loud/plots.py:218`
- evidence: `record_60pct.yaml` describes its validation curve as showing when the run crosses 3.5. The reference records `val_loss` and cumulative `train_seconds` in `metrics.jsonl`, and `loud/reference.py` copies that log into the harness run. However, `curves_plot` consumes only `train_loss`, `seqs` and `heldout.jsonl`; E12 has no extra plot handler. Its reference runs therefore get a final-loss chart and training curve, but no target-crossing plot.
- fix: Add an E12 plot of recorded validation loss against cumulative training time, with a 3.5 target line and a distinct final merged result.

## Checked and fine

- PR #46 and PR #44 match the supplied Noise README records: 3.338 and 3.499 in 8 min 20 s.
- The `pr46` proxy imports the pinned record’s model construction and optimizer, including ANVIL weight decay 0.75 and warmdown to 30%.
- The 60% launcher truncates the hash-checked training split before permutation and schedule construction: 8,788 rows become 8,756 update rows plus 32 held-out rows.
- E5 queues the 20% and 100% checkpoints, runs no training, and leaves implementation pending the rules answer.
- Plot inputs come from run artifacts; no dataset or checkpoint is needed.
- Focused CPU launch and plotting tests passed: **2 passed**. CUDA execution and full diagnostic outputs were not exercised.