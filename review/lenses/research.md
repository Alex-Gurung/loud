# Lens: fidelity to the research plan

The plan is `Noise Forward-Only Research Harness Engineering Handoff.md` at the repo root. The reference code is
pinned in `external.json` and checked out under `/mnt/disk/loud/external/` (`noise`, `boost-cut`). The question:
will runs from this harness answer the plan's questions, with numbers comparable to the published records?

1. **Each built piece against its spec** in sections 2 and 3 of the plan (the README status table lists the pieces).
   Report what is missing, different or extra. A difference the README does not flag is a finding. A flagged
   difference still needs a stated reason.
2. **The proxy** (`loud/proxy.py`) against `external/boost-cut/noise/analysis/backprop_preset.py` and the trainer module it imports.
   Read both and compare behavior, not text. Cover:
   - model and init; optimizer groups, rates and betas;
   - the warmdown formula and its step indexing;
   - data order and the untimed warmup;
   - loss reduction, gradient averaging and dtype casts, bf16 autocast;
   - what the timer covers.
3. **Data and scoring:** `loud/data.py` and `loud/score.py` against noise's `utils/`. Cover the eval loop, `val.pt`,
   the hashes and the test split.
4. **The benchmark's rules:**
   - nothing in a run reads `val.pt` to make a decision;
   - one epoch;
   - held-out sequences never feed an update;
   - no checkpoint is chosen on validation.
5. **The queued configs** in `configs/` against the plan's E1 and its three verification steps:
   - correct expectations and tolerances (the plan's gate is seed noise, 0.01);
   - consistent seeds and held-out settings across cells that will be compared.
6. **Anything that biases a comparison between variants:**
   - warmup or compile state;
   - code version or GPU type;
   - held-out size;
   - bf16 versus fp32 checkpoints.
