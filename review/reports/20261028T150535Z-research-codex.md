# research review: Codex

- source: frozen copy `e32e300d93b77497` at 20261028T150535Z (git 3bb0197 plus the working tree)
- reviewer: Codex 0.164.0, model gpt-6.1-sol, session 01a1830b-920f-7b52-9e0e-d79cc04006d4

The harness does not yet enforce the plan’s verification requirements or preserve all promised H4 artifacts. I found no data leakage in the queued proxy configs, but experiment jobs can proceed without the claimed reference comparison, and failed checks still enter aggregate results. GPU numerical equivalence remains unverified; all 12 CPU tests passed.

### 1. [major] The verification gate accepts deviations above the plan’s stop threshold
- where: `configs/verify/bp_b8.yaml:14`
- evidence: The config permits `3.708 ± 0.02`, whereas the plan’s final verification instruction says to stop above seed noise, 0.01. Calling `loud.runlog.check` with loss `3.723` returned `pass`, despite a 0.015 deviation. Both E1 configs depend on this gate.
- fix: Set the verification tolerance to 0.01, including the script and one-forward verification configs.

### 2. [major] The claimed step-by-step reference comparison is never checked
- where: `configs/e1/bp_b4.yaml:8`, `configs/verify/bp_b8_onefwd.yaml:1`
- evidence: The one-forward config says its per-step losses “must agree” with the unmodified script. Both jobs only check their final validation loss against the same broad target; neither reads the other’s metrics. E1 depends only on `score_record` and `bp_b8`, so it can start even if either comparison job fails. Searching the implementation found no metrics comparator.
- fix: Add a comparison job that checks matching steps and training losses with an explicit numerical tolerance. Make E1 depend on its success.

### 3. [major] Queued baselines never retain the nine trajectory means promised by H4
- where: `loud/options.py:21`, `configs/e1/bp_b4.yaml:9`
- evidence: Both `means` and `save_means` default to `False`; no config overrides either. Consequently, `TrajectoryMeans.observe` immediately returns, and `proxy.py:222–225` saves nothing. H4 explicitly requires these means for reuse in merge experiments, but the README marks H4 done without flagging their omission from queued runs.
- fix: Enable both options in the baseline configs intended to supply reusable artifacts. Document and provision their storage requirement before running them.

### 4. [major] Failed checks are included in reported means
- where: `loud/collect.py:67`
- evidence: Aggregation filters only on completeness and the presence of validation loss. The failed-check condition merely adds a flag after calculating statistics. A direct probe with a complete, failed-check row produced an aggregate with `n=1` and `val_mean=4.0`. This contradicts the module’s statement that failed-check runs are listed separately and undermines the verification stop rule.
- fix: Exclude failed-check runs from aggregate statistics and list their failures separately.

### 5. [minor] H3 uses four forwards instead of the specified single forward
- where: `loud/heldout.py:11`
- evidence: The default batch size is eight, and `loss()` splits the 32 held-out sequences into four model calls. H3 specifies scoring them in one forward pass. This departure is not documented or justified; it becomes recurring overhead for experiments that score candidates frequently.
- fix: Default to a single 32-sequence forward, or document and measure why smaller batches are necessary.

## Checked and fine

- Live train, validation and test tensors passed the pinned shape and token-hash checks; the archived 10M prefix is checked by the passing data-order test.
- The test preparation script follows the reference stream boundaries and additionally checks the published test hash.
- The proxy reuses the pinned model initialization and optimizer factories, including optimizer groups, rates and betas.
- Warmdown uses the reference’s one-based step indexing.
- Default micro-batches form a float32 mean of per-sequence gradients before casting to parameter dtype; the objective is mean token loss under bf16 autocast.
- Warmup uses random tokens, followed by fresh model and optimizer construction.
- Queued E1 cells share seeds `[0, 1, 2]`, order and 32 held-out sequences. Held-out rows never feed updates, and training consumes each remaining row once.
- Queued experiment configs disable in-run validation. Final scoring delegates validation to the pinned official evaluator with strict model loading; no validation-based checkpoint selection exists.
- Training timing excludes warmup, diagnostics and final scoring. Final checkpoints retain exact weights; intermediate bf16 storage and its disk-space rationale are documented.
- H5, H6 and identity-path verification are explicitly marked unfinished.
- CPU suite: **12 passed**. CUDA execution, compilation behavior and published-loss reproduction could not be tested here.