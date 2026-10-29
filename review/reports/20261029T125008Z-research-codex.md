# research review: Codex

- source: frozen copy `8d093195803e27fb` at 20261029T125008Z (git 3ef4441 plus the working tree)
- reviewer: Codex 0.165.0, model gpt-6.1-sol, session 01a187b5-fca7-7c03-b386-c1d07f7dfccd
- focus: Round 7: the v2 update (Noise Forward-Only Research Harness Engineering Handoff v2.md is the plan; v1 is history). New or rewritten: loud/h8.py (H8 cut, synthesis and Boost modes, Scheme, Replay/reforward, E8/E10/E11/E13 rules, E14/E15 latent objectives), loud/arch.py (E16), loud/diagnostics.py (v2 H6 tools incl. snr_spectrum and synth_vs_real), loud/latent.py, loud/ema.py, loud/ngram.py, loud/snr.py, loud/fisher.py, loud/population.py (E3 local SGD), loud/variants.py (E2 blur, E4), loud/anvil.py (ShapedBatchedAnvil), loud/score.py (track-3 model_file), loud/plots.py, loud/jobqueue.py claim order, configs/ (v2 experiments, priorities, gates), README.md. Earlier rounds' fixes are in review/LEDGER.md. Check each experiment's arms, settings, gates and success measures against v2 section 5, and H8/H6 against v2 section 3.

Changes are needed before interpreting the v2 experiments. Four verified issues prevent E13 from running, change E10’s stated intervention, bypass H8 validation for baselines, or leave E15’s success measure incomplete. GPU behavior was assessed from source only.

### 1. [major] E13’s diagnostic crashes before producing the results its gates need

- where: `loud/diagnostics.py:365`
- evidence: `summary, tensors = ngram.tiers_tool(...), {}` assigns the entire return tuple to `summary`. `tiers_tool()` returns `(summary_dict, tensor_dict)` through `ngram.tiers()`, so `json.dumps(summary)` encounters tensors. A CPU reproduction using the real tiny-model diagnostic raised `TypeError: Object of type Tensor is not JSON serializable`. Both tiers jobs therefore fail, blocking all three E13 arms.
- fix: Unpack the result directly: `summary, tensors = ngram.tiers_tool(...)`. Exercise the diagnostic task wrapper through JSON and tensor-file output.

### 2. [major] E10’s “70% agreement” filter can retain entries with only 40% agreement

- where: `loud/h8.py:440`
- evidence: The mask uses `abs(sign(G).T @ sign(X)) >= 0.4 * tokens`. This represents 70% agreement only when every contribution is nonzero. Calling `agreed_gradients()` with four positive contributions and six zero contributions retained the entry and returned gradient `4.0`: only 40% of tokens supported that sign. Zeros occur naturally in the model’s ReLU-squared MLP. The README acknowledges the nonzero caveat, but the implemented rule does not satisfy v2 E10’s requirement that at least 70% of tokens agree.
- fix: Count nonzero contributions as well as signed votes. Keep an entry when `(nonzero_count + abs(net_votes)) / 2 >= 0.7 * tokens`.

### 3. [major] E1’s H8 baselines run even when trained-checkpoint validation fails

- where: `configs/e1/h8_synth.yaml:9`
- evidence: This config, `h8_cut.yaml:8`, and `h8_cut_noboost.yaml:8` depend only on scoring verification and the H8 smoke run. None requires `h6.h8_validation.s0`. I evaluated their actual dependency lists with the listed dependencies passing and H8 validation failing; `JobQueue.dependency_state()` returned `ready` for all three. This contradicts v2’s validation-before-use requirement and the README’s claim that every H8 experiment waits for it.
- fix: Add `h6.h8_validation.s0` to all three baseline dependency lists. The gate depends on the backprop checkpoint, so this creates no dependency cycle.

### 4. [major] E15 cannot compare handoff cosine against its baseline

- where: `loud/h8.py:729`
- evidence: The handoff measurement requires `self.targets is not None`, which is true only for the latent Boost variants. E15 reuses `e1.h8_cut` as its ordinary logit-lens baseline; that run has no latent targets and records no handoff cosine. Exercising `H8.diagnose()` at the handoff returned `{}` for the baseline and `{'handoff_cut_cosine': 0.8}` for a latent variant with the measurement mocked identically. Thus the planned success criterion “raises the handoff cosine at equal final loss” cannot be evaluated. The baseline config also saves no handoff checkpoint for reconstruction.
- fix: Record the handoff cosine for ordinary Boost → Cut baselines on the same held-out rows as E15.

## Checked and fine

- Pinned external files and train, validation and test token hashes passed `loud.refs --check`.
- Proxy model construction, optimizer groups, warmdown indexing, gradient averaging, autocast and warmup were compared with the pinned preset and imported trainer.
- Final scoring delegates to the official evaluator, including strict track-3 model loading. Reviewed decision paths use held-out training rows.
- E2/E3/E4 arm settings, E6 row assignments, E14’s objective and EMA teacher, and E16’s architecture path were examined.
- All 16 H8 CPU checks passed, plus the proxy gradient and proxy-loop checks. The broader suite was stopped after those 18 passes; it was not completed.