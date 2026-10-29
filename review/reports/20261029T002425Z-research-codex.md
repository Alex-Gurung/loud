# research review: Codex

- source: frozen copy `8471dde7db828f85` at 20261029T002425Z (git 6daad47 plus the working tree)
- reviewer: Codex 0.165.0, model gpt-6.1-sol, session 01a1850b-4544-7671-b830-5c99b27deafa
- focus: Round 6: fidelity to the plan ("Noise Forward-Only Research Harness Engineering Handoff.md") of what was added or changed since commit d99a223: E11 on the Cut chain (loud/cut.py target_damping; configs/e11), E7's design change (tiled rows read the residual of the identity-path guess, a control variate; is that what the plan's E7 intends, and is it unbiased?), E7's self-check (rows_check; configs/e7/check_*), the E7 precondition gate (configs/e7/precondition.yaml, loud/gate.py: is the rule's reading sound?), E8's real test on the chain (configs/e8/cut_rows128*.yaml), E10's literal control (configs/e10/agree_all_t70.yaml), E3's calibration cell (configs/e3/bp_b16.yaml) and checkpoints, E4's clock change, the smoke gates. Is anything the plan asks for still missing?

The additions still leave E8’s forward-only test unimplemented, and its prerequisite diagnostic reconstructs the wrong pending update. E7’s follow-up scheduling and E10’s literal sweep also remain incomplete. GPU numerical fidelity and timing were not verified.

### 1. [major] E8’s “real test” still uses backpropagation during training
- where: `loud/cut.py:609`, `loud/curvature.py:118`, `configs/e8/cut_rows128_damped.yaml:3`
- evidence: Cut calls `shaper.after_step()`, which measures curvature through `torch.autograd.grad(ref(x, y), p)` at perturbed weights. These measurements determine subsequent updates. The plan explicitly specifies forward-only ± probes for this stage; autograd diagnostics and proxy experiments do not establish whether that measurement works. The config acknowledges autograd but supplies no reason for replacing the required forward-only test.
- fix: Implement the probe-based curvature measurement for the chain. Until then, label this cell as a hybrid development control and mark the forward-only E8 test as pending.

### 2. [major] H6’s pending-update directions mix incompatible gradient scales
- where: `loud/diagnostics.py:263`, `loud/diagnostics.py:280`
- evidence: `direction_stats` differentiates a token-summed loss, averages across sequences, then mixes it with saved optimizer rails. Training uses token-mean gradients, making the diagnostic gradient 2,048 times larger on benchmark sequences. A numerical reproduction changed the leading singular vector from `[1, 0]` to `[0, 1]` solely through this scaling. The reconstruction also omits the fast/slow rail recurrences that the reference optimizer applies before forming its pending update. Consequently, the requested E8 distribution describes different directions.
- fix: Normalize by valid-token count and reproduce the rail updates before constructing the pending tensor. Compare that tensor against a captured reference optimizer input to `polar()`.

### 3. [minor] E7’s self-check runs without establishing its follow-up condition
- where: `configs/e7/check_045.yaml:16`, `configs/e7/check_077.yaml:11`
- evidence: Both cells depend only on completion of the four ramps for their seed. Queue dependency handling checks completion, not whether the winning ramp differs between seeds. Both cells also hardcode `ramp64_f60`, regardless of the observed winners. Comments instruct the operator to cancel or edit them, but bulk submission automatically releases them without that review.
- fix: Exclude these cells from automatic submission until the ramp comparison is complete, or add a decision gate that checks the cross-seed condition and records which ramp the follow-up extends.

### 4. [minor] E10’s literal interpretation is missing two planned thresholds
- where: `configs/e10/agree_all_t70.yaml:24`, `loud/options.py:104`
- evidence: The plan requests agreement thresholds `{0.6, 0.7, 0.8}` across all tokens. Only `agree_all_t70` implements that denominator. The existing `agree_t60` and `agree_t80` inherit `agree_over='active'`, so they test a different criterion. The added control resolves the 70% case but does not complete the planned sweep.
- fix: Add matching `agree_over: all` cells at 0.6 and 0.8.

## Checked and fine

- E7’s residual estimator has the control-variate form `guess + decoded(measurement − projected_guess)`. Independent random column signs preserve its expectation relative to the plain estimator; existing finite-difference bias remains.
- Focused CPU checks `cut_windows` and `cut_targets` passed, covering window decoding, residual arithmetic, held-out row errors, runway selection and ridge fitting.
- E7’s median-of-site-medians gate is an explicitly documented interpretation of the plan’s ambiguous aggregate criterion.
- E3’s batch-16 calibration matches the baseline recipe, seeds and held-out setting; population checkpoints now preserve the trunk and individual copies.
- E4 restores update state while retaining the data-step schedule clock.
- Cut configs pin H100 and matching held-out settings. The published-prefix tolerance is now 0.01.
- Data hashes, fixed ordering and official scorer delegation remain intact. Inspected selection paths use training-held-out data.
- The 76 tests outside `test_methods.py` passed. The full suite was interrupted after 50 passes; CUDA execution was not exercised.