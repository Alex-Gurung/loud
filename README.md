# loud: a one-GPU research harness for the Noise benchmark

Implements the [engineering handoff](Noise%20Forward-Only%20Research%20Harness%20Engineering%20Handoff.md) (the
plan): a single-GPU harness for [Noise](https://github.com/qlabs-eng/noise), pretraining a fixed 37.7M-parameter GPT
without backpropagation. It includes a GPU job queue and workers that follow the stylo setup (Kubernetes Jobs on
EIDF, the GPU cluster, sharing one volume).

**Terms.**
- **Proxy:** backprop with the record's optimizer, a development stand-in, never an entry.
- **Identity path:** track 2's forward-only estimator (PR #26). One Hadamard cycle of 512 probes at the top stream,
  with credit handed down the residual scalars.
- **Cut chain:** track 1's record estimator (PR #40).
- **ANVIL II:** the records' optimizer for block matrices.
- **H1–H7 and E1–E12:** the plan's infrastructure pieces (its section 3) and experiments (section 4).
- **Verification 1–3:** the plan's three checks before any experiment (section 2).

## Status

| Piece | State |
| --- | --- |
| Data: `train.pt`, `val.pt` (pinned hashes), `test.pt` (boost-cut's split, `cb7a29f0…`) | built in `data/` |
| Record checkpoint `track1_boostcut_final.pt` (`499dff9f…`) | downloaded, hash checked |
| H1 backprop proxy (`loud/proxy.py`) on the shared loop (`loud/train.py`) | done; CPU tests on a tiny model |
| H2 per-sequence gradients (`loud/pergrad.py`) | done, with an unbiased per-matrix noise scale |
| H3 held-out scorer (`loud/heldout.py`) | done; every call logged to `heldout.jsonl` |
| H4 fraction checkpoints and the record's nine trajectory means (`loud/checkpoints.py`) | done; means scored on H3 |
| H5 identity path, PR #26 on one GPU (`loud/identity.py`), with E6's `rows` and `row_offset_per_step` | done; CPU tests: one update equals PR #26's own |
| H5 Cut chain, PR #40 on one GPU (`loud/cut.py`, task `cut`), with E6's rows and E7's ramp | done; CPU tests: 4 updates equal PR #40's own loop on 8 processes |
| H6 diagnostic toolkit (`loud/diagnostics.py`, task `diagnose`): all five tools | done; CPU tests on a tiny model; configs in `configs/h6/` |
| E2, E4, E10 proxy variants (`loud/variants.py`; `configs/e2/`, `e4/`, `e10/`: 32 jobs) | done; CPU tests: brute force, exact undo |
| E3 population on split data (`loud/population.py`, task `population`) | done; CPU tests of the split, schedule and rules; `configs/e3/` |
| E8 and E9 proxy versions (`loud/curvature.py`, `loud/fisher.py`, through `loud/anvil.py`) | done; CPU tests against exact HVPs and per-token sums; `configs/e8/`, `configs/e9/` |
| H7 runner and results table (`run.py`, `collect.py`); per-experiment plot scripts | runner and table done; plots not started |
| Port checks against their sources (`compare` task) | done |
| GPU queue and workers (`loud/jobqueue.py`, `loud/worker.py`, `worker/`) | done; tested end to end on CPU |
| Independent review (Codex and Claude reviewers, [review/](review/README.md)) | round 1 triaged in [the ledger](review/LEDGER.md) |

Nothing has run on a GPU yet. Queued (48 jobs):
- verification 1 to 3 with their per-step checks and two smoke runs (10 jobs);
- E1: nine backprop cells and the identity path;
- E6 on the identity path (12 cells, with a control);
- H6's diagnostics on E1's checkpoints (16).

The plan estimates 5 to 7 minutes per 10M run on one GPU. Add compilation, scoring and the H6 tools, and the queue
comes to about 3.5 GPU-hours, roughly 1.5 hours on four workers. The longest chain is smoke, verification 2, `e1.bp_b4`
and then H6.

## Running on GPUs

Workers are Kubernetes Jobs, launched from the root node as in stylo. Each takes one GPU and runs queued jobs one at a
time from `artifacts/queue/` on the shared volume. A worker exits after 20 idle minutes; EIDF reclaims idle GPUs
after about 30.

```bash
kubectl -n eidf107ns create -f worker/gpu_worker.yaml       # H100 only (comparable wall-clocks)
kubectl -n eidf107ns create -f worker/gpu_worker_any.yaml   # H100 or H200 (jobs without a gpu: requirement)
```

The image is stylo's pinned `vllm/vllm-openai:v0.32.0`, used only for its PyTorch stack (torch 2.13). Nothing is
installed at startup. Each pod writes its startup phases to `artifacts/queue/bootstrap_logs/<pod>.log`, then checks the
pinned external code, the downloads and all three data splits (`python -m loud.refs --check`), and stops if any is
missing or changed. `loud.status` lists pods whose startup failed.

From the CPU pod, in `/mnt/disk/loud` (plain `python3`; no torch needed):

```bash
python3 -m loud.submit configs/verify configs/e1 configs/e6   # queue every seed of every config (idempotent)
python3 -m loud.submit configs/e1 --dry-run                   # what each job would do, checked against the queue
python3 -m loud.status                                        # workers, running, queued (and why), failed, done
python3 collect.py                                            # the results table (also artifacts/results/table.md)
```

On a GPU you already have: `python run.py --config configs/verify/bp_b8.yaml`. A direct run works in
`s<seed>.direct<k>/`. It becomes `s<seed>/` when it finishes, unless that exists or the queue holds the same job.

### What runs, and the gates

The plan says to stop if a verification disagrees by more than seed noise (0.01), so every experiment cell waits for
its gates. A job runs only when each job in its `after` list is done and passed its own check.

1. `verify.smoke`: a complete 57-update proxy run on the 1M range through every H1–H4 path, the saved means and
   optimizer state and official scoring. Every other verification job waits for it.
2. `verify.score_record` (verification 1): the record checkpoint must rescore to 3.94481 (val) and 3.93696 (test),
   within 1e-5.
3. `verify.bp_b8` (verification 2): the proxy on the 10M set, one sequence per forward (the 8-GPU layout). Target:
   boost-cut's ledger run `bp38_s0` of `backprop_preset.py`, val 3.7083 and test 3.7014, within 0.01. Three checks
   use the per-step logs:
   - `verify.bp_b8_ledger`: the training loss at every update the ledger logged (every 61) agrees within 0.01;
   - `verify.bp_b8_script` and `verify.bp_b8_onefwd`: the unmodified script, and the proxy with whole-batch forwards
     like it;
   - `verify.bp_b8_match`: those two agree within 0.002 for 50 updates, and within 0.01 at the end.
4. `verify.identity_smoke`: a 57-update identity-path run through E6's windows, the drift check and H4.
5. `verify.identity` (verification 3): the identity path for a full 10M epoch, against PR #26's 4.1772 (within 0.01).
   `verify.identity_match` compares its first 20 updates with PR #26's published 8-GPU log (same seed, data and draws).

6. `verify.cut_smoke`: a complete 16-update Cut-chain run through E7's ramp, the drift check, H4 and the end merge.
   `verify.cut_prefix` then runs the 10M schedule for 262 updates (all 242 Boost and 20 Cut), and
   `verify.cut_prefix_match` holds them against PR #40's published 8-GPU log: within 0.003 over the first 20 updates
   (the plan's steps 2 and 10), within 0.02 over all 262 (step 100 and the first Cut updates). Same seed, data and
   draws, so only float rounding separates them. The Cut-chain cells wait for both.

The comparison jobs wait only for their runs to finish (`after_done`), not to pass, so they still run when a
final-loss gate fails, which is when they are needed to tell a port error from an environment difference.
7. E1: backprop at batch 8 and batch 4, three seeds each, with held-out scoring and the nine means:
   - `bp_b4` saves H4 checkpoints and means for later diagnostics;
   - `bp_b4_pr40` is batch 4 with PR #40's own recipe, since no published run anchors boost-cut's rates at batch 4;
   - `e1.identity`, seed 0.
8. E6 on the identity path: K = 1, 4, 16 (512, 128, 32 probe rows per update), three seeds each, with the drift check
   every 50 updates. A control cell, `identity_k4_fresh`, reads 128 rows of a fresh basis each update, so E6 can say
   whether stitching a cycle over K updates matters or only the row count does.
   - The Cut chain on the 1M slice: `e1.cut_1m` (seed 0); E6's `cut_k1`, `cut_k4`, `cut_k16` and the control
     `cut_k4_fresh` (three seeds each, drift check every 10 updates); E7's four ramps (`configs/e7/`, two seeds each,
     against `e6.cut_k1`). About 35 minutes per run on one H100. E7 is the plan's to run only if H6's `credit_dump`
     shows the identity-path guess good early and poor late, so read that before submitting `configs/e7`.
9. H6 on `e1.bp_b4.s0`'s checkpoints (`configs/h6/`), the diagnostics behind E5, E7 and E8:
   - `credit_dump` and `cos_to_autograd` at all five fractions;
   - `token_map_svd`, `linearity_check` and `direction_stats` at 20% and 100%.

   Each measures held-out sequences, never training ones. `direction_stats` takes the top singular directions of the
   batch gradient: checkpoints keep no optimizer state, so they stand in for the plan's pending update. Its curvature
   is a central difference of autograd gradients. The plan asks for a Hessian-vector product, and flex attention has
   no double backward.
9. E3, E8 and E9 (`configs/e3/`, `configs/e8/`, `configs/e9/`), each first gated on its own complete 1M smoke run
   (`verify.population_smoke`, `verify.curvature_smoke`, `verify.fisher_smoke`). E8 also waits for H6's
   `direction_stats`, which the plan runs first and which says where E8's knob should sit.

### How the queue works

- **Jobs.** A job is one (experiment, variant, seed). Its id is `<experiment>.<variant>.s<seed>`, and its run is
  `artifacts/runs/<experiment>/<variant>/s<seed>/`.
- **Frozen code.** Code is frozen at submission (`artifacts/code/<digest>/`, from the checkout `loud/` sits in), so
  later edits never change queued or running work.
- **Attempts.** Each attempt runs in its own `s<seed>.attempt<k>/`, which doubles as its lease: only the current
  attempt can renew it or record a result. A successful attempt is renamed to `s<seed>/`. Failed and abandoned
  attempts stay where they are.
- **Leases.** The worker heartbeats every 30 s. If a pod dies, its job's lease expires after 5 minutes and the job is
  queued again, with up to 3 attempts. SIGTERM (pod deletion) gives the job back immediately.
- **Resubmitting** the same config:
  - a job that has not started takes the newer code, priority, `after` and `gpu`;
  - a crashed (failed) job is queued again;
  - a done job (including one that failed its check) and a cancelled job change only with `--rerun`, which keeps the
    old run as `s<seed>.superseded<k>`.

  A submission is checked in full first: duplicate ids, unknown `after` ids and conflicts stop it before anything is
  written.
- **Disk.** Workers pause when the shared volume has less than `LOUD_MIN_FREE_GB` free (default 2), and `loud.status`
  says so.
- Nothing is ever deleted: attempts, superseded runs, snapshots and job history all stay.

### When a job fails or is blocked

- `loud.status` gives the cause: `blocked: verify.bp_b8.s0 failed its check`, `waiting for …`, or a failed job's last
  log line and its log path.
- A crashed job (under `failed`): fix the cause and resubmit its config; it is queued again.
- A job that ran but failed its check is done, so resubmitting changes nothing: after fixing the cause, resubmit it
  with `--rerun` (for example `python3 -m loud.submit configs/verify/bp_b8.yaml --rerun`). Its dependents become ready
  when it passes.
- If the gate itself was wrong, edit the dependent configs' `after` and resubmit them: waiting jobs take the new
  dependencies.
- `python3 -m loud.submit --cancel <job id>` cancels a queued job.

## Layout

```
loud/            the harness package (frozen into each job's snapshot)
  train.py       the shared loop: warmup, timing, H3/H4, scoring      data.py      hash-checked splits, order, held-out
  proxy.py       H1 backprop proxy (+ H2 per-sequence gradients)      identity.py  H5 identity path (PR #26)
  heldout.py     H3                                                   checkpoints.py  H4
  score.py       official utils/eval.py scoring in a fresh process    compare.py   per-update loss check of two runs
  diagnostics.py H6: five offline measurements on H4 checkpoints
  population.py  E3: copies of the proxy on split data                anvil.py     ANVIL II with a hook for E8, E9
  curvature.py   E8: steps by curvature-to-noise (proxy)              fisher.py    E9: true-Fisher left factor (proxy)
  reference.py   run a pinned external script unmodified               refs.py      pinned external imports
  options.py     task registry, option tables and checks (no torch)   run.py       H7 runner
  collect.py     H7 results table
  jobqueue.py, worker.py, submit.py, status.py, snapshot.py            the GPU queue
configs/         one YAML per variant; seeds listed inside
review/          the review rubric, lenses, Codex runner, reports and ledger
worker/          Kubernetes Job YAMLs and the pod bootstrap
scripts/         fetch_external.sh, prepare_data.sh, prepare_test.py (CPU)
tests/           queue, worker, harness and method tests (CPU)
external.json    pinned commits and file hashes of everything imported from other repos
external/  data/  artifacts/        shared-volume state, gitignored
```

Recreate `external/` with `scripts/fetch_external.sh` (it clones Noise again at PR #26's commit as
`external/noise-pr26`), and the data with `scripts/prepare_data.sh`. Run the tests with
`PYTHONPATH=. /opt/loud-cpu-venv/bin/python -m pytest tests`, about 6 minutes on this pod. The method tests use
tiny models and short sequences, with dense attention instead of flex attention (which has no CPU backward); the
Cut chain's checks run it eagerly (no compilation) with a 512-token vocabulary.

## Run configs

```yaml
experiment: e1
variant: bp_b4          # one change per variant; the file is configs/<experiment>/<variant>.yaml
task: proxy             # proxy | identity | population | score | reference | compare | diagnose | noop (loud/options.py)
seeds: [0, 1, 2]
priority: 50            # lower runs first
after: [verify.bp_b8.s0]
gpu: H100               # optional: only workers whose GPU name contains this take the job
args: {recipe: boostcut, sequences: 10M, heldout: 32, batch: 4, checkpoints: true}
expect: {val_loss: [3.7083, 0.01]}   # optional gate; recorded as pass/fail in result.json
```

To add a variant, copy a config, change `variant` and the one option under test, and submit it. Unknown keys and
options, bad values, an `expect` on a metric the task does not produce and duplicate job ids all fail at submission.

Every training run writes:
- `config.json`: the spec, the code snapshot and its git state, the pinned external commits, torch and the GPU;
- `metrics.jsonl`, one line per update: training loss, step time, the effective rates and ANVIL II's beta, plus
  experiment series such as `grad_noise` (H2) and `drift` (E6);
- `heldout.jsonl`, `final.pt` and the H4 files;
- `result.json`: the official validation loss from `utils/eval.py` in a fresh process, the test loss, the held-out
  loss, each mean's held-out loss, the training time and the check.

The `reference` task's `val_loss` is the script's own in-process `evaluate()`, the same computation.

## Methods

**The proxy** (`recipe: boostcut`, the default) takes model construction, init, ANVIL II + Adam and rates verbatim
from boost-cut's `train/boostcut/run.py`, the module `backprop_preset.py@539386e` imports. `recipe: pr40` instead
takes PR #40's rates and betas, its warmdown to 30% and head std 0.02, and batched ANVIL II.

- **One GPU = eight.** The 8-GPU script gives each GPU one sequence and averages the float32 gradients. Here each
  micro-batch is one sequence, and the same float32 mean is formed before the cast to each parameter's dtype; a CPU
  test checks this exactly. The loss is the batch mean, which answers the plan's open question: mean, not sum.
- **Data order and held-out set.** Every record permutes its range with `randperm(N, seed 1)`, for every seed (the
  seed changes the init and the draws). The held-out set is the last 32 sequences of that order, as in PR #40.
  Verification runs use `heldout: 0`, exactly as the published runs. Experiment cells use 32, so their baselines and
  variants train on the same 4,848 sequences.
- **Matching a published log** uses the 10M range with `stop_after`. `sequences: 1M` permutes rows 0 to 487 on its
  own, so its updates are not the 10M log's first updates.
- **H4** is on in every experiment cell: the five checkpoints (about 0.45 GB per run) and the nine means, scored on
  the held-out set (saved only by `bp_b4`). All five checkpoints store 2-D weights in one dtype (bf16 by default,
  the same forward under bf16 autocast); `final.pt` is always exact. `ckpt_100.pt` exists only for complete runs.
  Verification runs reproduce the published layout and keep only `final.pt`.

**Proxy variants** (`loud/variants.py`) are one option each, off by default and at most one per run. The default
path is unchanged: the CPU test of the float32 mean still holds bit for bit. All three cells run at batch 8 against
`e1.bp_b8`.
- **E2, `smooth_tau`.** The sites are the raw embedding and each block's mid stream. Each token gets a random r of
  length tau × 0.01 × its RMS × √512. That is PR #40's chain probe, a ±1 Hadamard row times `chain_sigma` (0.01) times
  the per-token RMS, which answers the plan's open question: the step is relative to each token's RMS. A clean no-grad
  forward gives the RMS and `train_loss`; two forwards add +r and −r, and the mean of their gradients is the blurred
  loss's gradient. `blurred_loss` is logged too. A CPU test checks the length, the shared r and the odd-order
  cancellation (the error falls 4× when r halves). About 2.3× a baseline update.
- **E4, `step_control`.** `reject`: after the step, one no-grad forward on the same micro-batches; if the loss rose,
  the weights and all optimizer state (ANVIL II's rails and energy, Adam's moments, both step clocks) are restored
  from a snapshot, and the batch feeds no update. `adapt` also multiplies every rate (and so ANVIL II's decay) by
  `accept_up` after an accept and `accept_down` after a reject, on top of the schedule. `warmdown: 0` removes the hand
  warmdown; ANVIL II's own momentum schedule stays. The plan's 1.05 and 0.8 settle at 82% acceptance, so `adapt_t60`
  uses 0.9294 for its stated 0.6. Logged: `accepted`, `trial_loss`, `lr_multiplier`, and `lr` is the effective rate.
  About 1.35× a baseline update.
- **E10, `token_shaping`.** Every `nn.Linear` weight gradient (blocks, value-embedding gates, LM head; the embedding
  tables keep theirs) is formed in hooks from each token's credit, the output gradient, and its input, with autograd's
  own matmul (bit for bit on CPU). `cap` caps each token's credit norm at the batch's `cap_quantile`, per layer; it
  keeps credits and inputs in bf16 until the batch is done (about 0.3 GB per sequence), roughly 1.1× a baseline update.
  `agree` keeps entries where at least `agree_threshold` of the tokens with a nonzero contribution share the majority
  sign; the counts are exact sign matmuls summed over micro-batches, roughly 2× a baseline update.

**The identity path** imports PR #26's `train/speedrun/run.py@2604407` unchanged: its model, init, optimizer,
settings and `FastRelay` estimator. There is one estimator per virtual GPU, seeded by its rank as on 8 GPUs. Their
per-sequence gradients are summed where the script all-reduces, then one optimizer step follows. A CPU test checks that
one update over two virtual GPUs equals PR #26's FastRelay at ranks 0 and 1 with `Boost.step`'s sum. Differences from PR #26:
- **Data order:** the records' fixed order, `randperm(seed 1)`, for every seed. PR #26 shuffles with `seed + 1`;
  the two are identical at seed 0.
- **`rows`** (E6) limits only the top-stream probe, the one cycle that reads the loss, to a window of the signed
  Hadamard rows, scaled by width / rows. That is unbiased, because the basis carries random column signs.
- **`row_offset_per_step`**: the K = 512 / rows consecutive updates of a cycle read complementary windows of one
  signed basis, so every direction is measured once per cycle. Each update still draws its signs as PR #26 does, so
  every K consumes the same random stream. CPU tests check that the K windows of a cycle average to the full cycle,
  and that the full cycle runs PR #26's `probe()` itself.
- **The drift check** (`drift_every`) uses its own estimator, so its draws never touch training. It measures a held-out
  sequence (no momentum holds it) against the blend ANVIL II steps along, and against the fast and slow momenta alone.
  One sequence's cosine is noisy: compare it across K.

**The Cut chain** (task `cut`) imports PR #40's `archive/10M/train/val/run.py@4ef285e` (10M record #14, 3.676 in
2101 s on 8×H100) unchanged: model, init, settings, the Boost and Cut estimators, optimizers, schedule and end merge.
That script runs at batch 4 on 8 GPUs, two per sequence. The two GPUs of a sequence take complementary halves of every
cycle and add their parts with an all-reduce in the middle of each estimate. Here the 8 GPUs are virtual ranks with
the record's seeds and roles (`virtual_gpus`, default 8). The two halves of a sequence are threads that take turns:
one runs at a time and hands over at each of those all-reduces, where the parts are added. So the run draws the 8-GPU
run's numbers. A CPU test runs PR #40's own `train_stages` on 8 gloo processes for 2 Boost and 2 Cut updates, its
all-reduces summing in rank order as the port does, and finds bit-identical losses and weights (on GPUs, NCCL's
order changes only the rounding). Differences from PR #40:
- **Seed:** the job's seed replaces the module's `SEED` constant (seed 0 is the record's). Data order and held-out set
  are the records' (`randperm(seed 1)`, the last 32), which the end merge scores.
- **The end merge** runs after the last update inside the clock, as in the record. Its means are the shared loop's
  (the same rule), and its held-out scorer adds the 32 sequences rank by rank, as the record's all-reduce does.
  `final_plain.pt` keeps the weights before the merge, scored as `val_loss_plain` (the record's 3.703 to 3.676).
- **E6** (`rows`, `row_offset_per_step`, `drift_every`) and **E7** (`rows_start`, `rows_full_at`) act only on the
  Cut phase's chain, at its nine sites (the embedding and the 8 mid streams). Boost, the head-only top probe and the
  key, gate, query and future probes keep their budgets. A token's credit at a site comes from its rows of the site's
  signed basis, scaled by 512 / rows. E6 reads window `s mod K` of bases held for a cycle of K updates. E7 grows the
  rows from `rows_start` at the first Cut update to 512 at the fraction `rows_full_at` of the run, in steps of 64;
  token t reads the window starting at `(t * rows) mod 512`, so every row is read in every update. Every shape the
  run will use compiles in the warmup. The full cycle runs the record's `chain_credits` itself. CPU tests check that,
  that the K windows of a cycle average to the full cycle, and that each tiled token gets its window's credit.
- **Cost on one H100:** about 1.6 s per Boost update and 16 s per Cut update (8 times the published per-GPU times),
  and about 8 minutes for the end merge. A 1M-slice run takes about 35 minutes; the full 10M epoch about 4.6 hours.

**E3, populations** (task `population`). The job runs the batch-4 baseline's first 20% itself (the same code and
seed as `e1.bp_b4`, so the same trajectory, about a minute) and branches into 4 copies. Copy c takes every 4th of the
remaining sequences, `rest[c::4]`, in batches of 4. The copies take turns, so compute equals one run. Each copy update
counts as the single run's next update number, which sets its rate and ANVIL II's clock, so the warmdown ends where
the single run's does. Rules: `average` (every N updates per copy, weights only; each copy keeps its optimizer
state), `select` (the worst copy takes the best's weights and optimizer state, by H3), `ensemble` (the mean at the end
if H3 prefers it to the best copy). `population.jsonl` logs every N updates each copy's and each pairwise midpoint's
held-out score, and the barrier (midpoint minus the worse end). Population runs have no H4 files: there is no single
trajectory to checkpoint after the branch.

**E8 and E9** replace ANVIL II's orthogonalization through one hook (`loud/anvil.py`); a CPU test checks the hooked
optimizer reproduces boost-cut's ANVIL II bit for bit, and with both off the proxy uses the original.
- **E8** (`curvature_damping`): every 50 updates, the top 16 singular directions of each block matrix's pending
  update, with the curvature along each (central differences of float32 gradients on 2 of the batch's sequences: 1,536
  passes, about 30 s per refresh) and the across-sequence noise of the gradient along it (H2, averaged until the next
  refresh). Each update's step along a direction is scaled by `1 / (1 + c / (B h^2 lambda))`, lambda in squared
  ANVIL II steps. About 20 minutes per 10M run on one H100 (an estimate: `verify.curvature_smoke` times it).
- **E9** (`fisher_damping`): a damped inverse root of the true Fisher's output-side factor (from a backward with
  labels sampled from the model) multiplies the pending update in place of `polar()`, at the polar factor's norm.
  About 8 minutes per 10M run (an estimate). The plan's kernel version in the Cut port waits on this test: if no
  damping beats the batch-4 baseline by 0.02, it is skipped.

## Review

Every piece and every result goes to two independent reviewers: Codex (run as an unprivileged user on a frozen copy)
and fresh Claude agents. Both get the same rubric and lenses: code, research fidelity, usability and results. Each
finding is triaged in [review/LEDGER.md](review/LEDGER.md). See [review/README.md](review/README.md).

## Differences from the handoff found so far

- The live track-1 record is now **3.338** (PR #46, a forward readout of cross-token credit, 56 min on 8×H100), not
  3.409. Track 2 is now **3.499 in 8 min 20 s** (PR #44, the identity path, then Cut-lite at 8 rows per update).
- The pod image ships torch 2.13. Noise pins torch 2.10.0, and the published runs used 2.10.
  `verify.score_record` (scoring) and the verification runs (training) show whether that matters.
- The backprop target is boost-cut's ledger, 3.70835 at seed 0. The handoff's 3.708 is the same run rounded.
- E8 says to damp before ANVIL II's orthogonalization. `polar()` maps every singular value of the pending update to
  about 1, so damping its top directions there leaves them at about 1 (and slightly enlarges the rest). E8 damps right
  after `polar()` by default; `damp_where: pending` is the literal placement, run once as a control.
