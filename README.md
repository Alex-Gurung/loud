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
| H1 backprop proxy (`loud/proxy.py`) on the shared loop (`loud/train.py`); recipes boost-cut, PR #40, PR #46 | done; CPU tests on a tiny model |
| H2 per-sequence gradients (`loud/pergrad.py`) | done, with an unbiased per-matrix noise scale |
| H3 held-out scorer (`loud/heldout.py`) | done; every call logged to `heldout.jsonl` |
| H4 fraction checkpoints and the record's nine trajectory means (`loud/checkpoints.py`); the record's end merge | done; means scored on H3 |
| H5 identity path, PR #26 on one GPU (`loud/identity.py`) | done; CPU tests: one update equals PR #26's own |
| H5 Cut chain, PR #40 on one GPU (`loud/cut.py`, task `cut`), with v2's `rows`, `row_assign`, `prior` and E13's `row_budget` hook | done; CPU tests: 4 updates equal PR #40's own loop on 8 processes |
| H6 diagnostic toolkit (`loud/diagnostics.py`, task `diagnose`): the plan's five tools and E5's `credit_recovery` | done; CPU tests on tiny models; `configs/h6/`, `configs/e5/` |
| H7 runner, results table, per-experiment plots (`run.py`, `collect.py`, `python3 -m loud.plots <exp>`) | done |
| E1 to E10 | configs in `configs/e<N>/`; see What runs |
| Port checks against their sources (`compare` task) | done |
| GPU queue and workers, one per GPU (`loud/jobqueue.py`, `loud/worker.py`, `worker/`) | done; tested end to end on CPU |
| Independent review (Codex and Claude reviewers, [review/](review/README.md)) | rounds 1 to 6 triaged in [the ledger](review/LEDGER.md) |

Not built, on purpose:
- **E5's implementation in the chain.** The plan says not to build it before the maintainers answer the rules question
  in its section 6 (post it as an issue on qlabs-eng/noise). The diagnostic runs regardless.
- **E9's kernel version** (the curvature term and sampled-label loss from the chain's probes). The plan builds it only
  if the proxy test (`configs/e9/`) beats the batch-4 baseline by 0.02.
- **E12** (30M runs, the plan's P3, for a rented 8×H100 node): its first one-GPU run, backprop on 30M with record #3's
  recipe and end merge, waits in `configs/followups/e12/`, which a directory submission skips.

Nothing has run on a GPU yet. Every config together is 184 one-GPU jobs, roughly 55 GPU-hours by the estimates in
the configs (most of it the Cut chain, about 35 minutes per 1M-slice run; a 10M proxy run is about 8 minutes), so
about 14 hours on four workers or 7 on eight. They write about 95 GB (the final weights, about 0.15 GB, and the H4
checkpoints, about 0.4 GB, per training run); workers pause when the shared volume has 15 GB left. All estimates
scale the published 8×H100 times; the first runs measure them.

## Running on GPUs

Workers are Kubernetes Jobs, launched from the root node as in stylo. Each worker runs queued jobs one at a time on
one GPU, from `artifacts/queue/` on the shared volume. A worker exits after 20 idle minutes; EIDF reclaims idle GPUs
after about 30.

```bash
kubectl -n eidf107ns create -f worker/gpu_worker.yaml       # one H100 (comparable wall-clocks); start several
kubectl -n eidf107ns create -f worker/gpu_worker_any.yaml   # one H100 or H200 (jobs without a gpu: requirement)
kubectl -n eidf107ns create -f worker/gpu_worker_2.yaml     # a two-H100 pod: two workers, one per GPU
```

The image is stylo's pinned `vllm/vllm-openai:v0.32.0`, used only for its PyTorch stack (torch 2.13). Nothing is
installed at startup. Each pod writes its startup phases to `artifacts/queue/bootstrap_logs/<pod>.log`, then checks the
pinned external code, the downloads and all three data splits (`python -m loud.refs --check`), and stops if any is
missing or changed. `loud.status` lists pods whose startup failed.

From the CPU pod, in `/mnt/disk/loud` (plain `python3`; no torch needed):

```bash
python3 -m loud.submit configs                  # queue every seed of every config (idempotent)
python3 -m loud.submit configs/e1 --dry-run     # what each job would do, checked against the queue
python3 -m loud.status                          # workers, running, queued (and why), failed, done
python3 collect.py                              # the results table (also artifacts/results/table.md)
python3 -m loud.plots e4                        # an experiment's plots and tables, in artifacts/plots/e4/
```

The CPU pod loses its pod-local disk on a restart: `scripts/cpu_env.sh` rebuilds the CPU Python for tests and plots
(`/opt/loud-cpu-venv`) and the reviewer user (`review/codex.sh` runs it itself).

On a GPU you already have: `python run.py --config configs/verify/bp_b8.yaml`. A direct run works in
`s<seed>.direct<k>/`. It becomes `s<seed>/` when it finishes, unless that exists or the queue holds the same job.

### What runs, and the gates

The plan says to stop if a verification disagrees by more than seed noise (0.01), so every experiment cell waits for
its gates. A job runs only when each job in its `after` list is done and passed its own check. The comparison jobs
wait only for their runs to finish (`after_done`), not to pass, so they still run when a final-loss gate fails, which
is when they are needed to tell a port error from an environment difference.

1. **Smoke runs.** `verify.smoke`: a complete 57-update proxy run on the 1M range through every H1–H4 path, the saved
   means and optimizer state and official scoring; every other job waits for it. Each new method has its own complete
   short run before its cells: `identity_smoke`, `cut_smoke` (128 token-assigned rows with the identity prior, H4,
   the end merge),
   `population_smoke`, `curvature_smoke` and `fisher_smoke`.
2. **Verification 1**, `verify.score_record`: the record checkpoint must rescore to 3.94481 (val) and 3.93696 (test),
   within 1e-5.
3. **Verification 2**, `verify.bp_b8`: the proxy on the 10M set, one sequence per forward (the 8-GPU layout). Target:
   boost-cut's ledger run `bp38_s0` of `backprop_preset.py`, val 3.7083 and test 3.7014, within 0.02. Three checks
   use the per-step logs:
   - `verify.bp_b8_ledger`: the training loss at every update the ledger logged (every 61) agrees within 0.01;
   - `verify.bp_b8_script` and `verify.bp_b8_onefwd`: the unmodified script, and the proxy with whole-batch forwards
     like it;
   - `verify.bp_b8_match`: those two agree within 0.002 for 50 updates, and within 0.01 at the end.
4. **Verification 3**, `verify.identity`: the identity path for a full 10M epoch, against PR #26's 4.1772 (within
   0.02). `verify.identity_match` compares its first 20 updates with PR #26's published 8-GPU log.
5. **The Cut chain's check**, `verify.cut_prefix`: the 10M schedule for 262 updates (all 242 Boost and 20 Cut), held by
   `verify.cut_prefix_match` against PR #40's published 8-GPU log: within 0.003 over the first 20 updates (the plan's
   steps 2 and 10) and within the plan's seed noise, 0.01, over all 262 (step 100 and the first Cut updates). Same
   seed, data and draws, so only float rounding separates them. No measurement sets those bounds yet:
   `verify.cut_prefix_reversed` runs the same prefix with the 8-way gradient sum in reversed rank order, and
   `verify.cut_prefix_reversed_match` reports how far rounding alone moves the losses, the reference for relaxing
   them.
6. **E1**, the backprop baselines at batch 8 and batch 4 (the same rates), three seeds each, with held-out scoring
   (`bp_b8` seed 0 keeps the H4 checkpoints and ANVIL II's state for the diagnostics); `e1.identity`, seed 0.
7. **H6** on `e1.bp_b4.s0`'s checkpoints (`configs/h6/`): `credit_dump` and `cos_to_autograd` at all five fractions;
   `token_map_svd`, `linearity_check` and `direction_stats` at 20% and 100%. Each measures held-out sequences.
   `direction_stats` takes the top directions of the pending update (from the optimizer state `bp_b4` saves with
   each checkpoint); its curvature is a central difference of autograd gradients, because flex attention has no
   double backward.
8. **E5's diagnostic** (`configs/e5/`): `credit_recovery` at 20% and 100%; `python3 -m loud.plots e5` prints the plan's
   9 × 2 table.
9. **Proxy experiments**, three seeds each, against their E1 baseline: E2 (`smooth_tau` 1, 3, 10) and E4 (`reject`,
   `adapt`, `adapt_t60`, and both without the warmdown at seed 0) and E10 (`cap`, `agree` at 0.6, 0.7, 0.8) at batch 8;
   E3 (five population rules, and `bp_b16`, the single batch-16 run averaging should land near) and E8 (three
   dampings and the literal placement) and E9 (three dampings) at batch 4; E10 also counts agreement over all tokens
   (`agree_all_t60/70/80`), the plan's literal reading. E8 also
   waits for H6's `direction_stats`, which the plan runs first and which says where its knob should sit.

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
- **Disk.** Workers pause when the shared volume has less than `LOUD_MIN_FREE_GB` free (default 15, headroom for the
  other projects on the volume), and `loud.status`
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
  cut.py         H5 Cut chain (PR #40)                                diagnostics.py  H6 and E5's diagnostic
  variants.py    E2, E4, E10 (proxy options)                          population.py   E3: copies of the proxy
  curvature.py   E8: steps by curvature-to-noise (proxy)              fisher.py    E9: true-Fisher left factor (proxy)
  anvil.py       ANVIL II with a hook for E8, E9                      plots.py     per-experiment plots and tables
  reference.py   run a pinned external script unmodified
  options.py     task registry, option tables and checks (no torch)   run.py       H7 runner
  collect.py     H7 results table                                     refs.py      pinned external imports
  jobqueue.py, worker.py, submit.py, status.py, snapshot.py            the GPU queue
configs/         one YAML per variant; seeds listed inside
review/          the review rubric, lenses, Codex runner, reports and ledger
worker/          Kubernetes Job YAMLs and the pod bootstrap
scripts/         fetch_external.sh, prepare_data.sh, prepare_test.py, cpu_env.sh (CPU)
tests/           queue, worker, harness and method tests (CPU)
external.json    pinned commits and file hashes of everything imported from other repos
external/  data/  artifacts/        shared-volume state, gitignored
```

Recreate `external/` with `scripts/fetch_external.sh` (it clones Noise again at PR #26's commit as
`external/noise-pr26`), and the data with `scripts/prepare_data.sh`. Run the tests with
`PYTHONPATH=. /opt/loud-cpu-venv/bin/python -m pytest tests`, about 13 minutes on this pod. The method tests use
tiny models and short sequences, with dense attention instead of flex attention (which has no CPU backward); the
Cut chain's checks run it eagerly (no compilation) with a 512-token vocabulary.

## Run configs

```yaml
experiment: e1
variant: bp_b4          # one change per variant; the file is configs/<experiment>/<variant>.yaml
task: proxy             # proxy | identity | cut | population | score | reference | compare | diagnose | noop
seeds: [0, 1, 2]
priority: 50            # lower runs first
after: [verify.bp_b8.s0]
gpu: H100               # optional: only workers whose GPU name contains this take the job
args: {recipe: boostcut, sequences: 10M, heldout: 32, batch: 4, checkpoints: [0]}   # H4 for seed 0 only
expect: {val_loss: [3.7083, 0.01]}   # optional gate; recorded as pass/fail in result.json
```

To add a variant, copy a config, change `variant` and the one option under test, and submit it. Unknown keys and
options, bad values, an `expect` on a metric the task does not produce and duplicate job ids all fail at submission.

Every training run writes:
- `config.json`: the spec, the code snapshot and its git state, the pinned external commits, torch and the GPU;
- `metrics.jsonl`, one line per update: training loss, step time, the effective rates and ANVIL II's beta, plus
  experiment series such as `grad_noise` (H2);
- `heldout.jsonl`, `final.pt` and the H4 files;
- `result.json`: the official validation loss from `utils/eval.py` in a fresh process, the test loss, the held-out
  loss, each mean's held-out loss, the training time and the check.

The `reference` task's `val_loss` is the script's own in-process `evaluate()`, the same computation.

Options for every task are in `loud/options.py`, one table per task with a comment per option.

## Methods

**The proxy** (`recipe: boostcut`, the default) takes model construction, init, ANVIL II + Adam and rates verbatim
from boost-cut's `train/boostcut/run.py`, the module `backprop_preset.py@539386e` imports. `recipe: pr40` instead
takes PR #40's rates and betas, its warmdown to 30% and head std 0.02, and batched ANVIL II; `recipe: pr46` the 30M
record's (the same, with ANVIL II weight decay 0.75). With `end_merge` (pr40 and pr46) the record's own end merge
runs after the last update, timed, on the 32 held-out sequences; `final_plain.pt` keeps the weights before it.

- **One GPU = eight.** The 8-GPU script gives each GPU one sequence and averages the float32 gradients. Here each
  micro-batch is one sequence, and the same float32 mean is formed before the cast to each parameter's dtype; a CPU
  test checks this exactly. The loss is the batch mean, which answers the plan's open question: mean, not sum.
- **Data order and held-out set.** Every record permutes its range with `randperm(N, seed 1)`, for every seed (the
  seed changes the init and the draws). The held-out set is the last 32 sequences of that order, as in PR #40.
  Verification runs use `heldout: 0`, exactly as the published runs. Experiment cells use 32, so their baselines and
  variants train on the same 4,848 sequences.
- **Matching a published log** uses the 10M range with `stop_after`. `sequences: 1M` permutes rows 0 to 487 on its
  own, so its updates are not the 10M log's first updates.
- **H4**: `checkpoints: true` keeps the five fraction checkpoints (about 0.4 GB per run), `checkpoints: [0]` only
  seed 0's (the ones the diagnostics read; the disk is shared). They store 2-D weights in one dtype (bf16 by default,
  the same forward under bf16 autocast); `final.pt`, the scored weights, is always exact. `ckpt_100.pt` exists only
  for complete runs. Verification runs reproduce the published layout and keep only `final.pt`.

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
  the weights and all optimizer state (ANVIL II's rails and energy, Adam's moments and counts) are restored from a
  snapshot, and the batch feeds no update. ANVIL II's clock, which drives its momentum schedule, keeps following the
  loop's update count, as the warmdown does. `adapt` also multiplies every rate (and so ANVIL II's decay) by
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
one update over two virtual GPUs equals PR #26's FastRelay at ranks 0 and 1 with `Boost.step`'s sum. It differs from
PR #26 only in the data order: the records' fixed order, `randperm(seed 1)`, for every seed (PR #26 shuffles with
`seed + 1`; the two are identical at seed 0). v2 uses it for E1's single run and verification 3.

**E3, populations** (task `population`). The job runs the batch-4 baseline's first 20% itself (the same code and
seed as `e1.bp_b4`, so the same trajectory, about a minute) and branches into 4 copies. Copy c takes every 4th of the
remaining sequences, `rest[c::4]`, in batches of 4. The copies take turns, so compute equals one run. Each copy update
counts as the single run's next update number, which sets its rate and ANVIL II's clock, so the warmdown ends where
the single run's does. Rules: `average` (every N updates per copy, weights only; each copy keeps its optimizer
state), `select` (the worst copy takes the best's weights and optimizer state, by H3), `ensemble` (the mean at the end
if H3 prefers it to the best copy). `population.jsonl` logs every N updates each copy's and each pairwise midpoint's
held-out score, and the barrier (midpoint minus the worse end). H4: the trunk's checkpoints up to the branch, then each
copy's at 50% and at its end (`ckpt_50.c<k>.pt`); no trajectory means, since the rules are E3's own merges.

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
  (the same rule), and its held-out scorer adds the 32 sequences rank by rank, as the record's all-reduce does
  (with `merge_batch`, in one forward of all 32, which changes only the bf16 rounding of close calls and saves most of
  the merge's ~8 minutes). Each of its scores is logged to `heldout.jsonl` (reason `merge`). `final_plain.pt` keeps
  the weights before the merge, scored as `val_loss_plain` (the record's 3.703 to 3.676).
- **The probe budget** (v2's H5 flags; the winners of the H8 experiments are confirmed with them) acts only on the Cut
  phase's chain, at its nine sites (the embedding and the 8 mid streams). Boost, the head-only top probe and the key,
  gate, query and future probes keep their budgets. With `rows` m below 512, a token's credit at a site is decoded from
  m rows of the site's freshly signed basis, scaled by 512 / m (unbiased, through the basis's random column signs).
  `row_assign: time`: every token of update s reads rows from (s mod K) m, K = 512 / m, rotating across updates;
  `token`: token t reads rows from (t mod K) m, so every update covers the full basis. `prior: identity` (E7): at the
  mid sites the rows measure the residual of the identity-path guess (the top stream's credit carried down the
  residual scalars, as H6's `credit_dump` guesses it), and the guess is added back; the embedding site has no guess
  and keeps the plain estimate. `row_budget` (E13's hook) names a `loud.cut.ROW_BUDGETS` strategy that gives each token
  its own count up to `rows`; the chain still evaluates the rows past a token's count, so it changes the estimate, not
  yet the cost. The full cycle runs the record's `chain_credits` itself. `chain_chunk` compiles one graph per site and
  moves per token (17 for one row count), all in the warmup, which fails if any planned graph is missing. CPU tests
  check that the full cycle is the record's, that the K time windows of one basis average to the full cycle (with the
  prior too), that each token-assigned token gets its window's credit, that a budget reads the token's first rows,
  and that the run's budget is built after the warmup and sees each Cut update's batch after the update.
- **Cost on one H100:** about 1.6 s per Boost update and 16 s per Cut update (8 times the published per-GPU times),
  and about 8 minutes for the end merge (less with `merge_batch`). A 1M-slice run takes about 35 minutes; the full
  10M epoch about 4.6 hours.

**E5's diagnostic** (`credit_recovery`, on a checkpoint, one held-out sequence). Each token's raw embedding takes a
full cycle of 512 probes (signed Hadamard rows times PR #40's step, 0.01 of the token's RMS), run forward as PR #40's
chain runs its cycles: forward differences against the clean pass, each MLP through the record's 512 routed units,
bf16 products (and float32 for comparison), every attention output frozen as the plan says (so without the chain's
own-query response). At each later site, the probes' changes there and the token's loss changes give a 512 × 512
system; its ridge solution toward the identity-path guess (ridges 0 to 1e-3, one SVD per site) is compared with a
direct cycle at that site, run the same way. A second launch, block 0's cycle read at mid1 to mid7, asks whether two
cycles could stand in for nine. The probe at the embedding also moves x0, which every block reads, so recovery from
it is biased at the bottom sites; a reviewer measured 0.83 to 0.92 at mid0 to mid2 on the 10M record's checkpoint
(128 tokens, on CPU), against 0.9999 for the mid0 launch at every site. The runs on E1's checkpoints decide.

**Plots** (`python3 -m loud.plots <exp>`) read only the run files. Every experiment gets the final validation loss per
variant (seeds and mean ± sd, with the E1 baseline and its −0.02 line where the plan compares against one) and the
train and held-out curves. Added where the plan names a series: E2's train/val gap, E3's basin barriers, E4's
acceptance rate per 50 updates and effective rate against the hand schedule, E6's and E7's loss and time against
the full cycle (`e1.h8_synth`), E8's damping, E12's validation loss over time with the 3.5 line and time to 3.5; E5
and E7 diagnostic tables, E8's h²/c per matrix type.

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
- E4's factors 1.05 and 0.8 hold steady at 82% acceptance, not the 60% the plan targets
  (a · ln 1.05 + (1 − a) · ln 0.8 = 0); `adapt_t60` keeps 1.05 and uses 0.9294 so that 60% is the steady state.
- E5's embedding launch is biased at the bottom sites by x0's skip paths (see E5 above); a mid0 launch is reported
  beside it.
- E8 says to damp before ANVIL II's orthogonalization. `polar()` maps every singular value of the pending update to
  about 1, so damping its top directions there leaves them at about 1 (and slightly enlarges the rest). E8 damps right
  after `polar()` by default; `damp_where: pending` is the literal placement, run once as a control.
