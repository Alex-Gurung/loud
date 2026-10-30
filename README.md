# loud: a one-GPU research harness for the Noise benchmark

Implements the [engineering handoff v2](Noise%20Forward-Only%20Research%20Harness%20Engineering%20Handoff%20v2.md)
(the plan; [v1](Noise%20Forward-Only%20Research%20Harness%20Engineering%20Handoff.md) is kept for its history): a
single-GPU harness for [Noise](https://github.com/qlabs-eng/noise), pretraining a fixed 37.7M-parameter GPT without
backpropagation. It includes a GPU job queue and workers that follow the stylo setup (Kubernetes Jobs on EIDF, the GPU
cluster, sharing one volume).

**Terms.**
- **Tracks 1–3:** Noise's leaderboard tracks. Track 1 is the validation-loss record at fixed data, track 2 the
  speedrun to 3.5, track 3 architecture changes.
- **1M, 10M, 30M:** the training ranges, the first 488, 4,880 and 14,648 sequences of `train.pt` (about that many
  tokens). The 10M record is track 1's record on the 10M range (PR #40).
- **Sites:** S0, the raw embedding, and S1 to S8, each block's mid stream (after attention, before the MLP).
- **Boost and Cut:** the two phases of the 10M record (PR #40). Boost, for the first 20% of updates, trains each
  sublayer against the shared head's readout of the stream right after it. Cut then measures each site's credit (the
  loss gradient there) with forward probes.
- **Proxy:** backprop with the record's optimizer, a development stand-in, never an entry.
- **H8:** an autograd emulation of the 10M record's Boost and Cut (`loud/h8.py`): ordinary autograd on the *cut
  graph*, the fixed network with the inputs to every key, value and gate held, which is what the chain measures. It
  runs at backprop speed, so most forward-only questions are asked there first and only winners go to the real chain.
- **Identity path:** track 2's forward-only estimator (PR #26). One Hadamard cycle of 512 probes at the top stream,
  with credit handed down the residual scalars.
- **Cut chain:** track 1's 10M record estimator (PR #40), the real forward-only chain (`loud/cut.py`).
- **ANVIL II:** the records' optimizer for block matrices.
- **boost-cut:** the repository with the backprop baseline script (`backprop_preset.py`), the test split and the
  ledger of Boost and Cut runs.
- **H1–H8 and E1–E16:** the plan's infrastructure pieces (its section 3) and experiments (section 5).
- **P0–P3:** the plan's priorities (section 4). P0 is the infrastructure and verifications, P1 the baselines and first
  experiments, P2 the second set, P3 work that runs only if earlier results call for it (E9, E5's implementation,
  E12).
- **Verification 1–3:** the plan's three checks before any experiment (section 2).

## Status

| Piece | State |
| --- | --- |
| Data: `train.pt`, `val.pt` (pinned hashes), `test.pt` (boost-cut's split, `cb7a29f0…`) | built in `data/` |
| Record checkpoint `track1_boostcut_final.pt` (`499dff9f…`) | downloaded, hash checked |
| H1 backprop proxy (`loud/proxy.py`) on the shared loop (`loud/train.py`); recipes boost-cut, PR #40, PR #46 | done; CPU tests on a tiny model |
| H2 per-sequence gradients (`loud/pergrad.py`) | done, with an unbiased per-matrix noise scale |
| H3 held-out scorer (`loud/heldout.py`), on all 32 rows or a logged subset | done; every call logged to `heldout.jsonl` |
| H4 fraction checkpoints (per seed), ANVIL II's state, the record's nine trajectory means (`loud/checkpoints.py`) | done; means scored on H3 |
| H5 identity path, PR #26 on one GPU (`loud/identity.py`) | done; CPU tests: one update equals PR #26's own |
| H5 Cut chain, PR #40 on one GPU (`loud/cut.py`), with v2's `rows`, `row_assign`, `prior` and E13's `row_budget` hook | done; CPU tests: 4 updates equal PR #40's own loop on 8 processes |
| H6 diagnostic toolkit (`loud/diagnostics.py`, task `diagnose`): v2's tools on H8's cut graph and reforward | done; CPU tests on tiny models |
| H7 runner, results table, per-experiment plots and tables (`run.py`, `collect.py`, `loud.plots`) | done |
| H8 (`loud/h8.py`): cut, synthesis and Boost modes, probing schemes, `reforward`; the track-3 model (`loud/arch.py`) | done; CPU tests: synthesis matches PR #40's Cut (cosine ≥ 0.996 per matrix), Boost matches PR #40's |
| E1 to E16 | configs in `configs/e<N>/`; see What runs |
| Port checks against their sources (`compare` task) | done |
| GPU queue and workers, one per GPU, two-GPU pods (`loud/jobqueue.py`, `loud/worker.py`, `worker/`) | done; tested end to end on CPU |
| Independent review (Codex and Claude reviewers, [review/](review/README.md)) | triaged in [the ledger](review/LEDGER.md) |

Waiting, on purpose:
- **E5's implementation in the chain** (P3): only if its diagnostic passes (all sites at cosine 0.95 or more, or a
  hybrid). Whether its decoded credit may feed nonlinear consumers is a rules question to settle then.
- **E8's intervention and E9** wait in `configs/followups/`, which a directory submission skips: E8's (both settings,
  proxy and H8, with their smoke runs) only if its diagnostic shows partial rows pushing a large share of the
  momentum's energy below SNR 1, E9 (P3) only if E8 shows per-direction scaling matters with exact gradients. E9's
  kernel version follows only if it beats `e1.bp_b8` by 0.02.
- **E14's and E15's follow-ups** (the teacher and onset variants) wait in `configs/followups/` for the first arms.
- **E12** (30M, P3, rented 8×H100): there is no 8-GPU path; its one-GPU backprop run waits in `configs/followups/e12/`.
- **Confirming a winner on the real chain** (the Cut chain on the 1M slice): its configs are written once H8 names
  the winners, with the H5 flags (`rows`, `row_assign`, `prior`, `row_budget`).

**First queue (2026-10-29, one H100, four jobs at a time):** all 147 runnable jobs finished, none failed, and 15
were skipped by their gates; it took about 8 hours of wall-clock. Every verification passed (verification 1 at
1e-4 under torch 2.13). Shared-GPU wall-clocks ran: 10M backprop 2 to 5 minutes, H8 cut and synthesis 1 to 2, the
identity path 7, E11's runway 21, the track-3 model 4 to 6; `collect.py` flags them `shared GPU`, so they bound the
plan's estimates without replacing them. The results, and each experiment's verdict, are on the board (below) and
in `python3 collect.py`; review round 8 (`review/LEDGER.md`) checked them. They write about 19 GB: each run's
`final.pt` is scored exact, then keeps its matrices in bf16 (`final_dtype`; 75 MB, against 130 MB exact; the
verifications and E1 keep theirs exact), the track-3 model's is about 1.2 GB, and H4 checkpoints exist only where a
diagnostic reads them (`e1.bp_b8.s0` with ANVIL II's state, `e1.h8_synth.s0`), in bf16. Workers pause when the
shared volume has 15 GB left.

**Follow-ups (2026-10-30, three pods, autonomous loop).** Each answers a question the first queue left open; the board
has every arm.
- **E17, where H8's cut loses credit:** of the 0.160 gap from `e1.h8_cut` to matched backprop, keys live in every
  block recover 71%, values 25%, both 91% (means of three seeds, per-seed spread about 3 points). The key path's
  share sits in three blocks: keys live only in block 3 recover 60%, block 5 64%, block 7 55%, each of the other
  five 1-2%, and blocks 3, 5 and 7 together 68%. The `key_paths` diagnostic shows the same split in one gradient: on
  the cut model's final weights block 5's key path removes 54% of the cut gradient's error, block 3 23%, block 7
  10%, the rest under 1%; at the 20% handoff it is blocks 1, 3 and 5 (15% each) and 7 (4%), the blocks with value
  embeddings, whose values carry the token itself, so their keys need context that earlier blocks must learn to
  write. Values are spread out: the value-embedding blocks' values recover 18%, the others' 10%; keys and values in
  blocks 3, 5, 7 recover 80%. Read through fewer rows (the record's input-key probes take 512 per token at every
  block), shared rows recover -4/27/49/63% at m = 32/64/128/256 and token-assigned rows 9/37/54% at 32/64/128;
  blocks 3, 5, 7 through 256 token-assigned rows still recover 63%, under a fifth of the record's key probes. At
  30M keys recover 71% of the (smaller) gap on both seeds. Running on the real chain: PR #40 unchanged at full
  length and with its key probes at blocks 3, 5, 7 or at block 5 only (`key_blocks`): Cut updates already take
  about 18% less time with them restricted.
- **E18, 30M:** H8 cut's gap to matched backprop shrinks from 0.160 to 0.123 (two seeds, 0.126 and 0.120); E6's
  partial-row cost holds or grows (K = 4 by token +0.158, by time +0.188, against +0.150 and +0.184 at 10M);
  NextLat helps more with the longer run: -0.005 at lambda 0.3, -0.008 at 1 and -0.007 at 3 (two seeds each; at 10M
  lambda 1 and 3 cost +0.008 and +0.048), so an auxiliary loss that hurts a short run pays in a longer one, though
  it levels off under E14's 0.02 bar. Keys live recover 71% of the 30M gap (both seeds), keys at blocks 3, 5, 7
  only 64%.
- **E10:** a binding clip (1x the median) costs +0.233 and agreement at 55% keeps 9.8% of the entries and costs
  +0.78: neither rule helps once it acts.
- **The last 9% of E17:** keys and values live with Boost off lands on matched backprop to 0.0001 (three seeds),
  so H8's machinery is exact and the rest of the gap that keys and values leave is Boost's first 20%, which costs
  about 0.019 when the remainder is backprop (in the cut setting Boost is neutral).
- **E20, the batch check:** at batch 8, K = 4 by token still costs +0.148 against every row at batch 8 (three
  seeds; batch 4: +0.150), so the partial-row cost does not shrink when an update averages more tokens: the rows'
  noise scales like the tokens' own gradient noise. Batch 16 waits for GPU memory (44 GB) beside the chain runs.
- **E11:** with the runway grid widened to 0.08-1.28 the rule acts (79-93% of tokens per site take the largest
  move) and costs +0.289 (three seeds, +0.26 to +0.30): weighting credit by runway hurts.
- **E4:** a 16-row judge moves `adapt_nowd` from +0.355 to +0.337: the judge's noise is not what fails.
- **E7:** the identity prior at m = 64 costs +0.149 and at m = 128 +0.086; it helps (E6's m = 128 costs +0.150)
  but not to 0.02.
- **E5 (diagnostic): passes.** Read with the chain's way of differencing in bf16 (`readout_delta`, each change
  carried as bf16 products), one probe cycle at S1 recovers every later site's credit at median cosine 0.997 or more
  at 20% and 0.9999 at 100% (float32: the same to 3 decimals; whole bf16 copies: 0.01-0.05, which is the rounding of
  that method, not of the chain). The 10th percentile at S7-S8 at 20% (0.65-0.71; float32 0.78-0.84) is the
  model's nonlinearity early on. By v2's rule this means implement, with the linear-consumer question open.
  **Built** (`deep_readout` in the Cut chain, loud/cut.py): S1's full cycle runs through a copy of the record's chain
  kernel that keeps each move's change at S2-S8, each token's 512 x 512 system is solved with v2's ridge, and the
  cycles at S2-S8 are skipped; S0's cycle and every mixing scalar's 4 moves stay. On the CPU test model S0, S1 and
  the scalars equal the record's and S2's solved credit matches its measured one per token (median cosine > 0.99).
  The solved credit is least sure along the stream itself, so the scalars keep their moves (with v2's ridge the
  residual scalar's credit came out 3x off). Queued: a GPU smoke, then the full 10M run against `e17.chain_record`.
- **E8:** in H8, SNR scaling makes partial rows worse (it recovers -92% of their cost) and trails the plain rate
  control; E9, gated on E8, does not run.
- **E19, the rate check:** PR #40's rates are about the best for both estimators: H8 cut at 0.5/0.7/1.4/2x costs
  +0.066/+0.022/+0.011/+0.068 (three seeds) and E6's K = 4 by token at 0.35/0.5/0.7/1.4x +0.14/+0.07/+0.018/+0.009
  against itself at 1x (two seeds). A noisier gradient does not want a smaller step here (ANVIL II's polar factor
  sets the step's size), so neither the cut's gap nor E6's +0.15 is the rate.

## Running on GPUs

Workers are Kubernetes Jobs, launched from the root node as in stylo. A worker pod pulls jobs from
`artifacts/queue/` on the shared volume. Every job is one run (one arm, one seed), so work is sharded by run: pods can
come and go at any time, each claims whatever is next, and more pods simply finish the queue sooner. The jobs are
small (a 38M-parameter model at 4 to 8 sequences per update leaves an H100 mostly idle), so each pod runs
`LOUD_JOBS_PER_GPU` of them side by side (4 in the YAMLs); a job reserves 16 GB of the GPU unless its config says
(`gpu_memory_gb`, 48 for the track-3 model), and a worker claims only what fits beside its GPU's other jobs. Set
`LOUD_JOBS_PER_GPU` to 1 for runs whose wall-clock is the result. Workers take an H100 or an H200: every experiment
here compares losses, and `collect.py` flags a variant whose seeds ran on mixed GPUs; a timing comparison pins its
configs with `gpu: H100`.

**Start, add, stop** (from the root node):

```bash
kubectl -n eidf107ns create -f worker/gpu_worker.yaml       # start: one H100 or H200, 4 jobs at a time; again to add
kubectl -n eidf107ns create -f worker/gpu_worker_2.yaml     # or a two-GPU pod (8 jobs at a time)
kubectl -n eidf107ns get jobs -l loud/role=worker           # what is running
kubectl -n eidf107ns delete job -l loud/role=worker         # stop now: each running job goes back to the queue
```

To stop without losing the runs in flight, drain instead: `python3 -m loud.drain on` (from the CPU pod) lets every
worker finish its current job and exit; `python3 -m loud.drain off` before launching again. A job stopped mid-run
(pod deleted, preempted) starts again from its first update on the next worker, with its earlier attempt kept for its
log; runs are minutes long, so nothing larger is lost. To send compute to one experiment first, resubmit its configs
with a lower priority (`python3 -m loud.submit configs/e6 --priority 20`): queued jobs take the new priority.

Nothing restarts a worker, so relaunch it with the same command whenever it stops:
- A worker exits after 20 idle minutes with nothing running anywhere and no ready job (it stays while other jobs run,
  for the work they unblock), or with the disk below its floor (EIDF reclaims idle GPUs after about 30). `loud.status` shows it as `exited: idle (…)`.
- A pod that dies (preempted, crashed or deleted) is not restarted (`backoffLimit: 0`, `restartPolicy: Never`).
  `loud.status` shows its worker as `LOST`.
- A dead worker's job stays `running` until its 5-minute lease expires, and it is queued again only when a worker
  next claims (`LEASE EXPIRED` in `loud.status`). With no worker running, nothing claims, so relaunching a worker is
  what moves the job.

**Disk.** Workers pause when the shared volume has less than `LOUD_MIN_FREE_GB` free (default 15 GB, headroom for the
other projects on the volume), then exit after 20 idle minutes. `loud.status` shows the free space, and a paused
worker as `waiting: shared disk nearly full`. To change the floor, edit `LOUD_MIN_FREE_GB` in the
`worker/gpu_worker*.yaml` you launch. If the volume runs low, free what no job reads: superseded runs
(`artifacts/runs/<experiment>/<variant>/s<seed>.superseded<k>/`, left by `--rerun`) and failed or abandoned attempts
(`s<seed>.attempt<k>/`, kept only for their logs). Never remove a running job's attempt (`loud.status` lists it) or a
finished run's `s<seed>/`. `ls -d artifacts/runs/*/*/s*.superseded* artifacts/runs/*/*/s*.attempt*` lists them.

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
python3 -m loud.board                           # the experiment board's data, in artifacts/board/
/opt/loud-cpu-venv/bin/python -m loud.plots e4  # an experiment's plots and tables, in artifacts/plots/e4/
```

The plots need matplotlib, which plain `python3` on this pod lacks, so they run under the CPU venv (below).

**The first v2 submission.** The queue on the shared volume still holds v1's jobs, all cancelled, and submitting
leaves a cancelled job cancelled unless you pass `--rerun`. Without it, the verifications and many other jobs stay
cancelled and everything else waits on them. So the first v2 submission is `python3 -m loud.submit configs --rerun`.
Nothing has finished, so `--rerun` supersedes nothing. Submit without `--rerun` after that. Both a submission and a
dry run give the reason for each job they leave as it is, e.g. `unchanged (cancelled; resubmit with --rerun)`.

**The experiment board** ([claude.ai link](https://claude.ai/artifact/YFQSqsRQV7HzjqfjMgSRtf), private to its
owner until shared) shows every experiment's question, setup, success test, gate, job progress and verdict, each
arm's final loss against its reference cell, the held-out curves and every job's queue state. `loud.board` writes its
tables (`experiments`, `arms`, `runs`, `curves`, `jobs`) from the configs, the queue and the run files; the page is
`loud/board.html`. Uploading new tables refreshes it (Claude does this when runs finish).

The CPU pod loses its pod-local disk on a restart: `scripts/cpu_env.sh` rebuilds the CPU Python for tests and plots
(`/opt/loud-cpu-venv`) and the reviewer user (`review/codex.sh` runs it itself).

On a GPU you already have: `python run.py --config configs/verify/bp_b8.yaml`. A direct run works in
`s<seed>.direct<k>/`. It becomes `s<seed>/` when it finishes, unless that exists or the queue holds the same job.

### What runs, and in what order

The plan says to stop if a verification disagrees by more than seed noise (0.01), so every experiment cell waits for
its gates. A job runs only when each job in its `after` list is done and passed its own check. The comparison jobs
wait only for their runs to finish (`after_done`), not to pass, so they still run when a final-loss gate fails, which
is when they are needed to tell a port error from an environment difference.

The queue claims by priority, then seed: within a priority band every arm's seed 0 runs before any seed 1, so one GPU
gives a first read of every arm early. The bands, in order (`priority:` in each config):

1. **Smoke runs (1–4).** `verify.smoke`: a complete 57-update proxy run on the 1M range through every H1–H4 path and
   official scoring; every other job waits for it. Each method has its own short complete run before its cells:
   `h8_smoke`, `nextlat_smoke`, `latent_boost_smoke`, `identity_smoke`, `population_smoke`, `blur_smoke`,
   `step_control_smoke`.
2. **Verifications (5–11).**
   - `verify.score_record` (verification 1): the record checkpoint must rescore to 3.94481 (val) and 3.93696 (test),
     within 1e-4. Under the image's torch 2.13 it gives 3.944756 and 3.936905, 5.6e-5 and 5.1e-5 below the torch 2.10
     numbers (kernel rounding; the data, checkpoint and scorer hashes all match).
   - `verify.bp_b8` (verification 2): the proxy on the 10M set, one sequence per forward. Target: boost-cut's ledger
     run `bp38_s0` of `backprop_preset.py`, val 3.7083 and test 3.7014, within 0.02. `verify.bp_b8_ledger` checks the
     training loss at every update the ledger logged (every 61) within 0.02, 0.005 on the mean (rounding over hundreds
     of updates moved one logged update 0.011 on the first run, the sign wandering); `verify.bp_b8_script` (the unmodified
     script) and `verify.bp_b8_onefwd` (the proxy with whole-batch forwards like it) must agree within 0.002 for 50
     updates and 0.01 at the end (`verify.bp_b8_match`).
   - `verify.identity` (verification 3): the identity path for a full 10M epoch, within 0.02 of PR #26's 4.1772;
     `verify.identity_match` compares its first 20 updates with PR #26's published 8-GPU log.

3. **H8's validation (20).** `h6.h8_validation` on `e1.bp_b8.s0`'s 50% checkpoint: H8 synthesis against PR #40's Cut
   estimate (cross-token patches off, every MLP unit routed), every matrix's cosine at least 0.99. Every H8 experiment
   waits for it.
4. **H6 diagnostics (30–40)** on `e1.h8_synth.s0`'s checkpoints, each on held-out sequences: `credit_dump` at 0, 10,
   20, 50 and 100% (E7's precondition: `e7.precondition` passes only if the variance ratio is at most 0.5 at 50% and
   100%), `cos_to_autograd`, `linearity_check` and `synth_vs_real` at 20% and 100%; `snr_spectrum` (E8's diagnostic)
   on `e1.bp_b8.s0`'s.
5. **E1, the baselines (50)**, three seeds: backprop at batch 8 and 4 (`bp_b8` seed 0 keeps the H4 checkpoints with
   ANVIL II's state), backprop at batch 4 on PR #40's recipe (`bp_b4_pr40`, the H8 experiments' backprop end), H8
   synthesis with the identity scheme (`h8_synth`, seed 0's checkpoints feed H6, E5 and E13), H8 cut mode with and
   without Boost; `e1.identity`, one seed.
6. **P1 experiments (55)**, three seeds each, one band, so every arm's seed 0 runs before any seed 1: E6 (`time` and
   `token` row assignment at K = 4, 8, 16) and E7 (the identity prior at m = 64, 128; gated by `e7.precondition`)
   against `e1.h8_synth`; E14 (NextLat at λ = 0.3, 1, 3) and E15 (latent target beside or instead of the lens)
   against `e1.h8_cut`; E2 (`blur` 0.03, 0.1, 0.3), E4 (`reject`, `adapt`, `adapt_nowd`) and E3 (local SGD: `avg1`,
   `avg10`, `avg50`, `select50`, `ensemble`) against `e1.bp_b8`.
7. **The real chain's checks (66)**, which only P2's confirmations need: `verify.cut_smoke`; `verify.cut_prefix`, the
   real Cut chain for 262 updates of the 10M schedule (242 Boost, 20 Cut), held to PR #40's published 8-GPU log within
   0.003 over 20 updates and 0.01 over all 262 (`verify.cut_prefix_reversed` measures how far float rounding alone
   moves them); and `verify.track3_smoke` (the track-3 model at full size, scored with its own model file).
8. **P2 (68–75)**: E5's diagnostic and E13's tiers diagnostic (68); E10 (`clip`, `agree`), E11 (`runway`) and E13
   (θ = 2, 3, 4, each gated on the tiers: the low tier under 10% of the credit energy with over 40% of the tokens) at
   70; E16 (75): the track-3 model under backprop and under H8 cut mode, two seeds each (`e1.bp_b4_pr40` and
   `e1.h8_cut` seeds 0–1 are the fixed model's arms).

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
- **Order.** Lower `priority` first, then lower seed, then submission time.
- **Resubmitting** the same config:
  - a job that has not started takes the newer code, priority, `after` and `gpu`;
  - a crashed (failed) job is queued again;
  - a done job (including one that failed its check) and a cancelled job change only with `--rerun`, which keeps the
    old run as `s<seed>.superseded<k>`. The submission says so: `unchanged (done; resubmit with --rerun)`.

  A submission is checked in full first: duplicate ids, unknown `after` ids, conflicts and config errors stop it
  before anything is written.
- **Disk.** Workers pause below `LOUD_MIN_FREE_GB` free; see Disk under Running on GPUs.
- The harness deletes nothing: attempts, superseded runs, snapshots and job history all stay. Weights are deleted only
  by hand and listed in `artifacts/deleted_weights.txt`. On 2026-10-30, with the volume at 5 GB free, that was the
  smoke runs' weights, E16's track-3 finals and the finals of finished arms E2-E15; every score stays in its run.

### When a job fails or is blocked

- `loud.status` gives the cause: `blocked: verify.bp_b8.s0 failed its check`, `waiting: …`, or a failed job's last
  log line and its log path. Blocking passes down the chain: a job whose dependency is queued but blocked is blocked
  too, and status names the chain (`blocked: h6.h8_validation.s0 <- e1.bp_b8.s0 <- verify.bp_b8.s0 failed its
  check`). So a failed verification 2 blocks every H8 experiment, through `e1.bp_b8` and `h6.h8_validation`.
- A crashed job (under `failed`): fix the cause and resubmit its config; it is queued again.
- A job that ran but failed its check is done, so resubmitting changes nothing: after fixing the cause, resubmit it
  with `--rerun` (for example `python3 -m loud.submit configs/verify/bp_b8.yaml --rerun`). Its dependents become ready
  when it passes.
- A `gate` job that fails its check (`e7.precondition`, `e13.gate_theta*`) is the plan saying skip, not an error. The
  arms that wait on it show as `skipped (gate e7.precondition.s0 says skip)` and never run; cancel them.
- If the gate itself was wrong, edit the dependent configs' `after` and resubmit them: waiting jobs take the new
  dependencies.
- `python3 -m loud.submit --cancel <job id or pattern> …` cancels queued jobs, by id or by a quoted glob pattern on
  job ids (`--cancel 'e7.prior_*' e13.theta2.s1`). With `--dry-run` it lists what it would cancel. An id that is not
  queued is reported, and the others are still cancelled.

## Layout

```
loud/            the harness package (frozen into each job's snapshot)
  train.py       the shared loop: warmup, timing, H3/H4, scoring      data.py      hash-checked splits, order, held-out
  proxy.py       H1 backprop proxy (+ H2 per-sequence gradients)      h8.py        H8: cut, synthesis and Boost modes
  heldout.py     H3                                                   checkpoints.py  H4
  score.py       official utils/eval.py scoring in a fresh process    compare.py   per-update loss check of two runs
  identity.py    H5 identity path (PR #26)                            cut.py       H5 Cut chain (PR #40)
  diagnostics.py H6, E5's and E8's diagnostics                        arch.py      E16: track-3 record #5's model
  variants.py    E2, E4 (proxy options)                               population.py   E3: local SGD over copies
  snr.py         E8: SNR spectrum and SNR-scaled steps (proxy)        fisher.py    E9: true-Fisher preconditioning
  latent.py      E14 NextLat, E15 latent targets (H8)                 ema.py       H4's EMA teacher (E14, E15)
  ngram.py       E13: streaming Witten-Bell n-grams, copies, tiers    anvil.py     ANVIL II with a hook for E8, E9
  reference.py   run a pinned external script unmodified              plots.py     per-experiment plots and tables
  options.py     task registry, option tables and checks (no torch)   run.py       H7 runner
  collect.py     H7 results table                                     refs.py      pinned external imports
  pergrad.py     H2 per-sequence gradients                            gate.py      the gate task: a plan rule as a check
  runlog.py      the files every run writes, and its check            noop.py      plumbing task for queue and workers
  board.py       the experiment board's tables (its page: board.html) paths.py     where shared state lives (LOUD_*)
  drain.py       wind workers down after their current jobs
  common.py      small shared helpers (stdlib only)
  jobqueue.py, worker.py, submit.py, status.py, snapshot.py            the GPU queue
configs/         one YAML per variant (configs/<experiment>/<variant>.yaml), its seeds listed inside
  verify/        the smoke runs and verifications 1-3
  h6/            H6's diagnostics, h8_validation among them
  e1/ ... e16/   each experiment's arms, gates and diagnostics
  followups/     configs the plan runs only after reading earlier results; a directory submission skips them
review/          the review rubric, lenses, Codex runner, reports and ledger
worker/          Kubernetes Job YAMLs and the pod bootstrap
scripts/         fetch_external.sh, prepare_data.sh, prepare_test.py, cpu_env.sh (CPU)
tests/           queue, worker, harness and method tests (CPU)
external.json    pinned commits and file hashes of everything imported from other repos
external/  data/  artifacts/        shared-volume state, gitignored
```

Recreate `external/` with `scripts/fetch_external.sh` (it clones Noise again at PR #26's commit as
`external/noise-pr26`), and the data with `scripts/prepare_data.sh`. Run the tests with
`PYTHONPATH=. /opt/loud-cpu-venv/bin/python -m pytest tests`, 126 tests in about 15 minutes on this pod. The method
tests use tiny models and short sequences, with dense attention instead of flex attention (which has no CPU
backward); the Cut chain's checks run it eagerly (no compilation) with a 512-token vocabulary.

## Run configs

```yaml
experiment: e1
variant: bp_b4          # one change per variant; the file is configs/<experiment>/<variant>.yaml
task: proxy             # proxy | h8 | identity | cut | population | score | reference | compare | diagnose | gate | noop
seeds: [0, 1, 2]        # or seed: 1; with neither, seed 0 only
priority: 50            # lower runs first
after: [verify.bp_b8.s0]
gpu: H100               # optional: only workers whose GPU name contains this take the job
args: {recipe: boostcut, sequences: 10M, heldout: 32, batch: 4, checkpoints: [0]}   # H4 for seed 0 only
expect: {val_loss: [3.7083, 0.01]}   # optional gate; recorded as pass/fail in result.json
```

To add a variant, copy a config, change `variant` and the one option under test, and submit it. A config without
`seeds` (or `seed`) runs seed 0 only, without a warning. Unknown keys and options, bad values, an `expect` on a metric
the task does not produce and duplicate job ids all fail at submission, with one line naming the config. So does a
job id in `args` that the job does not wait for. A diagnostic's `checkpoint: <job id>:<file>` must name a job in its
`after`. A comparison's `a` and `b` job ids must be in its `after_done`, so that it runs even when a check fails. A
gate's `values` job ids must be in either list.

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

**Proxy variants** are one option each, off by default and at most one per run; the default path is unchanged (the
CPU test of the float32 mean still holds bit for bit). Every proxy experiment runs at batch 8 against `e1.bp_b8`.
- **E2, `blur: s`** (`loud/variants.py`). The sites are the raw embedding and each block's mid stream. Each token gets
  r = s × its RMS × √512 × u, u uniform on the sphere: each coordinate's RMS displacement is s times the token's RMS,
  the units of PR #40's `chain_sigma` (0.01). A clean no-grad forward gives the RMS and `train_loss`; two forwards add
  +r and −r, and the mean of their gradients is the gradient of the loss averaged over activation noise, to second
  order. `blurred_loss` is logged too. Arms: s = 0.03, 0.1, 0.3. About 2.3× a baseline update.
- **E4, `step_control`** (`loud/variants.py`). Each update is judged on 4 of the 32 held-out rows, rotating, by
  their loss under the old and the new weights (2 forwards of 4 sequences). `reject`: if it rose, the weights and the
  optimizer state (ANVIL II's rails and energy, Adam's moments and counts) are restored from a snapshot and the batch
  feeds no update; ANVIL II's clock, which drives its momentum schedule, keeps following the loop's update count, as
  the warmdown does. `adapt` also multiplies every rate by 1.1 after an accept and 0.87 after a reject (steady at 59%
  acceptance), on top of the schedule; `adapt_nowd` removes the hand warmdown (`warmdown: 0`). Logged per update:
  `accepted`, `heldout_before`, `heldout_after`, `judge_rows`, `lr_multiplier`; `lr` is the effective rate.

**The identity path** imports PR #26's `train/speedrun/run.py@2604407` unchanged: its model, init, optimizer,
settings and `FastRelay` estimator. There is one estimator per virtual GPU, seeded by its rank as on 8 GPUs. Their
per-sequence gradients are summed where the script all-reduces, then one optimizer step follows. A CPU test checks that
one update over two virtual GPUs equals PR #26's FastRelay at ranks 0 and 1 with `Boost.step`'s sum. It differs from
PR #26 only in the data order: the records' fixed order, `randperm(seed 1)`, for every seed (PR #26 shuffles with
`seed + 1`; the two are identical at seed 0). v2 uses it for E1's single run and verification 3.

**E3, local SGD** (task `population`). The job runs `e1.bp_b8`'s trajectory (the same code and seed) to update 121 of
606 and branches into 4 copies; each update's 8 sequences then go 2 to each copy, so every copy makes all 485
remaining updates at local batch 2 with the baseline's rates and update numbers (and so ANVIL II's clock), each with
its own optimizer state. Rules: `average` (every H = 1, 10, 50 updates: the mean of the weights and of every
optimizer state tensor), `select` (every 50 updates the worst copy takes the best's weights and state, by H3),
`ensemble` (the mean at the end if H3 prefers it to the best copy). `population.jsonl` logs every 10 updates, before
the rule acts at that update, each copy's and each pairwise midpoint's held-out loss and the barrier (midpoint minus
the worse end). Forward and backward cost one run; optimizer compute and state are 4×.

**E8 and E9** change ANVIL II's orthogonalization through one hook (`loud/anvil.py`); a CPU test checks the hooked
optimizer reproduces boost-cut's ANVIL II bit for bit, and with both off the proxy uses the original.
- **E8** (`loud/snr.py`). The diagnostic is H6's `snr_spectrum` (below). The intervention (`snr_scaling`:
  `control`, `snr`, `rate`; in `configs/followups/e8/`): the step becomes U diag(r) Vᵀ, with U and V the singular
  directions of ANVIL II's momentum as it stood before the batch and r from the batch's 8 per-sequence projections
  onto them; ANVIL II's lane gain and renormalization are off in every arm. One SVD per block matrix per update. Its
  second setting is H8's (below).
- **E9** (`fisher`, `loud/fisher.py`; P3, in `configs/followups/e9/`): the momentum is left-preconditioned by
  (F + δI)^(−1/2), F the output-side Fisher from a backward with labels sampled from the model and δ = 1e-3 × its mean
  eigenvalue, before ANVIL II's unchanged orthogonalization. The control is `e1.bp_b8`. Run only if E8 shows that
  per-direction scaling matters with exact gradients; the kernel version follows only if this beats `e1.bp_b8` by 0.02.
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
  its own count up to `rows`: `ngram` is E13's rule (64 rows where a streaming n-gram model of the stream so far and
  the network both put the token's loss at most `budget_theta`, `rows` elsewhere; the table learns every update's
  batch after it). The chain still evaluates the rows past a token's count, so a budget changes the estimate, not
  yet the cost. The full cycle runs the record's `chain_credits` itself. `chain_chunk` compiles one graph per site and
  moves per token (17 for one row count), all in the warmup, which fails if any planned graph is missing. CPU tests
  check that the full cycle is the record's, that the K time windows of one basis average to the full cycle (with the
  prior too), that each token-assigned token gets its window's credit, that a budget reads the token's first rows,
  and that the run's budget is built after the warmup and sees each Cut update's batch after the update.
- **Cost on one H100:** about 1.6 s per Boost update and 16 s per Cut update (8 times the published per-GPU times),
  and about 8 minutes for the end merge (less with `merge_batch`). A 1M-slice run takes about 35 minutes; the full
  10M epoch about 4.6 hours.

**H8** (task `h8`, `loud/h8.py`) trains with PR #40's recipe (its model, init, ANVIL II + Adam, rates, warmdown,
Boost for the first 20% of updates, then Cut, and the optimizer handoff) at its 4 sequences per update, with one
forward of the whole batch. It rests on one fact: with the other tokens' keys and values held at their clean values
and each token's own query live, every path from a site to a token's loss stays inside that token's row. So autograd
on the *cut graph* (`cut_forward`: the fixed network with every stream-derived input to the keys, values and the
value-embedding gate detached) gives, at every site at once, what the chain measures. H8 has neither the record's
cross-token patches nor its top-512 MLP routes, so H8 and backprop bracket the cross-token credit.
- **Modes of the Cut phase.** `cut`: one backward of the cut graph (block i's MLP also gets the own-query path
  through block i + 1, which PR #40 leaves out). `synthesis`: the exact credits at S0–S8 and the top stream, put
  through a probing scheme, then each sublayer's parameter gradient from its own output's credit alone, one
  `autograd.grad` per sublayer, so a transformed credit never leaks between blocks or from an MLP into its attention.
  With the identity scheme this is PR #40's core estimate. Boost is PR #40's: each sublayer against the shared head's
  readout of the stream right after it (`boost: false` starts Cut at the first update).
- **Schemes** (synthesis): `rows` m reads m rows of a site's signed Hadamard basis per token, scaled by 512 / m
  (unbiased through the random column signs); `row_assign: time` (all tokens of update s share rows from
  (s mod K) m, K = 512 / m) or `token` (token t reads rows from (t mod K) m); `prior: identity` measures only the
  residual of the identity-path guess (E7) at the mid sites, as the chain does (S0 keeps the plain estimate). Logs:
  every `compare_every` updates the cosine of the summed weight gradient to the identity scheme's
  (`cosine_to_identity`), and every `rank_every` the stable rank of ANVIL II's orthogonalization input per matrix type
  (`stable_rank`); both are timed apart and left out of the clock. Every Boost run logs the cut gradient's cosine to
  backprop's at the handoff, and every run the effective ranks at 20% and 100% (E15's and E14's baselines).
- **`reforward`** (`Replay`) reruns the network from a moved site with every key and value held, as the chain's
  probes run; E11 and H6's readout and linearity tools use it.
- **Validation** (v2's bar before use): on CPU, synthesis with the identity scheme matches PR #40's Cut estimate
  (cross-token patches off, every unit routed) at cosine 0.996 to 1.0 per matrix, Boost matches PR #40's Boost at
  0.997 or more, and reforward's first-order response matches the credits (cosine above 0.999). `h6.h8_validation`
  repeats the check on a trained checkpoint on the GPU before any H8 experiment.

**H8 experiments**, one option per run, each a fixed rule:
- **E6** (`rows` 128, 64, 32 × `row_assign` time, token) and **E7** (`prior: identity` at m = 64, 128, token rows)
  against `e1.h8_synth`; E6's success is K = 4 within 0.02 of K = 1 and K = 16 within 0.05.
- **E10** `clip_credit`: each token's credit at a site capped at 3× the site's batch median norm. `agree`: each block
  weight-update entry kept only where at least 70% of the tokens back its majority sign: (N + |S|) / 2 ≥ 0.7 × tokens,
  with S = sign(G)ᵀ sign(X) the net votes and N = [G ≠ 0]ᵀ [X ≠ 0] the nonzero contributions (a zero, common after
  the ReLU², backs neither sign), and the share of entries kept logged (`agree_kept`).
- **E11** `runway`: each token's credit at a site weighted by the move along −g/|g| (0.01 to 0.08 of its RMS per
  coordinate, through `reforward`) that lowers its loss most, normalized to mean 1 per site; the chosen moves are
  logged per site (`runway`).
- **E13** `budget_theta`: 64 rows per site (token-assigned within the tier) where both a streaming Witten–Bell n-gram
  model of the training stream (orders 1 to 4, `loud/ngram.py`, causal, held-out rows excluded) and the network put
  the token's loss at most θ, the full cycle elsewhere. Its diagnostic `ngram_tiers` (the low tier's share of tokens
  and of credit energy, at 20% and 100% of `e1.h8_synth.s0`) gates each θ. The speedrun record's difficulty-balanced
  order stratifies whole sequences across batches; it does not give tokens different budgets.
- **E14** (`nextlat`, cut mode; `loud/latent.py`, `loud/ema.py`): NextLat's latent consistency, following its code
  (JaydenTeoh/NextLat at 3770be6) with v2's one-hidden-layer dynamics model: the final-norm latent and the next
  token's raw embedding predict the next latent, smooth-L1 against the EMA teacher's latent or the live one
  (`nextlat_target`), masked where the next token opens a document (`<|endoftext|>`, id 0), from the handoff or the
  first update (`nextlat_onset`). It trains the trunk and wte, never the head. Logged: its gradient's cosine to the
  next-token gradient at the top stream every 50 updates, and the effective ranks of the top latents and site credits
  at 20% and 100%; the plots give the fraction of the gap from `e1.h8_cut` to `e1.bp_b4_pr40` (backprop on H8's
  recipe and batch) closed.
- **E15** (`latent_boost`): during Boost, per-sublayer predictors to the teacher's latent (1 − cosine) beside the
  logit-lens readouts (`plus`) or instead of them (`only`; the head then learns from the final readout alone);
  logged: per-sublayer readout and latent losses, and the cut gradient's cosine to backprop's at the handoff. The EMA
  teacher's decay rises linearly from 0.996 to 0.9995 (ProteinJEPA's end points; it does not publish the curve).
- **E8's H8 setting** (`snr_scaling` in synthesis mode, `configs/followups/e8/h8_*`): at 8 sequences per update on
  128 token rows (E6's best assignment at K = 4, to set before submitting), the Cut phase's ANVIL II becomes
  `ShapedBatchedAnvil` (PR #40's batched ANVIL II with the same hook; a CPU test checks it reproduces the original
  bit for bit), with the momentum's directions from before the batch and r from each sequence's synthesized gradient
  Gᵦᵀ Xᵦ. Boost keeps the record's optimizer: partial
  rows exist only in the Cut phase. `h8_exact_control` (every row, the control arm's optimizer) measures what the
  partial rows cost; the plots give the share SNR scaling recovers.
- **E16** (`arch: ngram`, `loud/arch.py`): track-3 record #5's design, the pinned `train/arch/model.py` (12 × 1,024,
  value embeddings on every block plus per-block projections of hashed bigram, trigram and skip-bigram tables, 614M
  parameters), built with PR #40's init, under H8 cut mode (its n-gram embeddings flow through the cut graph, Boost and
  reforward) and under the proxy with `recipe: pr40` at batch 4. PR #40's optimizer has no rule for the n-gram tables,
  so they join wte's Adam group (300× the base rate), record #5's own rule (`arch.embedding_rates`). The official
  scorer builds the model from the same file, as Noise scores track 3, so only the record's own shape
  (`preset: track3`, the default table rows) can be scored. The fixed model's backprop arm is `e1.bp_b4_pr40`.

**H6, the diagnostic toolkit** (task `diagnose`, `loud/diagnostics.py`), offline on H4 checkpoints, one held-out
sequence unless noted. Exact credits are the cut graph's in float32; probes run through `reforward` in bf16 where the
chain does.
- `credit_dump`: per site, the exact credit, the identity-path guess, their per-token cosine and E7's variance ratio
  ‖g − p‖² / ‖g‖².
- `cos_to_autograd`: an estimator's weight gradient against autograd's, per matrix type, block and group:
  `estimator: identity` (PR #26), `h8` (H8 synthesis; also, per site, the cut graph's credit against the full
  graph's: how much cross-token credit the cut drops) or `cut` (PR #40's Cut at the record's settings).
- `linearity_check`: per site and probe size, how far each token's loss response to +v and −v is from linear.
- `deep_readout_check` (E5): one 512-probe signed Hadamard cycle launched at S1, read at S2 to S8: per site the
  512 × 512 system's median condition number and the median cosine of its solution to the exact credit (plain, and
  with the ridge 1e-3 × mean diag toward the guess). S0 keeps its own cycle in E5, since its probes also move x0.
- `h8_validation`: H8 synthesis against PR #40's Cut estimate, per matrix.
- `synth_vs_real`: H8 synthesis against the real chain's partial-row estimate (PR #40's Cut through `loud/cut.py`'s
  row flags) with the same rows and signs: per site, the per-token cosine of the two and of each to the exact credit,
  at 128 rows by token and by time, with the record's MLP routes and with every unit (to separate route truncation
  from finite-difference and bf16 error).
- `snr_spectrum` (E8): per block matrix, the SNR at batch 8 along each singular direction of the checkpoint's ANVIL
  II momentum, over the 64 training sequences the run reads next, and the share of the momentum's energy below SNR 1,
  for four estimators of the per-sequence gradients: backprop, H8 synthesis with exact credits, and partial rows at
  K = 4 by time and by token.
- `ngram_tiers` (E13): the low tier's share of tokens and of credit energy on the 32 held-out sequences.

**Plots** (`python3 -m loud.plots <exp>`) read only the run files. Every experiment gets the final validation loss per
variant (seeds and mean ± sd, with the reference cell and its −0.02 line where the plan compares against one) and the
train and held-out curves. Tables where the plan names a comparison: each H8 experiment's difference from its
reference cell (E6 and E7 with the cosine to K = 1 and the stable rank), E14's and E15's gap closed, E16's ranking
of the two models under both trainers, E2's train/val gap, E3's basin barriers, E4's acceptance rate and effective
rate, E8's energy below SNR 1 per estimator; the diagnostics E5 (the 7 × 2 table and its decision rule), E7
(the variance ratio per site and checkpoint) and E13 (the tiers and the build condition).

## Review

Every piece and every result goes to two independent reviewers: Codex (run as an unprivileged user on a frozen copy)
and fresh Claude agents. Both get the same rubric and lenses: code, research fidelity, usability and results. Each
finding is triaged in [review/LEDGER.md](review/LEDGER.md). See [review/README.md](review/README.md).

## Differences from the handoff found so far

- The speedrun record (track 2) is now 22.9 s (PR #52), whose difficulty-balanced order stratifies whole sequences
  across batches; it does not give tokens different probe budgets, so E13 stands.
- The pod image ships torch 2.13. Noise pins torch 2.10.0, and the published runs used 2.10. Scoring the record
  checkpoint under 2.13 gives 3.944756 against the published 3.944812 (5.6e-5 lower; test 5.1e-5 lower), so
  verification 1's tolerance is 1e-4. The training verifications show whether training moves more.
- The backprop target is boost-cut's ledger, 3.70835 at seed 0. The handoff's 3.708 is the same run rounded.
- H8's cut mode gives each block's MLP the own-query path through the next block, which PR #40's out-credit leaves
  out. Synthesis mode matches PR #40's core estimate at 0.99 or more on every block matrix in float32
  (`h6.h8_validation_fp32`: the bf16 query gap is the record's rounding; `attn.c_v` and `attn.c_proj` differ slightly
  but systematically, 0.995 to 0.999); the 8 `x0_lambdas` scalars stay at 0.980, and both estimates are far from
  autograd there, cause not yet found.
- In H8, Boost does not help: `e1.h8_cut_noboost` ties `e1.h8_cut` (3.802 against 3.799, paired differences of
  -0.005 to +0.009), and the cut gradient's cosine to backprop is already 0.992 at the handoff. E15 asks whether
  Boost's lens is a crutch for the chain's weak early gradient, which H8 does not have, so its H8 result does not
  carry to the chain.
- E4's `adapt` multiplier rises in the warmdown tail (to 32-52x by the last update): once the hand schedule shrinks
  the steps, the 4-row judge accepts nearly every one, so the rule undoes the end of the warmdown.
- E5's float32 readout tests the method without rounding; the plan's test is from bf16 forward differences as the
  chain's kernels take them (differencing the changes, which the bf16 replay does not), so a float32 pass is
  necessary, not sufficient.
- E11's runway grid (0.01 to 0.08) is too short for this model: 99.98% of tokens still gain at 0.08, so every weight
  is 1 and the arm is the baseline; E10's clip changes the gradient by a cosine of 0.999. Both are nulls by inaction.
- E8's diagnostic takes its directions from the checkpoint's momentum (its rails) rather than from the update that
  also holds the measured sequences' own gradient: directions chosen from the same sequences make their noise look like
  signal (a pure-noise check showed SNR ~100 along them). It measures training sequences, since the held-out set has
  only 32.
- E16's backprop arms, and the backprop end of E14's and E15's gap (`e1.bp_b4_pr40`), use PR #40's recipe at batch 4,
  the same as H8's arms, so the comparison isolates the trainer.
- E8's intervention takes its directions from the momentum before the batch, as its diagnostic does: the update
  itself holds the batch's gradient at weight 0.14 to 0.21, and directions taken from it put a pure-noise batch's mean
  r at 0.85 instead of 0.18 (review round 7). The batch enters the step only through r.
- E6's mechanism ("each writer's gradient has rank at most m on its output side") holds per sequence: the record
  draws fresh signs per sequence, so a batch of B time-assigned sequences spans up to B·m directions (measured 413,
  252, 128 at m = 128, 64, 32 and batch 4). The stable-rank log shows how much of the effect survives.
