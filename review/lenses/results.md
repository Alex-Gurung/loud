# Lens: results

Where things are, under `/mnt/disk/loud/artifacts/`:
- runs: `runs/<experiment>/<variant>/s<seed>/`;
- the queue: `queue/`;
- the results table: `results/table.md`. Rebuild your own copy with `python3 collect.py --runs --out <scratch>`.

The question: are the numbers right, and do they support what is claimed about them?

1. **Trace every number in the table** to a `result.json`, and recompute each mean and standard deviation.
2. **Check each run's provenance.** Each run:
   - is complete and passed its own check;
   - used the intended code digest, config and GPU type.

   Cells that are compared share seeds, code, held-out setting and GPU type.
3. **The logs agree with each other:**
   - `metrics.jsonl` (step count, final train loss, time) matches `result.json`;
   - held-out scoring appears in `heldout.jsonl`;
   - nothing read `val.pt` during training (`eval_every` is off in experiment cells).
4. **Against the plan:**
   - the verification gates: 3.94481; 3.708 ± 0.02; about 4.18;
   - each experiment's success and kill criteria, computed from the table;
   - a difference under 0.02 with fewer than three seeds is not a result.
5. **Wall-clock** is compared only within one GPU type. Say what the timer includes: not compile or scoring.
6. **Anomalies worth a rerun:** loss spikes, NaNs, a seed far from the others.
