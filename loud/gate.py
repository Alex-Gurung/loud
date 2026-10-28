"""Task 'gate': one of the plan's decision rules on finished diagnostics, as a check that other jobs can wait for.
Each named value is one field of a diagnose run's summary file, taken at every site the file lists and reduced to the
median over the sites; `expect` (with {min, max} bounds) passes or fails the gate, and a job with the gate in its
`after` runs only if it passes. No GPU work.

    args:
      values:
        early: [h6.credit_dump_c20.s0, credit_dump.json, median]   # job, its summary file, the field per site
    expect: {early: {min: 0.9}}
"""
import statistics

from . import paths
from .common import read_json
from .runlog import Run


def run_task(spec, run_dir):
    run = Run(run_dir, spec)
    result = dict(complete=True)
    for name, (job, file, field) in spec['args']['values'].items():
        experiment, variant, seed = job.rsplit('.', 2)
        summary = read_json(paths.run_dir(experiment, variant, int(seed[1:])) / file)
        sites = {site: v[field] for site, v in summary.items() if isinstance(v, dict) and field in v}
        if not sites:
            raise ValueError(f'{job}: {file} has no {field!r} at any site')
        result[name], result[f'{name}_sites'] = statistics.median(sites.values()), sites
    return run.finish(result, spec.get('expect'))
