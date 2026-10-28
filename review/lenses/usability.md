# Lens: ease of use

The reader is a researcher new to the repo. Using only `README.md` and `--help`, they must run the verification,
add an experiment variant and read the results.

1. **Read the README top to bottom.**
   - Is every command correct as written? Does every claim match the code?
   - Run every command that works without a GPU: `loud.status`, `loud.submit --dry-run` against a scratch queue,
     `collect.py --out <scratch>`, the tests.
2. **Adding a variant.**
   - How many files must change? Is the config format obvious?
   - Do bad inputs fail fast with a clear message? Try a misspelled option, a missing seed, an unknown task.
3. **Failures a user will meet:**
   - a run that crashes;
   - a pod that dies;
   - a full disk;
   - a failed check that blocks the jobs after it.

   Does `loud.status` show each one, and is the fix obvious?
4. **Output.**
   - Are run directories, `result.json` and the results table easy to read?
   - Is each name the same in the config, the queue, the run directory and the table?
5. **Friction:**
   - steps that could go;
   - defaults that surprise;
   - jargon used before it is defined.
