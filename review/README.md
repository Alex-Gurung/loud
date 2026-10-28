# Review

Every piece of the harness, and every result, is reviewed by two independent reviewers before it is used: Codex (an
OpenAI model) and a fresh Claude agent. Neither has the author's context, and the two model families catch
different mistakes. The author triages every finding in [LEDGER.md](LEDGER.md).

## Lenses

| Lens | Question |
| --- | --- |
| [code](lenses/code.md) | Is it correct, clean, free of slop, efficient on the GPU path? |
| [research](lenses/research.md) | Does it implement the plan and match the reference scripts, so results answer the plan's questions? |
| [usability](lenses/usability.md) | Can a new researcher run, extend and read it from the README alone? |
| [results](lenses/results.md) | Are the numbers right, and do they support the claims? |

Every reviewer gets [RUBRIC.md](RUBRIC.md) (ground rules, severities, report format) plus one lens.

## When

1. **Before an experiment's runs are queued:** code and research lenses on the code they use.
2. **After each piece of infrastructure** (H1 to H7) and after any large change: all three code-facing lenses.
3. **When an experiment's runs finish:** the results lens, before any number from them is reported.

**Gate:** nothing is queued while a blocker or major finding against its code is open, and no result is reported
before its results review.

## How

Run Codex as root on the CPU pod. It reviews a frozen copy of the working tree and writes
`reports/<stamp>-<lens>-codex.md`:

```bash
review/codex.sh code
review/codex.sh research "Focus on loud/identity.py against PR #26."
```

Codex's own sandbox cannot start in this pod: bubblewrap needs mount privileges the container lacks. So the script
runs Codex as the unprivileged user `loud-auditor`, which can read everything and write only its work directory,
`/var/tmp/loud-review/<stamp>-<lens>/`. The prompt, the log and the frozen source copy stay there. The script copies
your Codex login to that user and syncs whichever copy refreshed last back to the other.

Run the Claude reviewer from Claude Code with the project agent [`loud-reviewer`](../.claude/agents/loud-reviewer.md).
Give it the lens, a frozen copy (the one Codex reviewed works), a scratch directory and the report path
`reports/<stamp>-<lens>-claude.md`. Give it focus lines the same way as Codex, for example for a later round.

## Triage

Each finding gets one row in [LEDGER.md](LEDGER.md), after the author checks it:

- **fixed**: what changed;
- **rejected**: the evidence that the finding is wrong;
- **deferred**: why, and what it waits on.

A fix to a blocker or major finding goes back to a reviewer with a focus line naming the finding.
