# ADR-0005: The CI gate compares runs by event name; a missing baseline skips rather than fails

| | |
|---|---|
| Status | Accepted |
| Decided | 2026-11-15 22:55 IST — `62f76a7` (`--against`, `new_finding`, `new_crash`, `regression`). Extended 2026-11-19 15:39 IST — `1d7fdd8` (`new_event_crash`) |
| Recorded | 2026-11-19 |
| Owner | Soham Agarwal |

## Context
A pull request should fail on what it broke, not on what `main` already
had. That requires comparing this run with an earlier one. The open
questions were how to match a perturbation in one run to "the same" one
in another, and what to do when there is no earlier run.

## Decision
- Perturbations are matched by **event name**, for example
  `AutoChaos.flip_bool[approved]`. Names are stable for a given graph and
  seed, so the docs tell users to keep `--seed` fixed.
- `new_crash` fails when a perturbation that ran clean before now crashes,
  or when the baseline newly crashes.
- On 2026-11-19, **`new_event_crash`** was added: a crash on a perturbation
  that did not exist in the earlier run. That happens when a user turns on
  a new fault layer, adds a tool or changes the schema. `new_crash` cannot
  judge these, and when they occur without the new rule, the gate now adds
  a note naming them.
- With no previous result (the first run on a branch, or a missing file),
  these rules are **skipped with a note, not failed**. An unreadable file
  is an error.

## Alternatives considered
- **Match by position in the plan.** Rejected. Any change to sampling
  shifts every position.
- **Fold "new event crashes" into `new_crash`.** Rejected. It would make
  turning on a new fault layer fail every pull request until `main` caught
  up. As a separate rule it is opt-in.
- **Fail when there is no previous result.** Rejected. Every new branch
  would start red.

## Consequences
- Regression gating is deterministic, given a fixed seed.
- Renaming a pattern breaks matching for that pattern for one run.
