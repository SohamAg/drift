# ADR-0003: User rules are a declarative drift.yml language, not user Python

| | |
|---|---|
| Status | Accepted |
| Decided | 2026-11-11 15:00 IST — `f36a760` |
| Recorded | 2026-11-19 (after the fact) |
| Owner | Soham Agarwal |

## Context
Teams know rules no generic detector can: "the researcher never sends
email", "approve before refund", "at most five searches". Comparable tools
let users express these by writing Python callbacks. That is flexible, but
the rules become code a reviewer has to read and that can fail in its own
ways. Nobody outside engineering can check them.

## Decision
Rules live in a `drift.yml` file with a small, closed set of
deterministic rule kinds: `forbid_tool`, `tool_order`, `max_tool_calls`,
`max_steps`, state expressions and others (`docs/guidelines.md`). Every
rule is checked on every run, baseline included. A violation becomes a
finding `guideline:<id>` that the gate can fail on (`--fail-on guideline`).
Python remains possible through `@drift.guideline` for the rare rule the
language cannot express. Plain-English checks go to the optional judge
(ADR-0001).

## Alternatives considered
- **Python callbacks only.** Rejected as the main path, for the reasons
  above. Kept as an escape hatch.
- **Natural-language rules checked by an LLM.** Rejected as the main path:
  they are non-deterministic (ADR-0001). Kept as `judge:` entries.

## Consequences
- A rules file is readable by someone who does not write Python, can be
  reviewed in a pull request, and gives the same verdict every time.
- The language has to grow when a common rule cannot be expressed. Each
  new kind is a change to drift.
- The same file declares chaos (`chaos:`), so one reviewed file says both
  what to break and what must still hold.
