# Design decision records

One file per design decision that shaped drift: what was decided, what
else was considered, why, and what it costs. Format: lightweight
Architecture Decision Records (ADRs).

## How these are dated

Each record has two dates, and they are kept apart on purpose:

- **Decided**: when the decision took effect in the code, with the commit
  that carries it. Check it with `git show <hash>`.
- **Recorded**: when this file was written. Its own git commit is the
  timestamp. Records 0001–0009 were all written on 2026-11-19, some of them
  after the fact, for decisions made between June and November 2026.

A record is never edited to change what was decided. When a decision is
replaced, a new record says so and the old one's status becomes
"Superseded by ADR-NNNN". Small clarifications are allowed, and git keeps
them visible.

Owner of every decision: Soham Agarwal (Sutra Labs). Much of the
implementation was done with Claude Code, an AI coding agent, in sessions
the owner directed. The commits show which.

| # | Decision | Status | Decided |
|---|---|---|---|
| [0001](0001-deterministic-detectors-first.md) | Deterministic detectors are the primary failure signal; the LLM judge is optional | Accepted | 2026-06-25 |
| [0002](0002-tool-faults-via-toolnode-interceptors.md) | Tool faults are injected through ToolNode's own interceptor, not by patching tools | Accepted | 2026-11-11 |
| [0003](0003-declarative-rule-language.md) | User rules are a declarative drift.yml language, not user Python | Accepted | 2026-11-11 |
| [0004](0004-checkpoint-faults-opt-in.md) | Model and control-flow faults are written onto baseline checkpoints, and are opt-in | Accepted | 2026-11-15 |
| [0005](0005-ci-gate-compares-by-event-name.md) | The CI gate compares runs by event name; a missing baseline skips rather than fails | Accepted | 2026-11-15, extended 2026-11-19 |
| [0006](0006-handoff-misroute-at-tool-layer.md) | A supervisor's misrouted handoff is injected at the tool layer | Accepted | 2026-11-19 |
| [0007](0007-unapplied-faults-are-not-results.md) | A fault that never landed is not a result: excluded from every count and rate | Accepted | 2026-11-19 |
| [0008](0008-no-ungrounded-number-detector-yet.md) | No "ungrounded numbers" detector until it can tell fabrication from arithmetic | Deferred | 2026-11-19 |
| [0009](0009-distribution-and-license.md) | Ship the dashboard inside the package; install from GitHub until a PyPI name and license are chosen | Accepted in part; license open | 2026-11-19 |

New record: copy the header block of any record, take the next number, and
commit it on the day the decision is made.
