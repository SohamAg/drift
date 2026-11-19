# ADR-0007: A fault that never landed is not a result

| | |
|---|---|
| Status | Accepted |
| Decided | 2026-11-19, 16:34–16:41 IST — `0e8e753`, `265dbc1`, `6f881fe`, `9e61e04`, `9b2a71c`, `38c4105`, `6fa4a0f` |
| Recorded | 2026-11-19, the same day |
| Owner | Soham Agarwal |

## Context
Since ADR-0002 and ADR-0004, some scheduled faults can fail to happen:
- a tool fault whose tool was never called on that run;
- a model or control-flow fault the baseline gave no place to land.

Each such perturbation was marked `fault_fired=False`, but every count
still treated it as a run that **changed nothing**, which reads as the
graph coping with the fault. It also sat in the denominator of every rate.
In a test case, three unapplied faults turned a 50% crash rate into 20%
and passed a `max_crash_rate=0.3` gate. Their runtime of about 0 seconds
also pulled the regression tracker's mean runtime down, so turning on a
fault layer could read as a speed-up.

## Decision
A fault that did not land is reported, but it is **not a test result**:
- **Counts:** not unchanged, not crashed, not diverged; tracked as
  `n_not_applied`.
- **Rates and regression snapshots:** left out of both.
- **Gate:** adds a note when any are left out. A run in which nothing
  landed fails as `no_perturbations` instead of passing.
- **JUnit:** written as `<skipped>`, not a passing test. The Markdown
  summary says "not applied".
- **Graph-run count (`n_graph_runs`):** keeps a tool fault that did not
  fire, because that graph did run. Only model and control-flow faults
  that never ran are dropped from it.

## Alternatives considered
- **Leave counts alone and document it.** Rejected. A gate that passes
  because tests did not run is the failure drift exists to prevent.
- **Drop unapplied faults from results entirely.** Rejected. Users need
  to see that a fault they declared had nowhere to land, and why.

## Consequences
- Rates can rise for existing users when model or control-flow chaos is
  on. That is the correction, and the CHANGELOG says so.
- The first version of the graph-run fix also dropped tool faults that
  did not fire. `6fa4a0f` corrected that within minutes. The commit
  history shows both.
