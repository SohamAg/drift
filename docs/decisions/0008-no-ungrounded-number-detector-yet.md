# ADR-0008: No "ungrounded numbers" detector until it can tell fabrication from arithmetic

| | |
|---|---|
| Status | Deferred |
| Decided | 2026-11-19, about 16:45 IST. Nothing was committed: the decision was *not* to ship |
| Recorded | 2026-11-19, the same day |
| Owner | Soham Agarwal |

## Context
Two live runs that day showed the same failure. Agents stated precise
facts that appeared in no tool result. A math agent reported Tokyo's
population as "13,929,286" after its research step was skipped or
misrouted, and the supervisor passed the figure on (ADR-0006). A detector
for it looked simple: flag numbers in the final answer that appear in no
tool result and no user message.

## Evidence
That heuristic was run over every saved result under `results/` that keeps
its message history:

| Result set | Fired / runs checked | What it caught |
|---|---|---|
| model_flow_faults | 2 / 16 | the fabricated populations, and their computed sum |
| supervisor_message_chaos | 6 / 31 | mostly the speed of light, a well-known constant, on baselines |
| web_saved | 1 / 3 | a fabricated population on a swarm baseline (real) |
| tool_faults, react_hijack, midrun_injection, cli_smoke | 0 / 29 | — |

The real fabrications were caught. But a correct sum of two numbers is
also "a number in no tool result", and so is a physical constant. Most
fires that were not fabrications came from one of those two.

## Decision
Do not ship it yet. A detector that fires on correct arithmetic would
train users to ignore it, which costs more than the failures it catches
(ADR-0001: detectors must be trustworthy enough to gate on).

## To revisit, it needs
1. **A guard for computed values:** a number that results from simple
   arithmetic on grounded numbers is grounded.
2. **A guard for well-known constants**, or a rule restricting the check to
   answers the task asked to be researched.
3. **Re-measurement** on the same saved runs, with each fire labelled by
   hand as a fabrication or not.
