# ADR-0006: A supervisor's misrouted handoff is injected at the tool layer

| | |
|---|---|
| Status | Accepted |
| Decided | 2026-11-19 16:22 IST — `b554e80` |
| Recorded | 2026-11-19, the same day |
| Owner | Soham Agarwal |

## Context
"The supervisor sent the work to the wrong specialist" is a central
multi-agent failure, and drift could not produce it. The natural place for
it is the model layer (ADR-0004): rewrite the supervisor's
`transfer_to_research_expert` call into `transfer_to_math_expert`. A probe
showed it never lands. In langgraph-supervisor, the supervisor is itself a
subgraph, and its call is made and executed inside that subgraph.
Model-layer faults only see the top-level graph's checkpoints, so the
pending call never appears there. drift reported the fault as "not
applied".

## Decision
Make it a **tool-layer** fault, `handoff_misroute`. The supervisor's
handoff tools live on a ToolNode inside its subgraph (`supervisor/tools`),
which tool chaos can already intercept (ADR-0002). When the targeted
handoff runs, drift re-points the request at another handoff tool on the
same node with `ToolCallRequest.override`. That tool really runs and
routes to the other agent. Its own message says where control went.

The fault is offered only for handoff tools: those carrying LangGraph's
`__handoff_destination` metadata, or named `transfer_to_*`, and only when
the same node has another destination. Declaring it on any other tool is
a clear error. It is part of the default automatic plan.

## Alternatives considered
- **Edit the subgraph's past checkpoints.** Rejected. When the parent is
  forked and resumed, LangGraph re-runs the subgraph under a new task id,
  so the edit is lost.
- **Rewrite the handoff's message to hide the misroute.** Rejected. It
  tests a different failure (being lied to) and is harder to read in a
  trace.

## Consequences
- Live on gpt-4o-mini, 2 of 2 runs: misrouted from research to math, the
  supervisor passed on math_expert's made-up population figures, even
  though the handoff message said "transferred to math_expert". The
  `empty_handoff` detector fired on both (`docs/checkpoint_chaos.md`).
- Works for any framework that hands off through a ToolNode tool. It does
  not cover routers written as conditional edges.
