# `empty_handoff`

> The supervisor handed off to a specialist, the handoff tool said
> "Successfully transferred", and the specialist never said a word. The
> next turn is someone else's, and nobody noticed.

## What fires

For every handoff call (`transfer_to_<agent>`, `handoff_to_<agent>`,
`delegate_to_<agent>`, `route_to_<agent>`) whose tool result is not an
error, the detector looks at the **next AI message** after that result. It
fires when that message is not from `<agent>`, matched against the
message's `name` and against the root and leaf of the node that wrote it.

**Deliberately NOT flagged:**
- `transfer_back_to_*`, a specialist returning control. That is the end of
  a handoff, not a new one.
- A handoff that **failed** (error result). The caller was told.
- A handoff that is the **last thing in the run**: no later AI message, so
  nothing "carried on". A run that stops right after handing off is
  `premature_termination` territory.
- Handoff calls drift itself planted (`drift_call_*`).

## What this typically means

- **The specialist crashed and something swallowed it.** An error handler,
  a retry wrapper or a fallback edge returned control without the
  specialist's output.
- **A conditional edge sent control back early.** The graph's wiring
  skipped the node the handoff was supposed to reach.
- **The subgraph's output never reached the parent.** An output mode or a
  state key mismatch dropped the specialist's messages on the way up.

In every case the caller assumed the handoff contract ("I asked X, X
answered") held, and never checked.

## Concrete example

Live, drift `skip_node[research_expert]` on the 2-agent
langgraph-supervisor system, gpt-4o-mini, 2 of 2 rollouts
(`examples/adapters/run_drift_on_model_flow_faults.py --part flow`):

```
human       Look up the population of Tokyo and the population of Paris, then compute their sum.
supervisor  → transfer_to_research_expert
tool        Successfully transferred to research_expert
supervisor  → transfer_to_math_expert                       ← research_expert never spoke
tool        Successfully transferred to math_expert
math_expert The population of Tokyo is approximately 13,929,286, and the population of Paris…
supervisor  The total population of Tokyo and Paris is approximately 16,069,812.
```

The figures came from math_expert's own knowledge. The answer reads as
researched and is not. Before this detector, only `novel_tool_call` fired
(math tools the baseline never called), which names a symptom, not the
cause. Now:

```
empty_handoff: supervisor handed off to 'research_expert' and the handoff succeeded,
but 'research_expert' produced no message: the next turn is 'supervisor', which handed
off again — nothing checked that 'research_expert' answered
```

Scanned against the 92 saved baseline and perturbed runs under `results/`
(supervisor, deepagents, adversarial and react systems), it fired exactly
once: on this run.

## How to fix — architecture, not runtime

- **Make the handoff return value carry the answer.** Route the
  specialist's result back through the handoff, so "transferred" and
  "answered" are the same event. A handoff that comes back without content
  is then an error the caller sees.
- **Verify before routing onward.** The supervisor prompt or a guard node
  checks that the last specialist message exists and addresses the
  request before it delegates the next step. A `drift.yml` rule can
  enforce the shape:
  ```yaml
  rules:
    - id: research-before-math
      tool_order: {before: transfer_back_to_supervisor, after: transfer_to_math_expert}
  ```
- **Surface specialist failures.** An error handler that returns control
  must also return the error as a message.

## Sources

- MAST (arXiv:2503.13657): 2.4 Information withholding; 3.2 No or
  incomplete verification.
- LangGraph supervisor / swarm handoff tools (`transfer_to_<agent>`).
