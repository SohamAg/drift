# `ungrounded_number`

> The final answer says Tokyo has 13,929,286 people. No tool returned that
> number, the user didn't give it, and no arithmetic over what the run did
> see produces it. It reads as researched and was not.

## What fires

Once per run, when the run had at least one successful tool result (so
grounding was available) and either:

1. **The final answer states an unsupported specific number.** The final
   answer is the last AI message with text and no tool calls. A *specific*
   number has at least 4 significant digits. It is unsupported when it
   matches nothing in a human, system or tool message, or in the initial
   state, at the precision the answer states it. "13.93 million" matches
   13,929,286. "13.95 million" does not.
2. **A tool call's arguments carry one.** An agent called a tool with a
   specific number nothing earlier in the run supplied, for example
   `multiply(a=13929286, b=0.5)` after a search that returned nothing. This
   is usually where a fabricated figure first enters. That call's result is
   then **not** treated as grounding. Otherwise the fabricated input is
   "explained" backwards: 13,929,286 = 6,964,643 / 0.5.

**Deliberately NOT flagged:**
- **Simple computations over grounded numbers:** the sum, difference,
  product, ratio or percentage of any two, the sum of any three, and the
  sum of all of them.
- **Well-known constants**, compared scale-free so 299,792,458 m/s and
  299,792 km/s both match: the speed of light, pi, e, Avogadro, Planck, the
  gravitational constant, absolute zero, seconds per year, powers of two,
  and similar.
- **Numbers under 4 significant digits** (7, 56, 2.1 million, 14 million),
  **bare years** (1000–2100), and digits inside identifiers, dates and
  versions (`2026-11-21`, `v1.2.3`, `gpt-4o`, `#A1234`).
- **Runs with no successful tool result.** An agent with no tools answering
  from its own knowledge is a different question.
- Tool results drift itself planted (`drift_call_*`) do not count as
  "grounding was available".

## What this typically means

- **The retrieval step never ran.** It was skipped, misrouted, or crashed
  and the error was swallowed. A downstream agent filled the gap from its
  weights.
- **The tool returned nothing useful, and the agent didn't say so.** An
  empty search or a mock result was followed by confident figures.
- **A sub-agent answered from memory instead of calling its tool.**

## Concrete example

Live, drift `handoff_misroute[transfer_to_research_expert]` on the 2-agent
langgraph-supervisor system, gpt-4o-mini
(`examples/adapters/run_drift_on_model_flow_faults.py --part handoff`):

```
human        Look up the population of Tokyo and the population of Paris, then compute their sum.
supervisor   → transfer_to_research_expert    (misrouted by drift to math_expert)
math_expert  → _add(a=13929286, b=2140526)    ← nothing supplied these
tool         16069812.0
supervisor   The population of Tokyo is approximately 13,929,286, and the population
             of Paris is about 2,140,526. Their sum is 16,069,812.
```

```
ungrounded_number: math_expert called _add with 13929286, 2140526 (and 1 more call),
which nothing in the run supplied; supervisor's final answer states 13,929,286,
2,140,526, 16,069,812, which no tool result, user message or prompt contained except
via those calls and no simple computation over them gives. Presented as researched;
the run never grounded it
```

`empty_handoff` names the broken handoff. This detector names what the
user ends up holding.

## Measured

Over the 110 saved runs under `results/` that have tool results
(2026-11-21), it fired **3 times**, and all three are real fabrications:
- `skip_node[research_expert]`, supervisor;
- `handoff_misroute[transfer_to_research_expert]`, supervisor;
- a swarm baseline whose mock search returned nothing.

The bare heuristic it replaces (see `docs/decisions/0008`) also fired on
the speed of light and on correct sums. Across those runs' final answers:

| Number | Count | Cleared by |
|---|---|---|
| Under 4 significant digits | 155 | the size rule |
| Speed of light | 19 | the constants guard |
| In a message or tool result | 3 | grounding |
| A bare year | 3 | the year rule |
| Correct computations | 2 | the computation guard |

Of the 2 the computation guard cleared, one was a fabricated figure
recovered backwards from its own tool result. Tool-argument tracking now
catches that case.

## How to fix — architecture, not runtime

- **Make empty retrieval explicit.** A search tool that finds nothing
  returns an error or a "no data" result the agent must surface, not an
  empty string.
- **Require citations for figures.** Ask agents to attach the tool call
  each number came from. A guard node, or a `drift.yml` rule, can reject
  answers whose numbers have no source.
- **Keep retrieval and computation in separate agents with a checked
  handoff**, so a math agent never receives a request it can only satisfy
  by inventing inputs (see `empty_handoff`).

## Sources

- MAST (arXiv:2503.13657): 3.3 Incorrect verification. Unsupported claims
  pass through unchecked.
- drift live runs, 2026-11-19 (`docs/checkpoint_chaos.md`).
- ADR-0008 (deferred), ADR-0010 (accepted).
