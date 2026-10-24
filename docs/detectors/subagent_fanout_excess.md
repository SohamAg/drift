# `subagent_fanout_excess`

> An orchestrator spawned far more subagents than the task plausibly
> required. Cost regressions in multi-agent systems are invisible
> without instrumentation — each subagent looks like normal work, but
> the fanout decision itself is the leverage point.

## What fires

The detector identifies a likely orchestrator/parent agent, then checks
the set of distinct subagents (all other agents in the trace) against
two independent rules. Either rule firing is sufficient:

1. **Hard count rule**: the number of distinct subagents meets or
   exceeds `max_subagents` (default 8). This short-circuits — if it
   fires, the ratio rule below is skipped to avoid double-reporting the
   same fanout.
2. **Ratio rule**: when the fanout is at least `min_subagents_for_ratio`
   (default 4), the detector counts *measurable outputs* landed in the
   final state (non-empty string fields, elements in list/dict
   containers, truthy scalars) and fires if
   `n_subagents / n_outputs > max_subagents_per_output` (default 3.0).
   A fanout of 9 subagents that produced only 3 distinct outputs trips
   this rule even though 9 is below the hard-count default.

**How the orchestrator is identified**: first, any agent whose declared
role (`roles_by_agent`) is one of `orchestrator`/`supervisor`/`manager`/
`planner`/`coordinator`/`router`, or whose name matches that pattern, is
treated as the parent. If none matches, the first distinct agent in the
trace is treated as the parent, but only when there are at least
`min_subagents_for_ratio - 1` other distinct agents following it —
otherwise no parent is assigned and the detector still evaluates the
hard-count rule against total distinct agents.

**Deliberately NOT flagged:**
- Fanouts below both thresholds — a 3-way map-reduce is normal and
  explicitly called out in the module's own threshold rationale
  (Anthropic's postmortem cites 50 subagents as the incident; this
  detector's default of 8 sits well below that and well above a typical
  2–5-way fan-out).
- A large subagent count that produced a proportional number of
  outputs — the ratio rule only fires when the fanout clearly
  outstripped what landed in state.

## What this typically means

The orchestrator's dispatch decision, not any individual subagent's
behavior, is where the cost leverage is. Concrete manifestations:

- **Over-decomposition**: a task that could be handled by 2–3 focused
  subagents gets split into many narrow slices, each paying full LLM
  call overhead for a sliver of work.
- **Retry-as-fanout**: a subagent fails or produces a low-confidence
  result and the orchestrator spawns more subagents to "try again" in
  parallel rather than retrying the one that failed — multiplying cost
  without multiplying signal.
- **Exploratory fanout with no convergence step**: many subagents are
  spawned to gather information, but there's no reduce step that
  consolidates their outputs — most of the fanned-out work never makes
  it into the final state, which is exactly what the ratio rule catches.

## Sources

- **Anthropic engineering blog — "How we built our multi-agent research
  system"** — the named incident: a system that spawned 50 subagents
  for queries that should have used a handful, and the ~15x token-cost
  threshold the team used to judge multi-agent viability against
  single-agent chat.
  ([anthropic.com/engineering/multi-agent-research-system](https://www.anthropic.com/engineering/multi-agent-research-system))
- MAST does not enumerate excess fanout as its own mode; this detector
  is tagged under MAST 2.3 (Task derailment) as the closest fit — fanout
  without proportional output is a form of the system drifting from the
  task's actual scope.

## Concrete example

**Trace:**

```
step  node                       update
────  ─────────────────────────  ─────────────────────────────
1     orchestrator               {"plan": "research 3 topics"}
2     worker_a                   {"finding": "..."}
3     worker_b                   {}
4     worker_c                   {}
5     worker_d                   {}
6     worker_e                   {}
7     worker_f                   {}
8     worker_g                   {}
9     worker_h                   {}
10    worker_i                   {"finding": "..."}
```

Finding:

```
{
  "failure_type": "subagent_fanout_excess",
  "agents_involved": ["orchestrator", "worker_a", "worker_b", ...],
  "timestep": 10,
  "summary": "orchestrator 'orchestrator' spawned 9 distinct subagents
              (worker_a, worker_b, worker_c, worker_d, worker_e,
              worker_f...) — threshold 8"
}
```

Nine subagents ran for a task that only two (`worker_a`, `worker_i`)
produced a measurable finding for — the hard-count rule already fires
here at 9 ≥ 8; had the count been 7, the ratio rule (9 workers would
need to become 7, still ≥ 4-worker minimum) would have caught the
7-workers-for-2-outputs ratio of 3.5 > 3.0 instead.

## How to fix — architecture, not runtime

### 1. Cap fanout width at the dispatch node

```python
def orchestrator(state):
    subtasks = _plan(state["query"])
    return {"dispatch_queue": subtasks[:MAX_FANOUT]}
```

A hard cap at the point of dispatch is cheaper and more legible than
catching the symptom after the fact.

### 2. Add a convergence/reduce step

If subagents are being spawned exploratively, make sure there's a node
downstream that consolidates their output into the final state — so
work that doesn't survive isn't silently wasted AND isn't invisible to
this detector's output-ratio check either.

### 3. Prefer fewer, broader subagents over many narrow ones

Anthropic's own postmortem conclusion: subagents should each own a
meaningfully-sized chunk of the task, not a sliver. If your planner
node is producing subtask lists in the double digits for routine
queries, the planning prompt — not the fanout mechanism — is usually
the thing to fix.

### What NOT to do

- **Don't add a "fanout governor" agent** that decides at runtime
  whether the fanout "looks reasonable." It has no more insight than
  the planner that produced the fanout in the first place.
- **Don't silence this by raising `max_subagents`** without checking
  whether the fanned-out work is actually landing in the final output —
  check the ratio first.

## False positives and known limitations

- **Blind to tool-call-hidden subagent delegation.** This detector
  (like the rest of the library) only sees LangGraph super-steps in the
  adapter trace. A framework that dispatches subagents via tool calls
  rather than graph nodes (e.g. `deepagents`'s `task()` tool) produces
  no visible fanout in the trace drift captures, even if the real
  subagent count is large. This was confirmed empirically — drift's
  detector library surfaced zero findings against a `deepagents`-based
  target specifically because its subagent delegation never shows up as
  graph nodes. This is a scope limitation, not a tuning problem: no
  threshold adjustment makes this detector see fanout it structurally
  cannot observe.
- **Orchestrator identification is heuristic.** Name/role-pattern
  matching can mis-assign the parent in graphs with unconventional
  naming, which skews which agents end up counted as "subagents" and
  which get excluded from the count.
- **Output counting is approximate by design** — it's a cheap proxy
  (non-empty strings, container lengths, truthy scalars), not a
  semantic judgment of whether each output was actually useful. A
  fanout that produced many short, low-value outputs still looks
  "proportional" to this detector.
- **Text-only variant (`detect_from_text`) only applies the hard-count
  rule** — no ratio check, since raw transcripts don't carry structured
  final state. It's also weaker at name extraction since it depends on
  matching a handful of common spawn-phrase patterns.

## Related detectors

- **`infinite_handoff`** — the opposite cost pathology: too much serial
  back-and-forth between a fixed pair of agents, instead of too much
  parallel spawning.
- **Judge-based coordination findings** — can sometimes infer excess
  delegation from narrative text even when the structural detector's
  tool-call blind spot applies, though with lower precision.

## Configuration

Optional kwargs to `detect()`:

| Arg | Default | Purpose |
|---|---|---|
| `max_subagents` | `8` | Hard distinct-subagent-count threshold |
| `max_subagents_per_output` | `3.0` | Subagents-per-measurable-output ratio threshold |
| `min_subagents_for_ratio` | `4` | Minimum fanout size before the ratio rule applies |

## Cost

Zero LLM calls. Runs in <1ms on typical traces. Safe to enable in CI.
