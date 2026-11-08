# `retry_amplification`

> Under one injected fault the graph did two, three, four times the work
> the baseline needed for the same task — and quite possibly still gave
> the right answer. The outcome buckets call that *unchanged*. The bill
> doesn't.

## What fires

The detector compares the **work the graph itself did** in a perturbed run
against the baseline run of the same task:

- **tool calls** — AI tool calls written at step ≥ 1,
- **LLM turns** — AI messages written at step ≥ 1,

both excluding anything that was already in the initial state (the user's
conversation history, and the call/result pair drift's own
`inject_tool_error_result` pattern plants). The baseline's numbers are read
from its final `messages` channel the same way.

It fires once when **either** metric is at least `min_ratio` × the
baseline (default 2.0) **and** at least `min_extra` higher in absolute
terms (default 3). A baseline that made no tool calls at all counts as
amplified when the perturbed run makes three or more.

**Deliberately NOT flagged:**
- A 1 → 2 bump. Doubling from one call is one retry, which is healthy.
- 4 → 7 calls. Large in absolute terms but 1.75× — within the slack a
  non-deterministic graph needs.
- The baseline run itself (no comparator), and graphs with no `messages`
  channel (nothing to count — use `duration_s` and the step count from the
  trace for those).
- Work drift injected. `drift_call_*` ids are excluded on both sides.

## What this typically means

Nothing in the loop bounds "try again". The model saw an error-shaped
tool result (or a garbled instruction, or a conversation with a hole in
it) and decided — unboundedly — to retry, re-plan, or re-delegate:

- **Retry without a counter.** Each failed call produces a fresh "let me
  try again". `tool_call_loop` catches the degenerate case (identical call,
  identical result); this detector catches the varied one — different
  arguments each time, a different specialist each time.
- **Supervisor re-delegation storms.** The supervisor hands the task to
  the research agent, gets nothing useful back, hands it to the math
  agent, gets nothing, hands it back to research… Every hop is two LLM
  turns and a handoff tool call.
- **Planner inflation.** A planner node that rewrites the plan after
  every setback, adding steps rather than cutting losses.

AgentChaos (ASE 2026) measured exactly this across six agent frameworks:
LLM calls per task rose to as much as **4.24× baseline** under injected
faults, with AutoGen and CrewAI the worst affected — and in most of those
runs the task still "succeeded".

## Sources

- **AgentChaos** — cost amplification under injected faults; the
  *Silent* outcome class is its sibling finding.
  ([arXiv:2608.06790](https://arxiv.org/abs/2608.06790))
- **MAST 1.3 — Step Repetition.**
  ([arXiv:2503.13657](https://arxiv.org/abs/2503.13657))
- **Think Locally, Explain Globally** — tool-repetition rate as a
  controller-failure signal.
  ([arXiv:2601.17915](https://arxiv.org/abs/2601.17915))

## Concrete example

Baseline: `get_weather(Paris)` → `18°C` → "It is 18°C in Paris." (1 call,
2 turns.)

Perturbed (`inject_tool_error_result`):

```
step  node    message
────  ──────  ─────────────────────────────────────────────
1     agent   ai   tool_calls=[get_weather({"city": "Paris"})]
2     tools   tool "Error: 503"
3     agent   ai   tool_calls=[get_weather({"city": "Paris, FR"})]
4     tools   tool "Error: 503"
5     agent   ai   tool_calls=[get_weather({"city": "Paris, France"})]
6     tools   tool "Error: 503"
7     agent   ai   tool_calls=[get_forecast({"city": "Paris"})]
8     tools   tool "Error: 503"
9     agent   ai   tool_calls=[get_weather({"city": "Paris"})]
10    tools   tool "Error: 503"
11    agent   ai   "It is probably 18°C in Paris."
```

Finding:

```
{
  "failure_type": "retry_amplification",
  "agents_involved": ["agent"],
  "timestep": 11,
  "summary": "under this perturbation the graph made 5 tool calls vs 1 at
              baseline (5.0×) and 6 LLM turns vs 2 at baseline (3.0×) — a
              retry / re-planning storm; the final answer may still be
              right, the cost is not"
}
```

Note `tool_call_loop` does **not** fire here — the arguments differ each
time. That's the gap this detector covers.

## How to fix — architecture, not runtime

### 1. Budget the loop in the graph

A per-run counter in state and a conditional edge that routes to a
fallback node once it's spent:

```python
def after_tools(state):
    attempts = state.get("tool_attempts", 0) + 1
    if attempts > 3:
        return "give_up"
    return "model"
```

### 2. Make failures terminal, not retryable, when they should be

A `503` is worth one retry with backoff. A `404` or a schema error is
not. Encode that in the tool result ("do not retry") or in the routing,
so the model never gets to decide.

### 3. Cap re-delegation at the supervisor

Track which specialist has already been asked and refuse to hand the same
task to the same agent twice without new information.

### What NOT to do

- **Don't just raise `recursion_limit`.** It makes the storm longer.
- **Don't treat an unchanged final answer as "the graph coped".** It
  coped at 4× the cost; at scale that is the outage.
