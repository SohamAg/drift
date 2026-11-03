# `tool_call_loop`

> An agent issued the same tool call, with the same arguments, three or
> more times — and got the same answer back every time. The conversation
> didn't change between calls, so the agent's situation didn't either.
> It looks like work; it is a stuck loop burning tool and LLM budget.

## What fires

The detector reads the chat messages the graph's nodes write (the
`messages` channel, in whatever dialect the graph uses) and pairs every
AI tool call with its tool result — by `tool_call_id` when the result
names an outstanding call, positionally otherwise. It then groups calls
by **tool name + canonical JSON of the arguments** and fires once per
group when:

1. The group has at least `min_repeats` calls (default 3), and
2. every result paired with those calls is byte-identical — or no result
   for them ever arrived at all.

The key is the structured tool call, not the surrounding AI message.
This is deliberate: an LLM never writes the same message twice (ids and
wording wobble), which is why `repeated_identical_action` — which needs
the whole node update to repeat verbatim — almost never fires on a real
tool loop. The model emits `get_weather({"city": "Paris"})` exactly,
and a deterministic tool returns the same string exactly.

**Deliberately NOT flagged:**
- The same call whose **results differ** between occurrences — polling a
  job status, paging through results with a cursor in the response,
  re-reading a file that is being written. That's progress against a
  changing world, not a loop.
- Calls to the same tool with **different arguments** (`Paris`, `Rome`,
  `Oslo`) — a fan-out, not a repeat.
- Two occurrences. A single retry is normal resilience.
- **Handoff / routing tools** — `transfer_to_<agent>`,
  `transfer_back_to_supervisor`, `handoff_*`, `route_to_*`,
  `delegate_to_*`. They take no arguments and return a canned
  acknowledgement by design, so "same call, same result" is their normal
  operation. Agents bouncing between each other is a node-sequence
  question that `infinite_handoff` owns.
- Tool calls drift itself injected into the initial state as chaos
  (`inject_tool_error_result`, `inject_tool_result_instruction`) — their
  ids carry the `drift_call_` prefix and are excluded, so only calls the
  graph's own nodes made are counted.

## What this typically means

The agent loop has no memory of having tried this already, and nothing in
the result tells it to stop. Concrete manifestations:

- **Retry on an error result with no backoff and no counter.** The tool
  returns `Error: 504`, the model decides to "try again", and the prompt
  gives it no reason not to try a fourth, fifth, sixth time. AgentChaos
  measured AutoGen's LLM calls per task rising 4.2× under an injected
  fault for exactly this reason.
- **A result the model can't parse into progress.** The tool returned
  something valid but unusable (an empty list, a schema it didn't
  expect) and the model keeps asking the same question hoping for a
  different answer.
- **A sub-agent that re-delegates the identical task** to the same
  specialist because the specialist's reply didn't land in the context
  it reads from (Cognition's cross-sibling sharing problem, at the tool
  layer).

## Sources

- **MAST 1.3 — Step Repetition.**
  ([arXiv:2503.13657](https://arxiv.org/abs/2503.13657))
- **Think Locally, Explain Globally** — defines *tool repetition rate*
  (share of tool calls that duplicate a previous invocation) as a
  controller-failure signal.
  ([arXiv:2601.17915](https://arxiv.org/abs/2601.17915))
- **AgentChaos** — retry amplification under injected faults (LLM calls
  per task up to 4.24× baseline).
  ([arXiv:2608.06790](https://arxiv.org/abs/2608.06790))

## Concrete example

**Trace (fires at step 6):**

```
step  node    message
────  ──────  ────────────────────────────────────────────────────
1     agent   ai   tool_calls=[get_weather({"city": "Paris"})]
2     tools   tool "Service unavailable"
3     agent   ai   tool_calls=[get_weather({"city": "Paris"})]
4     tools   tool "Service unavailable"
5     agent   ai   tool_calls=[get_weather({"city": "Paris"})]
6     tools   tool "Service unavailable"
7     agent   ai   "I could not get the weather."
```

Finding:

```
{
  "failure_type": "tool_call_loop",
  "agents_involved": ["agent"],
  "timestep": 5,
  "summary": "tool 'get_weather' was called 3 times with identical
              arguments {\"city\": \"Paris\"} (steps 1–5); every result
              was the same ('Service unavailable')"
}
```

## How to fix — architecture, not runtime

### 1. Put the retry policy in the graph, not in the model's judgement

A tool failure should route to an explicit retry node with a counter and
a ceiling, after which the graph takes a fallback edge. The model should
never be the thing deciding "try once more":

```python
def after_tools(state):
    last = state["messages"][-1]
    if getattr(last, "status", "") == "error":
        if state.get("attempts", 0) >= 2:
            return "fallback"
        return "retry"
    return "model"
```

### 2. Make identical re-invocation impossible by construction

Cache tool results per (name, args) within a run and return the cached
result with an explicit "you already asked this" marker. The model then
sees evidence that the question was already answered.

### 3. Give the tool a way to say "stop asking"

A tool that is down should return a result that *tells the agent what to
do next* ("service unavailable; do not retry; report to user"), not a
bare error string the model will read as "transient".

### What NOT to do

- **Don't just raise `recursion_limit`.** It makes the loop run longer
  before LangGraph kills it.
- **Don't add "do not repeat tool calls" to the system prompt** as the
  fix. It lowers the rate; it doesn't bound it. The graph has to.
