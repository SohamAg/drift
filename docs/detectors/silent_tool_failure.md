# `silent_tool_failure`

> A tool call failed and the system's final answer never let on. No
> retry, no fallback, no "I couldn't". The user gets a confident answer
> built on a step that didn't happen.

## What fires

The detector reads the graph's chat messages. It looks for a tool result
that is an **error**:

- `status="error"` on the ToolMessage (what LangGraph's `ToolNode` sets
  when `handle_tool_errors=True` catches an exception), or
- an error-shaped body — content starting with `Error` or containing a
  Python traceback — for tools that return failures as text.

For each such result it then scans the rest of the conversation. It
fires once per failed tool when **all** hold:

1. A later AI message exists with text content and no tool calls — the
   system *did* produce a final answer. (A trace that stops right after
   the error isn't silent; it's truncated or crashed, and the crash
   bucket already reports that.)
2. No later AI message calls the same tool again (no retry).
3. No later non-error result from the same tool arrives (no recovery).
4. No later AI message **surfaces** the failure. Two tiers:
   - **Strong** failure vocabulary — `error`, `failed`, `timed out`,
     `unavailable`, `retry`, `fallback`, `went wrong`, `rate limit`, … —
     counts wherever it appears.
   - **Weak** hedges — `can't`, `cannot`, `unable`, `couldn't`, `sorry`,
     `problem`, … — are ordinary prose. They count only when spoken by
     the agent that made the failed call, or in the final answer itself.
     A *different* specialist saying "I can't search the web" about its
     own tools is a capability statement, not an acknowledgment of your
     failed call — that exact sentence hid a real silent failure on a
     live supervisor run until this rule was split.

The failed call drift's own `inject_tool_error_result` chaos pattern
plants in the initial state **counts**. That pattern exists so this
detector can answer the pre-deploy question directly: when a dependency
fails, does your graph tell anyone?

**Deliberately NOT flagged:**
- The agent **retries** the tool, whether or not the retry succeeds.
- A later **successful** result from the same tool lands.
- The final answer **mentions** the failure in any recognisable way —
  even a hedge like "I couldn't verify this" counts as surfacing it.
- The **calling agent** hedges mid-run ("I can't get search results right
  now; answering from memory") — the agent that owned the call knew.
- An error result with **no final answer after it**.
- A second failure of the same tool in the same run (one finding per
  tool, anchored to the first unacknowledged occurrence).

## What this typically means

The error reached the model as a normal-looking tool message and the
model decided to answer anyway. AgentChaos (ASE 2026) names this outcome
class *Silent* — plausible output, no warning — and found it in 22–66%
of faulted runs across six agent frameworks, because an error inside a
tool result is an HTTP 200 to the agent loop and bypasses every
exception handler the developer wrote. Concrete manifestations:

- **The model fills the gap from prior knowledge.** The weather tool
  failed; the model "knows" Paris is usually mild; it answers.
- **A pipeline stage has no validation.** The failed result flows into
  the next node, which summarises whatever it was given.
- **Error handling exists but the routing doesn't check it.** The graph
  catches the exception, writes a ToolMessage, and the conditional edge
  routes on "did the model emit tool calls", not on "did the last tool
  fail".

## Sources

- **AgentChaos: Chaos Engineering for Agent Systems via Programmatic
  Fault Injection** — *Aborted / Reported / Silent* outcome classes;
  omission faults returning valid HTTP 200 bypass error handling.
  ([arXiv:2608.06790](https://arxiv.org/abs/2608.06790))
- **Anthropic — "How we built our multi-agent research system"** —
  errors silently treated as success named as a hard production failure.
  ([anthropic.com/engineering/built-multi-agent-research-system](https://www.anthropic.com/engineering/built-multi-agent-research-system))
- **MAST 3.3 — Incorrect Verification**, as the closest taxonomy slot:
  the output was never checked against the step it depends on.
  ([arXiv:2503.13657](https://arxiv.org/abs/2503.13657))

## Concrete example

**Trace (fires at step 3):**

```
step  node    message
────  ──────  ────────────────────────────────────────────────────────
1     agent   ai   tool_calls=[get_weather({"city": "Paris"})]
2     tools   tool status=error "Error: upstream timeout (HTTP 504)"
3     agent   ai   "It is 18°C and sunny in Paris today."
```

Finding:

```
{
  "failure_type": "silent_tool_failure",
  "agents_involved": ["agent", "tools"],
  "timestep": 3,
  "summary": "tool 'get_weather' returned an error at step 2 ('Error:
              upstream timeout (HTTP 504)'); the final answer at step 3
              ('It is 18°C and sunny in Paris today.') mentions no
              failure, and the tool was never retried"
}
```

## How to fix — architecture, not runtime

### 1. Route on tool status, not on tool-call presence

The conditional edge after `tools` should look at the last ToolMessage:

```python
def after_tools(state):
    last = state["messages"][-1]
    if getattr(last, "status", "") == "error":
        return "handle_failure"    # retry node, fallback source, or "tell the user"
    return "model"
```

### 2. Make the failure visible in state, not only in the message

Write `tool_errors: list[str]` via a reducer so a final "compose answer"
node can be *required* to mention anything in it. The
`unacknowledged_error` detector then covers the same failure from the
state side.

### 3. Validate before you summarise

If a node's job is to turn tool output into an answer, give it a schema
check first. An error string is not a weather report; refuse to compose.

### What NOT to do

- **Don't set `handle_tool_errors=False` to "make it crash instead".**
  A crash is louder, but now the user gets a stack trace and you've
  traded a silent failure for an outage.
- **Don't rely on "if a tool fails, say so" in the system prompt.** It
  helps at the margin. The graph has to enforce it.
