# `unacknowledged_error`

> An agent's output signals an error and nothing downstream
> acknowledges, retries, or resolves it. The system just keeps going as
> if the step that failed had succeeded.

## What fires

The detector walks the trace looking for a step whose `update` dict
contains a recognized error signal with a real value:

- A structured field — `error`, `exception`, `traceback`,
  `stack_trace`, `failure_reason`, or `err` — set to anything other
  than `None`, `""`, or `False`.
- A boolean flag in the "bad" direction — `failed=True`,
  `success=False`, or `ok=False`.

Once an error signal is found at step K, the detector looks at the next
`within_steps` steps (default 3), **from any agent**, for an
acknowledgment:

- A structured field — `retry`, `retrying`, `retried`, `recovered`,
  `resolved`, `resolved_error`, `fallback`, `handled_error`,
  `error_handled`, `acknowledged`, or `escalated` — set to a truthy
  value.
- Free text (in any field, or the step's `rationale`) containing a
  recognized acknowledgment phrase (`retry`, `retrying`, `recovered`,
  `resolved`, `fallback`, `escalating`, …).
- The **same error field being explicitly cleared** in a later step
  within the window (e.g. `error: None` after `error: "timeout"`) —
  treated as self-resolution even without an explicit acknowledgment
  token.

If none of those appear within the window, the detector fires once for
that error occurrence.

**Deliberately NOT flagged:**
- An error field present but falsy (`error: ""`, `error: None`) — the
  key existing isn't the signal, a real value is.
- An error that gets retried, escalated, or explicitly cleared within
  the window — these are all legitimate ways of actually handling an
  error, not swallowing it.
- Every error occurrence fires independently — a trace with 3 separate
  unacknowledged errors produces 3 findings, each pointing at its own
  step.

## What this typically means

The graph's error-handling path is a no-op, or doesn't exist for this
error shape. Concrete manifestations:

- **Tool call fails, the node proceeds with stale/default data anyway**:
  a node catches an exception internally, writes it to an `error`
  field for observability, but its control flow doesn't branch on it —
  the rest of the pipeline runs on whatever partial state existed
  before the failure.
- **Error surfaced to state but the routing function doesn't check it**:
  the conditional edge after this node routes purely on a different
  field (e.g. a status enum) that the error path never updated, so the
  graph takes the "happy path" edge regardless.
- **Logged, not handled**: the error is genuinely just logging — a
  human or downstream system is expected to notice it later — but
  nothing in the trace shows that expectation being met within this
  run, which from drift's point of view looks identical to the error
  being silently dropped.

## Sources

- **Anthropic engineering blog — "How we built our multi-agent research
  system"** — names silently-swallowed errors in stateful tool calls
  as one of the harder production bugs to catch, precisely because
  nothing about the trace looks obviously wrong after the error — the
  system just keeps producing output.
  ([anthropic.com/engineering/multi-agent-research-system](https://www.anthropic.com/engineering/multi-agent-research-system))
- MAST does not enumerate silent error-swallowing as its own mode; this
  detector is tagged under MAST 2.3 (Task derailment) as the closest
  fit — continuing forward progress from an unacknowledged failure is a
  form of the system drifting from a correct trajectory.

## Concrete example

**Trace (fires on step 1):**

```
step  node     update
────  ───────  ──────────────────────────────────────
1     worker   {"error": "connection timeout"}
2     worker   {"next_step": "proceeding anyway"}
3     worker   {"next_step": "still going"}
4     worker   {"next_step": "done"}
```

Finding:

```
{
  "failure_type": "unacknowledged_error",
  "agents_involved": ["worker"],
  "timestep": 1,
  "summary": "agent 'worker' signaled an error at step 1
              (error='connection timeout') with no acknowledgment,
              retry, or resolution in the next 3 step(s)"
}
```

The `worker` node recorded a connection timeout and then just...
continued, three more steps, with no retry, no escalation, and no
later step clearing the `error` field.

## How to fix — architecture, not runtime

### 1. Route on the error field, don't just record it

```python
def worker(state):
    try:
        result = call_tool(state)
        return {"result": result, "error": None}
    except ToolError as e:
        return {"error": str(e)}

def route(state):
    if state.get("error"):
        return "handle_error"
    return "next_step"
```

Writing the error to state is only half the fix — something downstream
has to actually branch on it.

### 2. Give the error path a real resolution, not just a label

```python
def handle_error(state):
    # retry, fall back, or escalate — but DO something, and clear the
    # error field once it's actually addressed.
    return {"error": None, "retry": True, "result": _fallback(state)}
```

### 3. Make "logged, not handled" explicit if that's genuinely the design

If your system intentionally logs certain errors for later human review
without blocking the run, represent that as a real field the detector
(and a human reading the trace) can recognize as a deliberate choice —
e.g. `error_logged_for_review: true` — rather than leaving a bare
`error` field that looks identical to a dropped failure.

### What NOT to do

- **Don't add a "try/except around everything" blanket handler** that
  catches and silently discards exceptions at the graph level — that's
  the exact failure this detector catches, just moved one layer up.
- **Don't just clear the error field without actually addressing it**
  to stop this detector from firing. Clearing the field is this
  detector's signal for "resolved" — using it to mean "ignored" defeats
  the point of having the field at all.

## False positives and known limitations

- **Field vocabulary is heuristic.** A node that signals failure
  through a domain-specific field name outside the recognized list
  (e.g. `validation_errors`, `http_status`) won't be recognized as an
  error signal at all — false negative, not false positive.
- **Acknowledgment is proximity-based, not causal.** Any agent's step
  within the window counts as acknowledgment, even if it's unrelated to
  the actual error — a coincidental `retry: true` on a different task
  within the window would suppress a real finding. Keep `within_steps`
  tight for graphs with a lot of concurrent unrelated activity.
- **A deliberately logged-and-ignored error (by design) will fire this
  detector** unless you represent that choice explicitly in a field
  name that isn't in the default error vocabulary, or increase
  `within_steps` to cover wherever your design DOES eventually resolve
  it.
- **Text-only variant (`detect_from_text`) is line-proximity based**,
  not per-field — it returns the FIRST unacknowledged error line it
  finds in the whole transcript rather than every occurrence, since raw
  text has no structured way to track multiple simultaneous error
  states. Weaker than the structured path by design.

## Related detectors

- **`verifier_always_approves`** — a different failure mode in the same
  neighborhood: not an unacknowledged error, but a verification step
  that exists yet never actually catches anything.
- **Judge `gate_bypass` / `emergent_decay`** — the LLM judge can catch
  semantically-equivalent "the system ignored a problem" patterns that
  don't use any of this detector's recognized field vocabulary.

## Configuration

Optional kwarg to `detect()`:

| Arg | Default | Purpose |
|---|---|---|
| `within_steps` | `3` | How many steps after the error to look for acknowledgment |

## Cost

Zero LLM calls. Runs in <1ms on typical traces. Safe to enable in CI.
