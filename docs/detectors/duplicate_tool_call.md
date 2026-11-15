# `duplicate_tool_call`

> One model turn asked for the same tool call twice — same tool, same
> arguments — and both executed. For a search, a wasted call. For
> `send_email`, `charge_card`, `create_ticket`: two emails, two charges,
> two tickets.

## What fires

For every AI message with two or more tool calls, the detector
groups the calls by tool name and JSON-serialised arguments (keys sorted).
It fires once per group of two or more calls in which **at least two calls
received a non-error tool result**, so it only fires when the tool actually
ran more than once.

Excluded:

- **Routing / handoff tools**: `transfer_to_*`, `handoff_*`, … (same rule
  as `novel_tool_call` and `tool_call_loop`).
- **Calls drift injected** as chaos (`drift_call_*` ids).

Side-effect tools (send / write / delete / pay / execute / create / …, the
same verb list `novel_tool_call` uses) are tagged and sorted first.

**Deliberately NOT flagged:**
- The same call **in a later turn**. After seeing a result, calling again
  is a decision the model made. A runaway repeat is `tool_call_loop`, and
  repeated work under a fault is `retry_amplification`.
- Identical calls where **only one executed** (the others errored). The
  side effect happened once.
- Calls with **different arguments**, even slightly different. Two
  searches for different queries in one turn is ordinary parallel tool use.

## What this typically means

- **The model emitted a parallel duplicate.** OpenAI parallel function
  calling is on by default, and models occasionally emit the same call
  twice in one turn. LangGraph's `ToolNode` executes every call it is
  given. Nothing between the model and the tool de-duplicates.
- **The tool is not idempotent, and nothing makes it so.** Payment and
  messaging APIs ship idempotency keys for exactly this. The tool wrapper
  never passes one.

## How to find out before a model does it

drift's model-layer chaos plants this fault on purpose:

```bash
drift run my_app.graph --model-chaos --fail-on finding:duplicate_tool_call
```

or one specific tool in `drift.yml`:

```yaml
chaos:
  - {tool: send_email, fault: model_duplicate_tool_call}
```

The fault rewrites the pending AI message at the baseline checkpoint so it
carries the call twice, then resumes. The ToolNode runs both calls for real.

## Concrete example

Live, gpt-4o-mini `create_react_agent` with a mocked `web_search`
(`examples/adapters/run_drift_on_model_flow_faults.py --part model`), 3
rollouts of `model_duplicate_tool_call`: the tool ran twice in 3/3 and the
final answer was correct and unremarkable. Before this detector existed, no
detector fired. With it:

```
duplicate_tool_call: one turn at step 2 requested tool '_web_search' 2× with
identical arguments {"query": "capital of France"} and 2 of those calls executed
```

## How to fix — architecture, not runtime

- **De-duplicate at the tool boundary.** Wrap the ToolNode, or the tool
  itself, so identical (name, args) pairs in one turn execute once and
  every call id gets the same result.
- **Make side-effecting tools idempotent.** Derive an idempotency key from
  the run/thread id plus the arguments and pass it to the API. A duplicate
  then becomes a no-op at the provider.
- **Turn off parallel tool calls** for agents that own side-effecting
  tools (`parallel_tool_calls=False` on OpenAI models) if their tasks
  never need it.

## Sources

- OpenAI function calling: parallel function calling is enabled by default.
- Stripe / AWS idempotency-key guidance for non-idempotent APIs.
- MAST 1.3 (Step Repetition), the tool-call-layer manifestation inside a
  single turn.
