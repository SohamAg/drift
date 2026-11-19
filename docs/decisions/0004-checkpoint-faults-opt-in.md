# ADR-0004: Model and control-flow faults are written onto baseline checkpoints, and are opt-in

| | |
|---|---|
| Status | Accepted |
| Decided | 2026-11-15 23:12 IST — `69c5f6e` (mechanism); 23:15 IST — `b382113` (flags) |
| Recorded | 2026-11-19 (after the fact) |
| Owner | Soham Agarwal |

## Context
Two failure sources lie between "the input was wrong" and "a tool
misbehaved":
- the model emits a bad tool call (wrong arguments, a tool that does not
  exist, a duplicate, no call at all);
- a node silently does not run.

Neither can be produced by editing the initial input. Wrapping the chat
model to corrupt its output was possible, but drift does not own the
user's model object, and every provider wraps models differently.

## Decision
drift runs the baseline on a checkpointer and finds the checkpoint where
the fault belongs. For a model fault, that is the checkpoint where the
model's call is pending. For a skipped node, it is the checkpoint where
that node runs next. drift rewrites that state with LangGraph's own
`update_state`, written *as the node that produced it*, and resumes, so
the graph's real ToolNode executes the corrupted call. Both layers are
**off by default**, behind `--model-chaos` / `--flow-chaos`, and can also
be declared one by one in drift.yml.

Two rules come with it:
- **ToolNodes are never skipped.** A skipped ToolNode leaves a tool call
  with no result, which every chat API rejects. The crash would be the
  test's fault, not the graph's.
- **A fault with nowhere to land is reported as not applied.** That
  happens when the baseline never called the tool or never ran the node.

## Alternatives considered
- **Wrap the chat model.** Rejected, for the reasons in the context.
- **On by default.** Rejected. Existing users' test plans and CI baselines
  would change under them with no action on their part.

## Consequences
- Works on any compiled LangGraph graph, whatever model it uses.
- Only reaches the top-level graph's checkpoints. A call made inside a
  subgraph, such as a supervisor's handoff, is out of reach. ADR-0006
  works around this for handoffs.
- Faults reported "not applied" need careful counting (ADR-0007).
