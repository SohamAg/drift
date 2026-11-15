# Model-layer and control-flow chaos — faults on the graph's own checkpoints

[State chaos](../README.md) changes what the graph is given. [Tool
chaos](tool_chaos.md) changes what a tool returns. Two failure sources sit
between them, and no edit to the input can reach either:

- **The model emits a bad tool call.** The agent node's own output is
  wrong: arguments that fail the tool's schema, a tool that does not exist,
  the same call twice, or nothing where it should have called a tool.
- **A node silently does not run.** A worker times out upstream, a
  conditional edge is wrong, a deploy drops a node. Execution continues as
  if it had run and produced nothing.

Both layers work on the **baseline's checkpoints**. drift runs the graph
once unperturbed on a checkpointer, finds the exact point each fault
targets, writes the fault onto that checkpoint with `update_state`, and
resumes. The rest of the run is the graph's own response.

## Model layer

One pattern per (tool the graph reaches through a ToolNode × fault), named
like `model_bad_tool_args[web_search]`.

| Pattern | Fault class | What drift writes | What runs next |
|---|---|---|---|
| `model_bad_tool_args` | crash | the pending call to the tool, with **all arguments removed** | the ToolNode executes it, and the tool's schema validation error reaches the model |
| `model_unknown_tool` | crash | the call renamed to `<tool>_v2`, a tool that does not exist | the ToolNode answers "`<tool>_v2` is not a valid tool, try one of [...]" |
| `model_duplicate_tool_call` | duplication | the call twice, same arguments, two ids | the tool **executes twice** |
| `model_dropped_tool_call` | loss | the AI message with its tool call removed and no text | the graph routes as if the model answered, with nothing |

Where: the first baseline checkpoint whose last message is an AI message
with a pending call to that tool. The rewritten message keeps its id, so
`add_messages` replaces it, and it is written **as the node that produced
it**, so the graph re-evaluates that node's outgoing edges. Provider-raw
`additional_kwargs["tool_calls"]` is cleared so no serializer brings the
original call back.

## Control-flow layer

`skip_node[N]`, one per root node that is not a ToolNode. At the
checkpoint where the baseline was about to run N, drift writes an empty
update **as N**. LangGraph records N as having run with no output and
continues along N's edges. Downstream nodes see whatever N would have
changed, unchanged.

ToolNodes are never skipped. Their caller's tool call would be left with no
result, a message sequence every chat API rejects. LangGraph's react agent
raises on it immediately, so the "failure" would be drift's, not the graph's.
Skipping one branch of a parallel step is not supported either, and is
reported as not applied.

## Turning them on

They are opt-in for the sampled plan, so existing plans don't change:

```bash
drift run my_app.graph --model-chaos --flow-chaos --fail-on crash,finding:duplicate_tool_call
```

```python
drift_test(graph=g, initial_state=s, model_chaos=True, flow_chaos=True)
```

Or name exactly the faults you want in `drift.yml`. These always run,
first, outside `max_perturbations`:

```yaml
chaos:
  - {tool: send_email, fault: model_duplicate_tool_call}
  - {tool: web_search, fault: model_bad_tool_args}
  - {skip_node: reviewer}
```

An unknown tool or node, or a ToolNode named in `skip_node`, is an error.
They need a compiled LangGraph graph; drift attaches an in-memory
checkpointer if it has none. The web server takes `model_chaos` /
`flow_chaos` on the run requests.

## When a fault has nowhere to land

If the baseline never made a call to the tool, or never had the node as
its next step, there is no checkpoint to write the fault on. The
perturbation is reported with `fault_fired = False` and the reason ("the
baseline never had 'math_expert' as its next node…"). The graph is not run
again, and the fault is not counted as one the graph coped with.

## Live results

`examples/adapters/run_drift_on_model_flow_faults.py`, gpt-4o-mini,
2026-11-15. These are rates over the rollouts, not verdicts.

**Model layer**: `create_react_agent` + a mocked `web_search`, 3 rollouts each.

| Fault | Fired | What the agent did | Findings |
|---|---|---|---|
| `model_bad_tool_args` | 3/3 | read the "query: Field required" error, re-called `web_search` with the right arguments, answered correctly | none (a recovery) |
| `model_unknown_tool` | 3/3 | read "try one of [_web_search]", called the real tool, answered correctly | none after the fix below (3/3 `silent_tool_failure` before it) |
| `model_duplicate_tool_call` | 3/3 | the tool ran twice, and the answer was correct and unremarkable | `duplicate_tool_call` 3/3 (a new detector; before it, nothing fired) |
| `model_dropped_tool_call` | 3/3 | the run ended with an empty answer | divergence only. There is no next model turn to recover in |

**Control-flow layer**: the 2-agent langgraph-supervisor system
(supervisor → research_expert / math_expert), "look up the populations of
Tokyo and Paris, then sum them", 2 rollouts each.

| Fault | Fired | What happened |
|---|---|---|
| `skip_node[supervisor]` | 2/2 | the run ended at once with an empty answer |
| `skip_node[research_expert]` | 2/2 | the supervisor did not notice its research handoff came back empty. It routed to `math_expert`, which supplied precise-looking populations from its own knowledge (13,929,286 / 2,140,526) and summed them. The answer reads as sourced and is not. `novel_tool_call` 2/2 (math tools the baseline never needed), and `empty_handoff` (added for this, see below) names the root cause |
| `skip_node[math_expert]` | 0/1 | not applied: in the baseline the supervisor summed the numbers itself and never ran `math_expert` |

The two runs found three real bugs in drift itself, all fixed with tests:
`silent_tool_failure` did not count a corrected tool name as a retry;
`conversation()` dropped all but one result of a parallel tool
super-step; and the "at least 2 root steps" rule of mid-run state timing
was blocking a skip of a graph's first node.

## Open

- ~~Empty handoff~~: shipped as the [`empty_handoff`](detectors/empty_handoff.md)
  detector. On the 92 saved runs under `results/`, it fires only on this
  `skip_node[research_expert]` run.
- Model faults on tools called outside a ToolNode are unreachable (same
  boundary as tool chaos).
- `skip_node` inside subgraphs: only root nodes are skippable. LangGraph
  cannot write a checkpoint inside a subgraph from the parent.
