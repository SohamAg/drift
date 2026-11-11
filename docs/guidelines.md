# Guideline rules — `drift.yml`

The detector library ships nineteen generic coordination failures. The
things your team actually cares about are specific to your graph: *the
researcher must never send email*, *refunds only after approval*, *the
answer must cite a ticket*, *the budget must not go negative*. Each is
one line in a `drift.yml` at your repo root, and each runs for free on
every run — the unperturbed baseline included, so a rule the graph
breaks **without** chaos is reported as a bug that needs no chaos.

```yaml
# drift.yml
judge:                                   # plain English, read by the LLM judge (costs judge calls)
  - "Refunds over $500 must be approved by the manager agent before being issued"

rules:                                   # deterministic, free, run on every graph run
  - id: approve-first
    tool_order: {before: approve_refund, after: issue_refund}
  - id: researcher-is-read-only
    forbid_tool: send_*
    agent: researcher
    severity: high
    why: researchers summarise; only the coordinator may contact customers
  - id: search-budget
    max_tool_calls: {tool: web_search, count: 3}
  - id: cite-ticket
    final_answer: {must_match: "#\\d+"}
  - id: no-ssn-anywhere
    no_message_match: "\\b\\d{3}-\\d{2}-\\d{4}\\b"
  - id: budget
    state: "budget_remaining >= 0"
  - id: idempotent-orders
    python: myapp.checks:no_duplicate_orders
```

`drift run` picks the file up automatically (`drift.yml`, `drift.yaml`,
`.drift/guidelines.yml`); `--guidelines path` points at another one,
`--no-guidelines` ignores it. In code, `drift_test(rules="drift.yml")`,
or pass a list of rule mappings / functions. In the web UI, paste the
file into the **Guideline rules** box on the Test or Import tab.

Every violation is a coordination finding with
`failure_type = "guideline:<id>"`, summary prefixed with the severity and
your `why`. Gate on them:

```bash
drift run my_app.graph:build --fail-on guideline               # any rule
drift run my_app.graph:build --fail-on finding:guideline:approve-first
```

A rule that **cannot be evaluated** — bad regex, a state field that does
not exist, an import that fails — is reported as
`guideline:<id>:error`. It never passes silently.

## Rule kinds

Each rule is a mapping with an `id` and exactly one of the keys below.
Optional on every rule: `severity` (`high` default, `medium`, `low`),
`why` (echoed in the finding), and — for tool rules — `agent`, a glob
over the calling node or sub-agent name (`researcher`, `*/tools`,
`critic`).

| Key | Value | Fires when |
|---|---|---|
| `forbid_tool` | glob | any matching tool is called (first call quoted, count given) |
| `require_tool` | glob | no matching tool was called by the end of the run |
| `tool_order` | `{before: glob, after: glob}` | an `after` tool is called before any `before` tool |
| `max_tool_calls` | int, or `{tool: glob, count: int}` | more matching calls than `count` |
| `max_steps` | int | the run took more super-steps |
| `final_answer` | `{must_match: regex}` and/or `{must_not_match: regex}` | the last AI message fails the regex (silent on graphs with no AI messages) |
| `no_message_match` | regex | any AI message text **or tool-call argument** matches |
| `state` | expression | the expression is false on the final state |
| `python` | `package.module:function` | the function returns a string or list of strings |

Globs are case-insensitive `fnmatch`: `send_*`, `transfer_to_*`, `*`.
Tool calls drift itself planted (`drift_call_*`) never count. Calls made
by a tool-dispatched sub-agent carry the sub-agent's name
(`critic (tools/model)`), so `agent: critic` matches them.

### `state:` expressions

The final state's top-level fields are names; so are `n_steps`,
`tool_calls` (tool names in call order), `final_answer`, `agents` and
`state` (the whole dict). Allowed: comparisons (chained too), `and` /
`or` / `not`, `+ - * / % //`, `in`, indexing and slicing, `x if c else
y`, list / tuple / set / dict displays, and the builtins `len min max abs
sum any all str int float bool sorted round`. No attribute access, no
other calls, no imports — the evaluator walks the AST and refuses
anything else at load time.

```yaml
- id: approval-gate
  state: "'issue_refund' not in tool_calls or 'approve_refund' in tool_calls"
- id: bounded
  state: "n_steps <= 20 and len(open_cases) <= 5"
- id: resolved
  state: "status in ['done', 'escalated']"
```

### Python rules

For anything the table cannot say. The function receives a `RunView`
and returns `None` / `""` to pass, a string for one violation, or a
list of strings.

```python
import drift

@drift.guideline("no-duplicate-orders", severity="medium", why="place_order is not idempotent")
def no_duplicate_orders(run):
    ids = [c.args.get("order_id") for c in run.calls("place_order")]
    if len(ids) != len(set(ids)):
        return f"place_order repeated for {ids}"
```

`RunView` fields: `steps`, `messages` (the conversation as `Msg`
records: `role`, `text`, `tool_calls`, `step`, `agent`), `tool_calls`
(`ToolCall`: `name`, `args`, `step`, `agent`, `index`), `final_state`,
`final_answer`, `n_steps`, `agents`, `calls(glob, agent=)`,
`ai_texts()`, and `ctx` (the raw `CoordinationDetectorContext`, with
`baseline_view()` for the unperturbed run when there is one).

Reference it from YAML as `python: myapp.checks:no_duplicate_orders`
(the decorator's id wins over the YAML `id`; undecorated functions work
too, using the function name as the id), or pass the function straight
to `drift_test(rules=[no_duplicate_orders])`.

## What this is not

Rules read the trace drift captured: node updates, the messages channel,
tool calls and results, the final state. They do not see inside a tool's
implementation, and a graph whose nodes write nothing to `messages` has
no tool calls or final answer for the tool and text rules to read — the
`state:` rules still work there. The LLM judge's `judge:` strings remain
the place for patterns only a reader can evaluate.
