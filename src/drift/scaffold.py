"""`drift init` — scaffold drift.yml and a GitHub workflow into a repo.

Given a graph spec it imports the graph (same loader as `drift run`) and
tailors the files: the workflow runs that graph, and drift.yml lists the
tools the graph can call and the state fields it carries as commented
starting points, so the first rule a user writes names a real tool.
Without a spec — or when the import fails — it writes the generic
templates and says why.

Never overwrites an existing file unless `force=True`. Every drift.yml it
writes parses with drift.guidelines before it is written.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["GraphFacts", "InitResult", "inspect_graph", "render_drift_yml", "render_workflow", "init"]

WORKFLOW_PATH = Path(".github") / "workflows" / "drift.yml"
RULES_PATH = Path("drift.yml")


@dataclass
class GraphFacts:
    spec: str = ""
    tools: list[str] = field(default_factory=list)
    handoff_tools: list[str] = field(default_factory=list)   # misroutable handoffs (transfer_to_*)
    skippable: list[str] = field(default_factory=list)       # nodes skip_node can target
    state_fields: list[str] = field(default_factory=list)
    nodes: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class InitResult:
    written: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    facts: GraphFacts = field(default_factory=GraphFacts)


def inspect_graph(spec: str, initial_state_fn: str = "initial_state") -> GraphFacts:
    """Import `module[:build_fn]` and read its tools, nodes and state fields."""
    from drift.cli import _import_graph, _parse_graph_spec

    facts = GraphFacts(spec=spec)
    try:
        module, build_fn = _parse_graph_spec(spec)
        graph, state, _meta = _import_graph(module, build_fn, initial_state_fn, None, None)
    except SystemExit as e:  # _import_graph reports user errors this way
        facts.error = str(e)
        return facts
    except Exception as e:  # noqa: BLE001 — scaffolding must not crash on a user's import
        facts.error = f"{type(e).__name__}: {e}"
        return facts
    try:
        from drift.chaos.tools import applies_to, tool_names

        facts.tools = list(tool_names(graph))
        facts.handoff_tools = [t for t in facts.tools if applies_to(graph, t, "handoff_misroute")]
    except Exception:  # noqa: BLE001
        facts.tools = []
    try:
        from drift.chaos.checkpoint import skippable_nodes

        facts.skippable = skippable_nodes(graph)
    except Exception:  # noqa: BLE001
        facts.skippable = []
    nodes = getattr(getattr(graph, "bound", graph), "nodes", None)
    if isinstance(nodes, dict):
        facts.nodes = [str(n) for n in nodes if n not in ("__start__", "__end__")]
    if isinstance(state, dict):
        facts.state_fields = [str(k) for k in state]
    return facts


def render_drift_yml(facts: GraphFacts | None = None) -> str:
    facts = facts or GraphFacts()
    lines = [
        "# drift.yml — rules drift checks on every run of your graph, baseline included.",
        "# Reference: https://github.com/SohamAg/drift/blob/main/docs/guidelines.md",
        "# A violation is a finding `guideline:<id>`; gate on it with --fail-on guideline.",
        "",
    ]
    if facts.spec and not facts.error:
        lines.append(f"# Generated from {facts.spec}:")
        lines.append(f"#   nodes:        {', '.join(facts.nodes) or '(none found)'}")
        lines.append(f"#   tools:        {', '.join(facts.tools) or '(no ToolNode found — tool rules will never fire)'}")
        lines.append(f"#   state fields: {', '.join(facts.state_fields) or '(none)'}")
        lines.append("")
    lines += [
        "# Plain-English checks for the LLM judge (only used with --judge openai / presets above quick).",
        "judge: []",
        "",
        "rules:",
        "  # Bound the work a run can do. Raise it if your graph legitimately needs more.",
        "  - id: bounded-run",
        "    max_steps: 50",
        "    severity: medium",
        "    why: a run that needs more super-steps than this is probably looping",
    ]
    # handoff tools route between agents; they are not side effects to forbid
    handoffs = set(facts.handoff_tools) | {t for t in facts.tools if t.startswith("transfer_to_")}
    tools = [t for t in facts.tools if t not in handoffs]
    if tools:
        first = tools[0]
        lines += [
            "",
            "  # Starting points for the tools this graph can call — uncomment and edit.",
            f"  # - id: {first.replace('_', '-')}-budget",
            f"  #   max_tool_calls: {{tool: {first}, count: 5}}",
        ]
        if len(tools) > 1:
            lines += [
                f"  # - id: {tools[0].replace('_', '-')}-before-{tools[1].replace('_', '-')}",
                f"  #   tool_order: {{before: {tools[0]}, after: {tools[1]}}}",
            ]
        side_effects = [t for t in tools if any(k in t.lower() for k in ("send", "email", "post", "delete",
                                                                         "pay", "refund", "write", "create",
                                                                         "update", "transfer"))]
        for t in side_effects[:3]:
            lines += [
                f"  # - id: no-unapproved-{t.replace('_', '-')}",
                f"  #   forbid_tool: {t}",
                "  #   agent: \"*\"            # narrow to the agents that must never call it",
            ]
    else:
        lines += [
            "",
            "  # Examples — uncomment and edit:",
            "  # - id: researcher-is-read-only",
            "  #   forbid_tool: send_*",
            "  #   agent: researcher",
            "  # - id: approve-first",
            "  #   tool_order: {before: approve_refund, after: issue_refund}",
        ]
    if facts.state_fields:
        f0 = facts.state_fields[0]
        lines += [
            "",
            "  # State rules are expressions over the final state's fields:",
            f"  # - id: {f0.replace('_', '-')}-present",
            f"  #   state: \"bool({f0})\"",
        ]
    lines += [
        "",
        "# Declared chaos: faults that always run, before the sampled plan.",
        "chaos: []",
    ]
    examples = []
    if tools:
        examples += [f"#   - {{tool: {tools[0]}, fault: tool_error}}            # the tool returns an API error",
                     f"#   - {{tool: {tools[0]}, fault: model_wrong_tool_args}}  # right tool, plausible wrong argument"]
    if facts.handoff_tools:
        examples.append(f"#   - {{tool: {facts.handoff_tools[0]}, fault: handoff_misroute}}  "
                        "# the handoff reaches a different agent")
    if facts.skippable:
        examples.append(f"#   - {{skip_node: {facts.skippable[-1]}}}               # that node silently does not run")
    if examples:
        lines += ["# For example (replace `chaos: []` with):", "# chaos:", *examples]
    return "\n".join(lines) + "\n"


def render_workflow(spec: str = "", *, fail_on: str = "crash,recursion_limit,guideline", preset: str = "quick") -> str:
    graph_line = f"          graph: {spec}" if spec else "          graph: my_app.graph:build   # module[:build_fn]; module exposes initial_state()"
    return f"""# drift — chaos-test the graph on every pull request.
# Docs: https://github.com/SohamAg/drift/blob/main/docs/ci.md
name: drift

on:
  pull_request:
  workflow_dispatch:

jobs:
  drift:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install your app
        run: pip install -e .   # or: pip install -r requirements.txt
      - uses: SohamAg/drift@main
        # env:
        #   OPENAI_API_KEY: ${{{{ secrets.OPENAI_API_KEY }}}}   # presets above quick
        with:
{graph_line}
          preset: {preset}
          fail-on: {fail_on}
          python-version: ""   # already set up above
"""


def init(directory: str | Path = ".", *, graph: str = "", initial_state_fn: str = "initial_state",
         force: bool = False, workflow: bool = True, rules: bool = True) -> InitResult:
    from drift.guidelines import parse_yaml

    root = Path(directory)
    res = InitResult()
    if graph:
        res.facts = inspect_graph(graph, initial_state_fn)
    files: list[tuple[Path, str]] = []
    if rules:
        text = render_drift_yml(res.facts)
        parse_yaml(text)  # never write a file drift itself would reject
        files.append((root / RULES_PATH, text))
    if workflow:
        files.append((root / WORKFLOW_PATH, render_workflow(graph)))
    for path, text in files:
        if path.exists() and not force:
            res.skipped.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        res.written.append(path)
    return res
