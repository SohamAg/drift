"""`drift init` — scaffolding drift.yml and the GitHub workflow."""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from drift import guidelines
from drift.cli import main
from drift.scaffold import GraphFacts, init, render_drift_yml, render_workflow

pytest.importorskip("langgraph")

TOOL_GRAPH = textwrap.dedent('''
    from typing import Annotated, TypedDict
    from langchain_core.tools import tool
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode

    @tool
    def web_search(q: str) -> str:
        """search"""
        return "r"

    @tool
    def send_email(to: str, body: str) -> str:
        """send"""
        return "ok"

    class S(TypedDict):
        messages: Annotated[list, add_messages]
        budget: int

    def build(model=None):
        sg = StateGraph(S)
        sg.add_node("tools", ToolNode([web_search, send_email]))
        sg.add_edge(START, "tools")
        sg.add_edge("tools", END)
        return sg.compile()

    def initial_state(query=None):
        return {"messages": [], "budget": 3}
''')


@pytest.fixture
def tool_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    name = "drift_scaffold_tool_graph"
    (tmp_path / f"{name}.py").write_text(TOOL_GRAPH, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    sys.modules.pop(name, None)
    yield name
    sys.modules.pop(name, None)
    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))


def test_generic_templates_parse():
    spec = guidelines.parse_yaml(render_drift_yml())
    assert [r.id for r in spec.rules] == ["bounded-run"]
    wf = yaml.safe_load(render_workflow())
    step = wf["jobs"]["drift"]["steps"][-1]
    assert step["uses"] == "SohamAg/drift@main" and "guideline" in step["with"]["fail-on"]


def test_init_tailors_rules_to_the_graphs_tools(tool_module: str, tmp_path: Path):
    res = init(tmp_path, graph=tool_module)
    assert not res.facts.error
    assert res.facts.tools == ["send_email", "web_search"]
    assert res.facts.state_fields == ["messages", "budget"]
    text = (tmp_path / "drift.yml").read_text(encoding="utf-8")
    assert "tools:        send_email, web_search" in text
    assert "forbid_tool: send_email" in text  # side-effecting tool suggested for a guard
    # every commented suggestion is itself a valid rule once uncommented
    uncommented = "\n".join(
        line.replace("  # - ", "  - ", 1).replace("  #   ", "    ", 1) if line.startswith(("  # - ", "  #   ")) else line
        for line in text.splitlines()
    )
    spec = guidelines.parse_yaml(uncommented)
    assert len(spec.rules) > 3
    wf = yaml.safe_load((tmp_path / ".github" / "workflows" / "drift.yml").read_text(encoding="utf-8"))
    assert wf["jobs"]["drift"]["steps"][-1]["with"]["graph"] == tool_module


def test_init_never_overwrites_without_force(tmp_path: Path):
    (tmp_path / "drift.yml").write_text("rules: []\n", encoding="utf-8")
    res = init(tmp_path)
    assert tmp_path / "drift.yml" in res.skipped
    assert (tmp_path / "drift.yml").read_text(encoding="utf-8") == "rules: []\n"
    res = init(tmp_path, force=True)
    assert tmp_path / "drift.yml" in res.written


def test_unimportable_graph_falls_back_to_generic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["init", "no_such_mod_abc", "--no-workflow"]) == 0
    out = capsys.readouterr().out
    assert "could not import" in out and "wrote" in out
    assert not (tmp_path / ".github").exists()
    guidelines.load(tmp_path / "drift.yml")


def test_facts_without_tools_say_so():
    text = render_drift_yml(GraphFacts(spec="m", nodes=["a"], state_fields=["x"]))
    assert "no ToolNode found" in text and 'state: "bool(x)"' in text
