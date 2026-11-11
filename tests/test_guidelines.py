"""drift.guidelines — the deterministic rule language (drift.yml `rules:`),
the `@drift.guideline` decorator, and their wiring into the library, the
adapter, the gate, the CLI and the server."""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

import drift
from drift import gate, guidelines as g
from drift.failures.library import run_all_on_trace
from drift.failures.library.base import from_adapter_trace


# ---- trace fixtures ---------------------------------------------------------

def _ai(text="", calls=None, name=""):
    m = {"type": "ai", "content": text}
    if name:
        m["name"] = name
    if calls:
        m["tool_calls"] = [{"name": n, "args": a, "id": i} for (n, a, i) in calls]
    return m


def _tool(text, call_id):
    return {"type": "tool", "content": text, "tool_call_id": call_id}


def _human(text):
    return {"type": "human", "content": text}


INIT = {"messages": [_human("Refund order 42 if it qualifies.")], "budget_remaining": 100}


def _trace(steps, initial=INIT, extra_state=None):
    running = list(initial["messages"])
    out = []
    state_extra = dict(extra_state or {})
    for i, (node, msgs, *rest) in enumerate(steps, start=1):
        running = running + msgs
        if rest:
            state_extra.update(rest[0])
        out.append({"step": i, "node": node, "update": {"messages": msgs, **(rest[0] if rest else {})},
                    "state_after": {"messages": list(running), "budget_remaining": initial["budget_remaining"],
                                    **state_extra}})
    return out


GOOD = _trace([
    ("agent", [_ai(calls=[("lookup_order", {"id": 42}, "c1")])]),
    ("tools", [_tool("order 42: $30, eligible", "c1")]),
    ("agent", [_ai(calls=[("approve_refund", {"id": 42}, "c2")])]),
    ("tools", [_tool("approved", "c2")]),
    ("agent", [_ai(calls=[("issue_refund", {"id": 42, "amount": 30}, "c3")])]),
    ("tools", [_tool("refunded", "c3")]),
    ("agent", [_ai("Refunded $30 for ticket #42.")]),
])

BAD = _trace([
    ("agent", [_ai(calls=[("lookup_order", {"id": 42}, "c1")])]),
    ("tools", [_tool("order 42: $30, eligible", "c1")]),
    ("agent", [_ai(calls=[("issue_refund", {"id": 42, "amount": 30}, "c2")])]),
    ("tools", [_tool("refunded", "c2")]),
    ("researcher", [_ai(calls=[("send_email", {"to": "x", "body": "SSN 123-45-6789"}, "c3")])]),
    ("tools", [_tool("sent", "c3")]),
    ("agent", [_ai("Done, refund issued and customer notified.")]),
], extra_state={"budget_remaining": -5})


def _ctx(trace):
    return from_adapter_trace(trace, initial_state=INIT)


def _types(findings):
    return sorted(f.failure_type for f in findings)


# ---- parsing ------------------------------------------------------------------

def test_parse_yaml_full_spec():
    spec = g.parse_yaml(textwrap.dedent("""
        judge:
          - "Refunds over $500 need a manager"
        rules:
          - id: no-email
            forbid_tool: send_*
            agent: researcher
            severity: medium
            why: researchers are read-only
          - id: approve-first
            tool_order: {before: approve_refund, after: issue_refund}
        chaos:
          - tool: lookup_order
            fault: timeout
    """))
    assert spec.judge == ["Refunds over $500 need a manager"]
    assert [r.id for r in spec.rules] == ["no-email", "approve-first"]
    assert spec.rules[0].kind == "forbid_tool" and spec.rules[0].agent == "researcher"
    assert spec.rules[0].severity == "medium" and spec.rules[0].why == "researchers are read-only"
    assert spec.chaos == [{"tool": "lookup_order", "fault": "timeout"}]
    assert spec.rule_dicts()[0] == {"id": "no-email", "forbid_tool": "send_*", "severity": "medium",
                                    "why": "researchers are read-only", "agent": "researcher"}


def test_parse_accepts_bare_lists_and_strings():
    assert g.parse(["a", "b"]).judge == ["a", "b"]
    assert g.parse([{"id": "x", "max_steps": 3}]).rules[0].kind == "max_steps"
    assert g.parse(None).is_empty
    assert g.parse("only one").judge == ["only one"]


@pytest.mark.parametrize("bad, msg", [
    ([{"max_steps": 3}], "no `id`"),
    ([{"id": "a", "max_steps": 3, "forbid_tool": "x"}], "exactly one of"),
    ([{"id": "a"}], "exactly one of"),
    ([{"id": "a", "max_steps": 0}], "int ≥ 1"),
    ([{"id": "a", "tool_order": {"before": "x"}}], "before"),
    ([{"id": "a", "final_answer": {"must_match": "("}}], "bad regex"),
    ([{"id": "a", "state": "import os"}], "does not parse"),
    ([{"id": "a", "state": "x.__class__"}], "not allowed"),
    ([{"id": "a", "state": "open('f')"}], "may be called"),
    ([{"id": "a", "python": "nomodule"}], "package.module:function"),
    ([{"id": "a", "max_steps": 3, "bogus": 1}], "unknown key"),
    ([{"id": "a", "max_steps": 3}, {"id": "a", "max_steps": 4}], "duplicate rule id"),
    ([{"id": "a", "max_steps": 3, "severity": "fatal"}], "severity"),
])
def test_parse_rejects_malformed_rules(bad, msg):
    with pytest.raises(g.GuidelineError, match=msg):
        g.parse_rules(bad)


def test_parse_rejects_unknown_top_level_keys_and_bad_yaml():
    with pytest.raises(g.GuidelineError, match="unknown top-level"):
        g.parse({"rule": []})
    with pytest.raises(g.GuidelineError, match="does not parse"):
        g.parse_yaml("rules: [\n  - id: x")


def test_load_txt_is_judge_lines_and_load_default_finds_drift_yml(tmp_path):
    (tmp_path / "g.txt").write_text("one\n\n two \n", encoding="utf-8")
    assert g.load(tmp_path / "g.txt").judge == ["one", "two"]
    assert g.load_default(tmp_path) is None
    (tmp_path / ".drift").mkdir()
    (tmp_path / ".drift" / "guidelines.yml").write_text("rules:\n  - id: s\n    max_steps: 2\n", encoding="utf-8")
    assert [r.id for r in g.load_default(tmp_path).rules] == ["s"]
    (tmp_path / "drift.yml").write_text("judge: [x]\n", encoding="utf-8")
    assert g.load_default(tmp_path).judge == ["x"]  # drift.yml wins over .drift/


# ---- each rule kind -------------------------------------------------------------

def test_forbid_tool_with_agent_filter_and_glob():
    rules = [{"id": "no-email", "forbid_tool": "send_*", "agent": "researcher"}]
    assert _types(g.evaluate(rules, _ctx(GOOD))) == []
    out = g.evaluate(rules, _ctx(BAD))
    assert _types(out) == ["guideline:no-email"]
    assert "send_email" in out[0].summary and "researcher" in out[0].summary and out[0].timestep == 5
    # agent filter: the same call from a different node does not count
    assert g.evaluate([{"id": "x", "forbid_tool": "send_*", "agent": "agent"}], _ctx(BAD)) == []


def test_require_tool_and_tool_order():
    assert g.evaluate([{"id": "r", "require_tool": "approve_*"}], _ctx(GOOD)) == []
    out = g.evaluate([{"id": "r", "require_tool": "approve_*"}], _ctx(BAD))
    assert _types(out) == ["guideline:r"] and "never called" in out[0].summary
    order = [{"id": "o", "tool_order": {"before": "approve_refund", "after": "issue_refund"}}]
    assert g.evaluate(order, _ctx(GOOD)) == []
    out = g.evaluate(order, _ctx(BAD))
    assert "'issue_refund' was called at step 3 before any 'approve_refund'" in out[0].summary
    # `after` never called → nothing to order
    short = _trace([("agent", [_ai("nothing to do")])])
    assert g.evaluate(order, _ctx(short)) == []


def test_max_tool_calls_and_max_steps():
    assert g.evaluate([{"id": "m", "max_tool_calls": 3}], _ctx(GOOD)) == []
    out = g.evaluate([{"id": "m", "max_tool_calls": 2}], _ctx(GOOD))
    assert "3 calls to '*' exceed the limit of 2" in out[0].summary
    out = g.evaluate([{"id": "m", "max_tool_calls": {"tool": "issue_*", "count": 0}}], _ctx(GOOD))
    assert _types(out) == ["guideline:m"]
    assert g.evaluate([{"id": "s", "max_steps": 7}], _ctx(GOOD)) == []
    out = g.evaluate([{"id": "s", "max_steps": 6}], _ctx(GOOD))
    assert "took 7 steps" in out[0].summary


def test_final_answer_and_no_message_match():
    cite = [{"id": "cite", "final_answer": {"must_match": r"#\d+"}}]
    assert g.evaluate(cite, _ctx(GOOD)) == []
    out = g.evaluate(cite, _ctx(BAD))
    assert "does not match" in out[0].summary
    out = g.evaluate([{"id": "n", "final_answer": {"must_not_match": "refund"}}], _ctx(GOOD))
    assert "forbidden" in out[0].summary
    ssn = [{"id": "pii", "no_message_match": r"\b\d{3}-\d{2}-\d{4}\b"}]
    assert g.evaluate(ssn, _ctx(GOOD)) == []
    out = g.evaluate(ssn, _ctx(BAD))
    assert out[0].timestep == 5 and "123-45-6789" in out[0].summary  # found in tool-call args
    # a graph with no AI text has no final answer to judge → silent
    no_ai = [{"step": 1, "node": "n", "update": {"x": 1}, "state_after": {"x": 1}}]
    assert g.evaluate(cite, from_adapter_trace(no_ai)) == []


def test_state_expressions_see_fields_and_run_facts():
    ok = [{"id": "b", "state": "budget_remaining >= 0"}]
    assert g.evaluate(ok, _ctx(GOOD)) == []
    out = g.evaluate(ok, _ctx(BAD))
    assert "is false" in out[0].summary and "-5" in out[0].summary
    assert g.evaluate([{"id": "t", "state": "'issue_refund' not in tool_calls or 'approve_refund' in tool_calls"}],
                      _ctx(GOOD)) == []
    assert _types(g.evaluate([{"id": "t", "state": "'issue_refund' not in tool_calls or 'approve_refund' in tool_calls"}],
                             _ctx(BAD))) == ["guideline:t"]
    assert g.evaluate([{"id": "n", "state": "n_steps <= 7 and len(messages) > 3 and '#42' in final_answer"}],
                      _ctx(GOOD)) == []
    assert g.evaluate([{"id": "c", "state": "max(1, 2) == 2 and (budget_remaining if True else 0) == 100"}],
                      _ctx(GOOD)) == []


def test_unevaluable_rule_is_an_error_finding_not_a_silent_pass():
    out = g.evaluate([{"id": "typo", "state": "budget_remaning >= 0"}], _ctx(GOOD))
    assert _types(out) == ["guideline:typo:error"]
    assert "NameError" in out[0].summary and "budget_remaining" in out[0].summary
    out = g.evaluate([{"id": "imp", "python": "drift.no_such_module:fn"}], _ctx(GOOD))
    assert _types(out) == ["guideline:imp:error"]


def test_python_rules_via_decorator_callable_and_module_path():
    @drift.guideline("no-dup-refund", severity="low", why="idempotency")
    def no_dup(run):
        ids = [c.args.get("id") for c in run.calls("issue_refund")]
        if len(ids) != len(set(ids)):
            return f"issue_refund repeated for {ids}"

    assert no_dup.__drift_guideline__.id == "no-dup-refund"
    assert g.evaluate([no_dup], _ctx(GOOD)) == []
    dup = _trace([
        ("agent", [_ai(calls=[("issue_refund", {"id": 1}, "a"), ("issue_refund", {"id": 1}, "b")])]),
        ("tools", [_tool("ok", "a"), _tool("ok", "b")]),
    ])
    out = g.evaluate([no_dup], _ctx(dup))
    assert out[0].failure_type == "guideline:no-dup-refund"
    assert out[0].summary.startswith("[low] issue_refund repeated") and "idempotency" in out[0].summary

    def plain(run):  # undecorated callables work too; id = function name
        return ["one", "two"] if run.n_steps > 1 else None
    assert [f.summary for f in g.evaluate([plain], _ctx(GOOD))] == ["[high] one", "[high] two"]

    out = g.evaluate([{"id": "mod", "python": "tests.test_guidelines:_module_rule"}], _ctx(BAD))
    assert _types(out) == ["guideline:mod"] and "researcher" in out[0].summary


def _module_rule(run):
    bad = [c for c in run.tool_calls if c.agent.startswith("researcher")]
    return f"researcher made {len(bad)} tool call(s)" if bad else None


def test_run_view_exposes_calls_with_subagent_labels():
    trace = _trace([
        ("supervisor", [_ai(calls=[("task", {"subagent_type": "critic", "description": "review"}, "c1")])]),
        ("tools/model", [_ai(calls=[("web_search", {"q": "x"}, "c2")])]),
        ("tools/tools", [_tool("r", "c2")]),
    ])
    trace[1]["subagent"] = "critic"
    view = g.build_view(from_adapter_trace(trace, initial_state=INIT))
    assert [c.name for c in view.tool_calls] == ["task", "web_search"]
    assert view.tool_calls[1].agent == "critic (tools/model)"
    assert g.evaluate([{"id": "c", "forbid_tool": "web_search", "agent": "critic"}], view.ctx)
    assert not g.evaluate([{"id": "c", "forbid_tool": "web_search", "agent": "supervisor"}], view.ctx)


# ---- wiring -----------------------------------------------------------------------

def test_library_runner_appends_guideline_findings():
    out = run_all_on_trace(BAD, initial_state=INIT,
                           guideline_rules=[{"id": "b", "state": "budget_remaining >= 0"}])
    assert "guideline:b" in [f.failure_type for f in out]
    assert "guideline:b" not in [f.failure_type for f in run_all_on_trace(BAD, initial_state=INIT)]


def test_gate_guideline_rule_and_finding_prefix():
    shaped = {
        "baseline": {"coordination_findings": [{"failure_type": "guideline:b"}], "judge_findings": []},
        "perturbations": [
            {"event_name": "p0", "crashed": False, "diverged": False,
             "coordination_findings": [{"failure_type": "guideline:no-email"}], "judge_findings": []},
            {"event_name": "p1", "crashed": False, "diverged": False, "coordination_findings": [],
             "judge_findings": []},
        ],
        "n_judge_findings": 0,
    }
    r = gate.evaluate(shaped, ["guideline"])
    assert not r.passed
    assert r.violations[0].summary == "guideline rule(s) b, no-email violated on 1/2 perturbation(s) and the baseline"
    assert r.violations[0].perturbations == ["p0"]
    assert gate.evaluate(shaped, ["finding:guideline:no-email"]).violations[0].perturbations == ["p0"]
    assert gate.parse_rules(["guideline"]) == ["guideline"]


@pytest.fixture
def tool_graph():
    pytest.importorskip("langgraph")
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    @tool
    def issue_refund(order_id: int) -> str:
        """Issue a refund."""
        return "refunded"

    def agent(state):
        n = sum(1 for m in state["messages"] if getattr(m, "type", "") == "tool")
        if n == 0:
            return {"messages": [AIMessage(content="", tool_calls=[
                {"name": "issue_refund", "args": {"order_id": 1}, "id": "k1"}])]}
        return {"messages": [AIMessage(content="Refund issued for #1.")]}

    def route(state):
        return "tools" if state["messages"][-1].tool_calls else END

    sg = StateGraph(MessagesState)
    sg.add_node("agent", agent)
    sg.add_node("tools", ToolNode([issue_refund]))
    sg.add_edge(START, "agent")
    sg.add_conditional_edges("agent", route, ["tools", END])
    sg.add_edge("tools", "agent")
    return sg.compile()


def test_drift_test_runs_rules_on_baseline_and_perturbations(tool_graph, tmp_path):
    from drift.adapters.langgraph import drift_test

    spec_path = tmp_path / "drift.yml"
    spec_path.write_text(textwrap.dedent("""
        judge: ["refunds need approval"]
        rules:
          - id: approve-first
            tool_order: {before: approve_refund, after: issue_refund}
          - id: cite
            final_answer: {must_match: "#\\\\d+"}
    """), encoding="utf-8")
    res = drift_test(graph=tool_graph, initial_state={"messages": [("user", "refund order 1")]},
                     intensity="light", max_perturbations=2, seed=3, rules=spec_path)
    base_types = [f["failure_type"] for f in res.baseline.coordination_findings]
    assert "guideline:approve-first" in base_types          # the graph never approves: a bug without chaos
    assert "guideline:cite" not in base_types
    for p in res.perturbations:
        assert all(not t.startswith("guideline:cite") for t in (f["failure_type"] for f in p.coordination_findings)) \
            or p.crashed or "#" not in json.dumps(p.final_state, default=str)

    # a list of rule dicts and a GuidelineSpec both work; a malformed one is a ValueError
    res2 = drift_test(graph=tool_graph, initial_state={"messages": [("user", "x")]}, intensity="off",
                      rules=[{"id": "m", "max_tool_calls": 0}])
    assert [f["failure_type"] for f in res2.baseline.coordination_findings if f["failure_type"].startswith("guideline")] == ["guideline:m"]
    with pytest.raises(ValueError, match="rules"):
        drift_test(graph=tool_graph, initial_state={"messages": []}, intensity="off", rules=[{"max_steps": 1}])


def test_cli_run_picks_up_drift_yml_and_fails_the_gate(tmp_path, monkeypatch, capsys):
    pytest.importorskip("langgraph")
    from drift.cli import main

    name = "drift_rules_cli_graph"
    (tmp_path / f"{name}.py").write_text(textwrap.dedent('''
        from typing import TypedDict
        from langgraph.graph import END, START, StateGraph

        class S(TypedDict):
            query: str
            budget: int
            reply: str

        def spend(state):
            return {"budget": state["budget"] - 150, "reply": "spent"}

        def build(model=None):
            sg = StateGraph(S)
            sg.add_node("spend", spend)
            sg.add_edge(START, "spend")
            sg.add_edge("spend", END)
            return sg.compile()

        def initial_state(query="hello"):
            return {"query": query, "budget": 100, "reply": ""}
    '''), encoding="utf-8")
    (tmp_path / "drift.yml").write_text("rules:\n  - id: budget\n    state: budget >= 0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    sys.modules.pop(name, None)
    try:
        out_json = tmp_path / "out.json"
        code = main(["run", f"{name}:build", "--json", str(out_json), "--fail-on", "guideline", "--seed", "1"])
        captured = capsys.readouterr()
        assert code == 1
        assert "guidelines: drift.yml" in captured.err
        assert "[guideline] guideline rule(s) budget violated" in captured.out
        shaped = json.loads(out_json.read_text(encoding="utf-8"))
        assert shaped["guideline_rules"] == [{"id": "budget", "state": "budget >= 0"}]
        assert "guideline:budget" in [f["failure_type"] for f in shaped["baseline"]["coordination_findings"]]
        # --no-guidelines ignores the file; --guidelines points at an explicit one
        assert main(["run", f"{name}:build", "--no-guidelines", "--fail-on", "guideline", "-q", "--seed", "1"]) == 0
        (tmp_path / "other.yml").write_text("rules:\n  - id: steps\n    max_steps: 1\n", encoding="utf-8")
        assert main(["run", f"{name}:build", "--guidelines", "other.yml", "--fail-on", "guideline", "-q"]) == 0
        (tmp_path / "broken.yml").write_text("rules:\n  - max_steps: 1\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="no `id`"):
            main(["run", f"{name}:build", "--guidelines", "broken.yml"])
    finally:
        sys.modules.pop(name, None)
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))


def test_server_accepts_rules_yaml_and_rejects_a_bad_one():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from drift.server import create_app

    client = TestClient(create_app())
    body = {"graph_name": "ticket_triage", "intensity": "off", "judge": "off",
            "rules_yaml": "judge: [be kind]\nrules:\n  - id: steps\n    max_steps: 1\n"}
    r = client.post("/api/adapter-demo", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["guideline_rules"] == [{"id": "steps", "max_steps": 1}]
    assert data["n_user_guidelines"] == 1
    assert "guideline:steps" in [f["failure_type"] for f in data["baseline"]["coordination_findings"]]
    r = client.post("/api/adapter-demo", json={**body, "rules_yaml": "rules:\n  - id: x\n"})
    assert r.status_code == 400 and "exactly one of" in r.json()["detail"]
    r = client.post("/api/gate", json={"result": data, "rules": ["guideline"]})
    assert r.status_code == 200 and r.json()["passed"] is False
