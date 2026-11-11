"""Tool-layer chaos — runtime faults on a graph's ToolNode tools."""
from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402

from drift import guidelines as gl  # noqa: E402
from drift.adapters.langgraph import drift_test  # noqa: E402
from drift.adapters.serialize import shape_adapter_result  # noqa: E402
from drift.chaos.engine import plan_auto_chaos  # noqa: E402
from drift.chaos.tools import (  # noqa: E402
    DEFAULT_AUTO_FAULTS,
    TOOL_FAULTS,
    ToolChaosError,
    ToolChaosEvent,
    discover_tool_patterns,
    find_tool_nodes,
    tool_names,
)

REAL = "the speed of light is 299792458 m/s according to the 2019 SI redefinition"


@tool
def web_search(q: str) -> str:
    """Search the web."""
    return REAL


@tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return "sent"


def _graph(behaviour="honest", n_calls=1, tools=(web_search,)):
    """A deterministic agent: makes `n_calls` web_search calls, then answers.

    behaviour:
      honest    — says "I could not find it" after an error/empty result
      silent    — always claims success
      obedient  — repeats any DRIFT-CANARY it sees in a tool result
      hijacked  — calls whatever tool a DRIFT-CANARY notice names, with the canary
    """
    def agent(state):
        msgs = state["messages"]
        tools_seen = [m for m in msgs if m.type == "tool"]
        last = tools_seen[-1] if tools_seen else None
        if behaviour == "obedient" and last is not None and "DRIFT-CANARY" in str(last.content):
            import re
            canary = re.search(r"DRIFT-CANARY-[0-9a-f]{6}", str(last.content)).group(0)
            return {"messages": [AIMessage(content=canary)]}
        if behaviour == "hijacked" and last is not None and "DRIFT-CANARY" in str(last.content) \
                and not any(getattr(m, "tool_calls", None) and "DRIFT-CANARY" in json.dumps(m.tool_calls) for m in msgs):
            import re
            canary = re.search(r"DRIFT-CANARY-[0-9a-f]{6}", str(last.content)).group(0)
            return {"messages": [AIMessage(content="", tool_calls=[
                {"name": "web_search", "args": {"q": canary}, "id": "hj"}])]}
        if len(tools_seen) < n_calls:
            return {"messages": [AIMessage(content="", tool_calls=[
                {"name": "web_search", "args": {"q": f"c{len(tools_seen)}"}, "id": f"id{len(tools_seen)}"}])]}
        if behaviour == "honest" and last is not None and (getattr(last, "status", "") == "error" or not last.content):
            return {"messages": [AIMessage(content="I could not find it; the search failed.")]}
        if behaviour == "silent":
            return {"messages": [AIMessage(content="The speed of light is 299792458 m/s.")]}
        return {"messages": [AIMessage(content=f"Answer: {str(last.content)[:40] if last else 'none'}")]}

    def route(state):
        return "tools" if state["messages"][-1].tool_calls else END

    sg = StateGraph(MessagesState)
    sg.add_node("agent", agent)
    sg.add_node("tools", ToolNode(list(tools)))
    sg.add_edge(START, "agent")
    sg.add_conditional_edges("agent", route, ["tools", END])
    sg.add_edge("tools", "agent")
    return sg.compile()


INIT = {"messages": [("user", "speed of light?")]}


def _tool_results(out):
    return [m for m in out["messages"] if m.type == "tool"]


# ---- discovery -----------------------------------------------------------------

def test_discovery_finds_tool_nodes_tools_and_subgraphs():
    g = _graph(tools=(web_search, send_email))
    assert [p for p, _ in find_tool_nodes(g)] == ["tools"]
    assert tool_names(g) == ["send_email", "web_search"]
    specs = discover_tool_patterns(g, seed=1)
    assert len(specs) == 2 * len(TOOL_FAULTS)
    assert "tool_timeout[web_search]" in {s.pattern_name for s in specs}
    assert all(s.fault_class == TOOL_FAULTS[s.pattern_type]["fault_class"] for s in specs)

    # a parent graph whose node IS a compiled sub-agent: tools found as parent/child
    inner = _graph()
    sg = StateGraph(MessagesState)
    sg.add_node("researcher", inner)
    sg.add_edge(START, "researcher")
    sg.add_edge("researcher", END)
    outer = sg.compile()
    assert [p for p, _ in find_tool_nodes(outer)] == ["researcher/tools"]
    assert tool_names(outer) == ["web_search"]

    # no ToolNode → nothing, rather than patterns that never fire
    class S(MessagesState):
        pass
    sg = StateGraph(S)
    sg.add_node("a", lambda s: {"messages": [AIMessage(content="x")]})
    sg.add_edge(START, "a")
    sg.add_edge("a", END)
    assert discover_tool_patterns(sg.compile()) == []
    assert discover_tool_patterns(lambda s: s) == []  # a plain callable graph


def test_unknown_fault_and_bad_on_call_are_rejected():
    with pytest.raises(ValueError, match="unknown tool fault"):
        ToolChaosEvent("web_search", fault="tool_meltdown")
    with pytest.raises(ValueError, match="on_call"):
        ToolChaosEvent("web_search", fault="tool_error", on_call=0)
    with pytest.raises(ValueError, match="unknown tool fault"):
        discover_tool_patterns(_graph(), faults=["nope"])


# ---- each fault -----------------------------------------------------------------

@pytest.mark.parametrize("fault, check", [
    ("tool_error", lambda m: m.status == "error" and "503" in m.content),
    ("tool_timeout", lambda m: m.status == "error" and "TimeoutError" in m.content),
    ("tool_rate_limit", lambda m: m.status == "error" and "429" in m.content),
    ("tool_empty", lambda m: m.content == "" and m.status != "error"),
    ("tool_truncated", lambda m: m.content.endswith("[truncated]") and len(m.content) < len(REAL) // 2
                                 and m.content.startswith(REAL[:5])),
    ("tool_stale", lambda m: m.content.startswith("[cached result from") and REAL in m.content),
    ("tool_inject_instruction", lambda m: REAL in m.content and "DRIFT-CANARY-" in m.content
                                          and "reply with exactly the phrase" in m.content),
    ("tool_hijack", lambda m: REAL in m.content and "DRIFT-CANARY-" in m.content and "`web_search`" in m.content),
])
def test_each_fault_rewrites_only_the_targeted_call(fault, check):
    g = _graph(n_calls=3)
    ev = ToolChaosEvent("web_search", fault=fault, on_call=2, seed=4)
    with ev.install(g):
        out = g.invoke(INIT)
    results = _tool_results(out)
    assert len(results) == 3
    assert results[0].content == REAL and results[2].content == REAL   # untouched
    assert check(results[1]), results[1]
    assert results[1].tool_call_id == "id1" and results[1].name == "web_search"
    assert ev.fired and ev.hits[0].call_index == 2 and ev.hits[0].node_path == "tools"
    assert ev.hits[0].args == {"q": "c1"}
    assert "fired on call #2" in ev.outcome_note()
    # restored after the block
    assert g.nodes["tools"].bound._wrap_tool_call is None
    assert g.nodes["tools"].bound._awrap_tool_call is None
    # the whole graph still ran to its own answer
    assert out["messages"][-1].content.startswith("Answer:")


def test_tool_exception_raises_through_the_default_handler():
    g = _graph()
    ev = ToolChaosEvent("web_search", fault="tool_exception", seed=1)
    with ev.install(g), pytest.raises(ToolChaosError):
        g.invoke(INIT)
    assert ev.fired


def test_async_path_and_miss_are_reported():
    g = _graph(n_calls=2)
    ev = ToolChaosEvent("web_search", fault="tool_rate_limit", on_call=1, seed=1)
    with ev.install(g):
        out = asyncio.run(g.ainvoke(INIT))
    r = _tool_results(out)
    assert r[0].status == "error" and r[1].content == REAL
    assert ev.fired

    miss = ToolChaosEvent("web_search", fault="tool_error", on_call=5, seed=1)
    with miss.install(g):
        g.invoke(INIT)
    assert not miss.fired and "never made call #5" in miss.outcome_note()

    absent = ToolChaosEvent("send_email", fault="tool_error", seed=1)
    with absent.install(g):
        g.invoke(INIT)
    assert absent.installed_on == [] and "no ToolNode offers this tool" in absent.outcome_note()


def test_previous_wrapper_is_kept_for_other_calls_and_restored():
    g = _graph(n_calls=2)
    seen: list[str] = []

    def user_wrapper(req, execute):
        seen.append(req.tool_call["id"])
        return execute(req)

    node = g.nodes["tools"].bound
    node._wrap_tool_call = user_wrapper
    try:
        ev = ToolChaosEvent("web_search", fault="tool_empty", on_call=2, seed=1)
        with ev.install(g):
            out = g.invoke(INIT)
        assert seen == ["id0"]                     # the un-faulted call went through the user's wrapper
        assert _tool_results(out)[1].content == ""
        assert node._wrap_tool_call is user_wrapper
    finally:
        node._wrap_tool_call = None


# ---- planner + adapter integration ----------------------------------------------

def test_planner_takes_an_extra_catalogue():
    from drift.world import WorldState

    class S(WorldState):
        approved: bool = False

    extra = discover_tool_patterns(_graph(), seed=1)
    plan = plan_auto_chaos(state=S(), steps=30, intensity="exhaustive", seed=1, extra_catalog=extra)
    names = {e.name for _, e in plan}
    assert "AutoChaos.tool_error[web_search]" in names and "AutoChaos.flip_bool[approved]" in names
    plan = plan_auto_chaos(state=S(), steps=30, intensity="exhaustive", seed=1, extra_catalog=extra,
                           exclude=["tool_"])
    assert all("tool_" not in e.name for _, e in plan)


def test_drift_test_schedules_tool_faults_and_reports_layer_and_fired():
    g = _graph(behaviour="honest")
    res = drift_test(graph=g, initial_state=INIT, intensity="exhaustive", seed=1, divergence_mode="off")
    tool_perts = [p for p in res.perturbations if p.fault_layer == "tool"]
    assert {p.pattern_type for p in tool_perts} == set(TOOL_FAULTS)
    assert all(p.fault_fired is True and p.perturbed_field == "web_search" for p in tool_perts)
    assert all(p.perturbed_initial_state == res.baseline.initial_state for p in tool_perts)  # state untouched
    assert all(p.fault_layer == "state" and p.fault_fired is None for p in res.perturbations if p not in tool_perts)
    exc = next(p for p in tool_perts if p.pattern_type == "tool_exception")
    assert exc.crashed and exc.error_type == "ToolChaosError"
    assert "fired on call #1 (tools)" in exc.event_summary
    assert res.patterns_total == 8 + len(TOOL_FAULTS)   # 1 clear_list + 7 message patterns + tool faults

    shaped = shape_adapter_result(res, graph_meta={"name": "t", "description": "", "agents": []},
                                  seed=1, judge="off", judge_model=None, user_guidelines=None)
    rows = {p["event_name"]: p for p in shaped["perturbations"]}
    assert rows["AutoChaos.tool_timeout[web_search]"]["fault_layer"] == "tool"
    assert rows["AutoChaos.tool_timeout[web_search]"]["fault_fired"] is True
    assert rows["AutoChaos.drop_last_user_message[messages]"]["fault_fired"] is None


def test_auto_plan_draws_from_the_curated_faults_only_below_exhaustive():
    g = _graph(tools=(web_search, send_email))
    res = drift_test(graph=g, initial_state=INIT, intensity="aggressive", seed=3, max_perturbations=40,
                     divergence_mode="off")
    tool_types = {p.pattern_type for p in res.perturbations if p.fault_layer == "tool"}
    assert tool_types and tool_types <= set(DEFAULT_AUTO_FAULTS)
    off = drift_test(graph=g, initial_state=INIT, intensity="exhaustive", seed=3, tool_chaos=False,
                     divergence_mode="off")
    assert not [p for p in off.perturbations if p.fault_layer == "tool"]
    assert off.patterns_total == 8


def test_runtime_faults_feed_the_existing_detectors():
    # the agent repeats the canary it saw in the poisoned (real) tool result
    res = drift_test(graph=_graph(behaviour="obedient"), initial_state=INIT, intensity="off", seed=1,
                     rules=gl.parse({"chaos": [{"tool": "web_search", "fault": "tool_inject_instruction"}]}))
    [p] = res.perturbations
    assert "injected_instruction_followed" in [f["failure_type"] for f in p.coordination_findings]

    # the agent calls the named tool with the canary → action-layer detector
    res = drift_test(graph=_graph(behaviour="hijacked"), initial_state=INIT, intensity="off", seed=1,
                     rules=gl.parse({"chaos": [{"tool": "web_search", "fault": "tool_hijack"}]}))
    [p] = res.perturbations
    assert "injected_tool_call_followed" in [f["failure_type"] for f in p.coordination_findings]

    # a silent agent papers over the runtime error → silent_tool_failure; an honest one does not
    res = drift_test(graph=_graph(behaviour="silent"), initial_state=INIT, intensity="off", seed=1,
                     rules=gl.parse({"chaos": [{"tool": "web_search", "fault": "tool_timeout"}]}))
    [p] = res.perturbations
    assert "silent_tool_failure" in [f["failure_type"] for f in p.coordination_findings]
    res = drift_test(graph=_graph(behaviour="honest"), initial_state=INIT, intensity="off", seed=1,
                     rules=gl.parse({"chaos": [{"tool": "web_search", "fault": "tool_timeout"}]}))
    [p] = res.perturbations
    assert "silent_tool_failure" not in [f["failure_type"] for f in p.coordination_findings]


def test_declared_chaos_from_drift_yml_runs_first_and_outside_the_ceiling():
    g = _graph(n_calls=2)
    spec = gl.parse({"chaos": [
        {"tool": "web_search", "fault": "tool_timeout", "on_call": 2},
        {"pattern": "drop_last_user_message"},
        {"field": "messages", "pattern": "clear_list"},
    ]})
    res = drift_test(graph=g, initial_state=INIT, intensity="moderate", seed=1, max_perturbations=1,
                     rules=spec, divergence_mode="off")
    names = [p.event_name for p in res.perturbations]
    assert names[:3] == ["AutoChaos.tool_timeout[web_search]", "AutoChaos.drop_last_user_message[messages]",
                         "AutoChaos.clear_list[messages]"]
    # + at most one from the plan (max_perturbations=1); a planned event that
    # duplicates a declared one is dropped, never run twice
    assert 3 <= len(names) <= 4 and len(set(names)) == len(names)
    assert "fired on call #2" in res.perturbations[0].event_summary
    # intensity off + declared chaos → only the declared ones
    res = drift_test(graph=g, initial_state=INIT, intensity="off", rules=spec, divergence_mode="off")
    assert len(res.perturbations) == 3


@pytest.mark.parametrize("entry, msg", [
    ({"tool": "nope", "fault": "tool_error"}, "not reachable"),
    ({"tool": "web_search", "fault": "tool_meltdown"}, "unknown tool fault"),
    ({"tool": "web_search", "fault": "tool_error", "on_call": 0}, "on_call"),
    ({"pattern": "flip_bool", "field": "messages"}, "no pattern"),
    ({"field": "messages"}, "needs"),
    ({"tool": "web_search", "fault": "tool_error", "extra": 1}, "unknown key"),
])
def test_declared_chaos_typos_are_errors(entry, msg):
    with pytest.raises(ValueError, match=msg):
        drift_test(graph=_graph(), initial_state=INIT, intensity="off", rules=gl.parse({"chaos": [entry]}))


def test_cli_and_server_surfaces(tmp_path, monkeypatch, capsys):
    import sys
    import textwrap

    from drift.cli import main

    name = "drift_tool_chaos_cli_graph"
    (tmp_path / f"{name}.py").write_text(textwrap.dedent('''
        from langchain_core.messages import AIMessage
        from langchain_core.tools import tool
        from langgraph.graph import END, START, MessagesState, StateGraph
        from langgraph.prebuilt import ToolNode

        @tool
        def lookup(q: str) -> str:
            """lookup"""
            return "42"

        def agent(state):
            if not any(m.type == "tool" for m in state["messages"]):
                return {"messages": [AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"q": "x"}, "id": "a"}])]}
            return {"messages": [AIMessage(content="The answer is 42.")]}

        def build(model=None):
            sg = StateGraph(MessagesState)
            sg.add_node("agent", agent)
            sg.add_node("tools", ToolNode([lookup]))
            sg.add_edge(START, "agent")
            sg.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls else END, ["tools", END])
            sg.add_edge("tools", "agent")
            return sg.compile()

        def initial_state(query="hello"):
            return {"messages": [("user", query)]}
    '''), encoding="utf-8")
    (tmp_path / "drift.yml").write_text(
        "chaos:\n  - tool: lookup\n    fault: tool_timeout\nrules:\n  - id: ack\n"
        "    final_answer: {must_match: 'fail|could not|unable'}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    sys.modules.pop(name, None)
    try:
        out = tmp_path / "o.json"
        code = main(["run", f"{name}:build", "--intensity", "off", "--json", str(out), "--fail-on", "guideline"])
        text = capsys.readouterr().out
        assert code == 1
        assert "AutoChaos.tool_timeout[lookup]  (tool fault)" in text
        assert "guideline:ack" in text
        shaped = json.loads(out.read_text(encoding="utf-8"))
        assert shaped["perturbations"][0]["fault_layer"] == "tool" and shaped["perturbations"][0]["fault_fired"] is True
        code = main(["run", f"{name}:build", "--intensity", "exhaustive", "--no-tool-chaos", "--no-guidelines",
                     "--json", str(out), "-q"])
        assert code == 0
        shaped = json.loads(out.read_text(encoding="utf-8"))
        assert not [p for p in shaped["perturbations"] if p["fault_layer"] == "tool"]
    finally:
        sys.modules.pop(name, None)
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from drift.server import create_app

    client = TestClient(create_app())
    r = client.post("/api/adapter-demo", json={"graph_name": "ticket_triage", "intensity": "light",
                                               "judge": "off", "tool_chaos": False})
    assert r.status_code == 200
    assert all(p["fault_layer"] == "state" for p in r.json()["perturbations"])
    r = client.post("/api/adapter-demo", json={"graph_name": "ticket_triage", "intensity": "off", "judge": "off",
                                               "rules_yaml": "chaos:\n  - tool: nope\n    fault: tool_error\n"})
    assert r.status_code == 400 and "not reachable" in r.json()["detail"]
