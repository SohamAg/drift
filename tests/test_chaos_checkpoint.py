"""Model-layer and control-flow chaos (drift.chaos.checkpoint), end to end on
real compiled LangGraph graphs with a scripted chat model — no API key."""
from __future__ import annotations

from typing import TypedDict

import pytest

pytest.importorskip("langgraph")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

from drift.adapters.langgraph import drift_test  # noqa: E402
from drift.chaos.checkpoint import (  # noqa: E402
    MODEL_FAULTS,
    CheckpointChaosEvent,
    discover_flow_patterns,
    discover_model_patterns,
    skippable_nodes,
)

CALLS: list[dict] = []


@tool
def web_search(query: str) -> str:
    """Search the web."""
    CALLS.append({"query": query})
    return f"result for {query}"


class _ScriptedModel(GenericFakeChatModel):
    """Calls web_search once, then answers with whatever tool results it saw."""

    def bind_tools(self, tools, **kw):  # noqa: D401 — fake
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        from langchain_core.outputs import ChatGeneration, ChatResult

        tool_msgs = [m for m in messages if m.type == "tool"]
        if not tool_msgs:
            msg = AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "capital of france"},
                                                     "id": "call_1"}])
        else:
            msg = AIMessage(content="Answer based on: " + " | ".join(str(m.content)[:60] for m in tool_msgs))
        return ChatResult(generations=[ChatGeneration(message=msg)])


try:  # LangGraph v1's agent lives in langchain >= 1.0
    from langchain.agents import create_agent as _create_agent
except ImportError:  # pragma: no cover - older stacks
    _create_agent = None
AGENT_NODE = "model" if _create_agent else "agent"
needs_create_agent = pytest.mark.skipif(_create_agent is None, reason="langchain>=1.0 not installed")


def _react():
    """The LangGraph v1 agent (langchain.agents.create_agent: nodes model +
    tools); the legacy prebuilt where langchain 1.x is not installed."""
    if _create_agent is None:
        return _react_legacy()
    return _create_agent(_ScriptedModel(messages=iter([])), [web_search])


def _legacy_prebuilt():
    import warnings

    from langgraph.prebuilt import create_react_agent

    def build(*a, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return create_react_agent(*a, **kw)
    return build


def _react_legacy():
    """The pre-v1 prebuilt (nodes agent + tools). Deprecated, still what
    langgraph-supervisor builds its workers with, so drift keeps supporting it."""
    return _legacy_prebuilt()(_ScriptedModel(messages=iter([])), [web_search])


def _react_state():
    return {"messages": [HumanMessage("What is the capital of France?")]}


class Ticket(TypedDict):
    ticket: str
    priority: str
    reply: str


def _pipeline():
    def triage(state):
        return {"priority": "high" if "outage" in state["ticket"] else "normal"}

    def reply(state):
        hours = {"normal": 24, "high": 4}[state["priority"]]
        return {"reply": f"within {hours}h"}

    sg = StateGraph(Ticket)
    sg.add_node("triage", triage)
    sg.add_node("reply", reply)
    sg.add_edge(START, "triage")
    sg.add_edge("triage", "reply")
    sg.add_edge("reply", END)
    return sg.compile()


def _by_pattern(result, pattern):
    return [p for p in result.perturbations if p.pattern_type == pattern]


# ---- catalogue -------------------------------------------------------------------------

def test_catalogue_names_and_skippable_nodes():
    g = _react()
    assert sorted(sp.pattern_name for sp in discover_model_patterns(g)) == sorted(
        f"{f}[web_search]" for f in MODEL_FAULTS)
    # the ToolNode is never a skip target
    assert skippable_nodes(g) == [AGENT_NODE]
    assert skippable_nodes(_react_legacy()) == ["agent"]
    assert [sp.pattern_name for sp in discover_flow_patterns(_pipeline())] == ["skip_node[triage]", "skip_node[reply]"]


def test_unknown_fault_is_rejected():
    with pytest.raises(ValueError, match="unknown checkpoint fault"):
        CheckpointChaosEvent("web_search", fault="model_explode")


# ---- model layer -----------------------------------------------------------------------

def _run_model_fault(fault):
    CALLS.clear()
    return drift_test(graph=_react(), initial_state=_react_state(), intensity="off",
                      rules=_spec([{"tool": "web_search", "fault": fault}]))


def _spec(chaos):
    from drift.guidelines import GuidelineSpec

    return GuidelineSpec(chaos=chaos)


def test_bad_tool_args_reach_the_tool_node_for_real():
    res = _run_model_fault("model_bad_tool_args")
    (p,) = res.perturbations
    assert p.fault_layer == "model" and p.fault_fired is True and p.chaos_timing == "mid_run"
    assert "dropped all arguments" in p.event_summary
    texts = [str(m["content"]) for m in p.final_state["messages"]]
    assert any("Field required" in t or "Error invoking tool" in t for t in texts), texts
    assert not p.crashed


def test_unknown_tool_gets_the_tool_nodes_own_error():
    (p,) = _run_model_fault("model_unknown_tool").perturbations
    texts = [str(m["content"]) for m in p.final_state["messages"]]
    assert any("web_search_v2 is not a valid tool" in t for t in texts), texts


def test_duplicate_tool_call_executes_the_tool_twice():
    res = _run_model_fault("model_duplicate_tool_call")
    (p,) = res.perturbations
    # baseline: 1 call; the resumed fork: 2 more (the original and the duplicate)
    assert len(CALLS) == 3, CALLS
    tool_msgs = [m for m in p.final_state["messages"] if m["type"] == "tool"]
    assert len(tool_msgs) == 2


def test_dropped_tool_call_ends_the_run_with_nothing():
    (p,) = _run_model_fault("model_dropped_tool_call").perturbations
    assert p.diverged
    assert not [m for m in p.final_state["messages"] if m["type"] == "tool"]


def test_model_chaos_flag_puts_model_faults_in_the_sampled_plan():
    res = drift_test(graph=_react(), initial_state=_react_state(), intensity="exhaustive",
                     model_chaos=True, tool_chaos=False, max_perturbations=100)
    layers = {p.fault_layer for p in res.perturbations}
    assert "model" in layers
    assert {p.pattern_type for p in res.perturbations if p.fault_layer == "model"} == set(MODEL_FAULTS)
    off = drift_test(graph=_react(), initial_state=_react_state(), intensity="exhaustive",
                     tool_chaos=False, max_perturbations=100)
    assert "model" not in {p.fault_layer for p in off.perturbations}


# ---- control flow ----------------------------------------------------------------------

def test_skip_node_exposes_an_unchecked_handoff():
    res = drift_test(graph=_pipeline(), initial_state={"ticket": "login slow", "priority": "", "reply": ""},
                     intensity="off", rules=_spec([{"skip_node": "triage"}]))
    (p,) = res.perturbations
    assert p.fault_layer == "flow" and p.fault_fired
    # reply trusted triage to have set priority — skipping triage crashes it
    assert p.crashed and p.error_type == "KeyError"
    assert "skipped 'triage'" in p.event_summary


def test_skip_last_node_diverges_without_crashing():
    res = drift_test(graph=_pipeline(), initial_state={"ticket": "outage!", "priority": "", "reply": ""},
                     intensity="off", rules=_spec([{"skip_node": "reply"}]))
    (p,) = res.perturbations
    assert not p.crashed and p.diverged
    assert p.final_state["reply"] == ""


def test_declared_skip_of_a_tool_node_is_refused():
    with pytest.raises(ValueError, match="ToolNodes cannot be skipped"):
        drift_test(graph=_react(), initial_state=_react_state(), intensity="off",
                   rules=_spec([{"skip_node": "tools"}]))


def test_fault_with_nowhere_to_land_is_reported_not_run():
    # a react run whose baseline never calls the tool: the model answers directly
    class _Direct(_ScriptedModel):
        def _generate(self, messages, stop=None, run_manager=None, **kw):
            from langchain_core.outputs import ChatGeneration, ChatResult
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Paris."))])

    g = (_create_agent or _legacy_prebuilt())(_Direct(messages=iter([])), [web_search])
    res = drift_test(graph=g, initial_state=_react_state(), intensity="off",
                     rules=_spec([{"tool": "web_search", "fault": "model_bad_tool_args"}]))
    (p,) = res.perturbations
    assert p.fault_fired is False and not p.crashed and not p.diverged
    assert "never had a pending call" in p.event_summary


# ---- surfaces --------------------------------------------------------------------------

def test_server_and_cli_expose_model_and_flow_chaos(capsys):
    from drift.cli import main

    code = main(["run", "examples.ci.demo_app", "--flow-chaos", "--intensity", "exhaustive",
                 "--max-perturbations", "20", "--fail-on", "crash", "-q"])
    out = capsys.readouterr().out
    assert code == 1 and "AutoChaos.skip_node[triage]" in out

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from drift.server import create_app

    client = TestClient(create_app())
    # the bundled ticket_triage demo is a hand-rolled stub (no checkpoints):
    # use the compiled CI demo through the custom-graph endpoint
    r = client.post("/api/custom-graph-run", json={"module": "examples.ci.demo_app", "intensity": "exhaustive",
                                                   "judge": "off", "flow_chaos": True, "max_perturbations": 100})
    assert r.status_code == 200, r.text
    layers = {p["fault_layer"] for p in r.json()["perturbations"]}
    assert "flow" in layers


@pytest.mark.parametrize("build", [pytest.param(_react, marks=needs_create_agent), _react_legacy],
                         ids=["create_agent", "create_react_agent"])
@pytest.mark.parametrize("fault", sorted(MODEL_FAULTS))
def test_every_model_fault_fires_on_both_agent_apis(build, fault):
    res = drift_test(graph=build(), initial_state=_react_state(), intensity="off",
                     rules=_spec([{"tool": "web_search", "fault": fault}]))
    (p,) = res.perturbations
    assert p.fault_fired is True and not p.crashed, p.event_summary
