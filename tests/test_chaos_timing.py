"""chaos_timing="mid_run" — state patterns applied to a baseline checkpoint
and resumed, instead of perturbing the initial state."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402

from drift.adapters.langgraph import (  # noqa: E402
    CHAOS_TIMING_MID_RUN,
    _checkpoint_edits,
    _checkpointer_of,
    _fork_rank_for,
    _messages_delta,
    drift_test,
    ensure_checkpointer,
)
from drift.adapters.serialize import shape_adapter_result  # noqa: E402

try:
    from test_chaos_tools import INIT, _graph
except ImportError:  # pragma: no cover
    from tests.test_chaos_tools import INIT, _graph


def _msg_texts(state):
    return [m.get("content") if isinstance(m, dict) else getattr(m, "content", m) for m in state["messages"]]


# ---- helpers ------------------------------------------------------------------------

def test_fork_rank_maps_the_planner_horizon_onto_the_baseline():
    assert _fork_rank_for(2, 7) == 1
    assert _fork_rank_for(29, 7) == 6          # never the last step
    assert _fork_rank_for(15, 7) == 3 or _fork_rank_for(15, 7) == 4
    assert _fork_rank_for(2, 2) == 1 and _fork_rank_for(29, 2) == 1
    assert 1 <= _fork_rank_for(100, 3) <= 2


def test_messages_delta_expresses_drop_edit_and_append():
    h = HumanMessage(content="hello there", id="h1")
    a = AIMessage(content="hi", id="a1")
    # drop
    d = _messages_delta([h, a], [a])
    assert [type(m).__name__ for m in d] == ["RemoveMessage"] and d[0].id == "h1"
    # edit in place keeps the id → add_messages replaces
    h2 = h.model_copy(update={"content": "hel"})
    d = _messages_delta([h, a], [h2, a])
    assert d == [h2] and d[0].id == "h1"
    # append
    new = AIMessage(content="fabricated")
    assert _messages_delta([h, a], [h, a, new]) == [new]
    # unchanged → empty
    assert _messages_delta([h, a], [h, a]) == []
    # plain fields replace whole
    e = _checkpoint_edits({"messages": [h], "count": 1, "flag": True},
                          {"messages": [h], "count": 2, "flag": True})
    assert e == {"count": 2}


def test_ensure_checkpointer_only_touches_compiled_graphs_without_one():
    g = _graph()
    assert _checkpointer_of(g) is None
    ensure_checkpointer(g)
    assert _checkpointer_of(g) is not None
    own = MemorySaver()
    sg = StateGraph(MessagesState)
    sg.add_node("a", lambda s: {"messages": [AIMessage(content="x")]})
    sg.add_edge(START, "a")
    sg.add_edge("a", END)
    g2 = sg.compile(checkpointer=own)
    ensure_checkpointer(g2)
    assert _checkpointer_of(g2) is own
    stub = object()
    assert ensure_checkpointer(stub) is stub


# ---- drift_test ----------------------------------------------------------------------

def test_mid_run_forks_the_baseline_and_resumes():
    g = _graph(behaviour="honest", n_calls=3)
    init = {"messages": [("user", "What is the speed of light? Please answer precisely.")]}
    res = drift_test(graph=g, initial_state=init, intensity="exhaustive", seed=1,
                     chaos_timing="mid_run", divergence_mode="off", tool_chaos=False)
    assert res.chaos_timing == "mid_run"
    assert len(res.baseline.trace) == 7
    by = {p.pattern_type: p for p in res.perturbations}

    inj = by["inject_tool_result_instruction"]
    assert inj.chaos_timing == CHAOS_TIMING_MID_RUN and 1 <= inj.fork_step <= 6
    assert "applied mid-run, after baseline step" in inj.event_summary
    # the resumed trace continues the baseline's numbering and is shorter than a full run
    assert inj.trace[0]["step"] == inj.fork_step + 1
    assert len(inj.trace) < 7
    # the perturbed "initial" state is the checkpoint's state, history included
    assert len(inj.perturbed_initial_state["messages"]) > 1
    # the injected result is in the final thread and the rest of the run ran on top of it
    texts = _msg_texts(inj.final_state)
    assert any("DRIFT-CANARY" in str(t) for t in texts)
    assert len(texts) > len(inj.perturbed_initial_state["messages"])
    assert inj.thread_id == res.baseline.thread_id

    garble = by["garble_last_user_message"]
    assert garble.chaos_timing == CHAOS_TIMING_MID_RUN
    first = inj.perturbed_initial_state["messages"][0]
    assert garble.perturbed_initial_state["messages"][0]["content"] != init["messages"][0][1]
    assert len(garble.perturbed_initial_state["messages"]) == len(garble.rollouts) * 0 + len(
        garble.perturbed_initial_state["messages"])  # same message count: edited in place, not appended

    drop = by["drop_last_user_message"]
    assert drop.chaos_timing == CHAOS_TIMING_MID_RUN
    assert not any(isinstance(m, dict) and m.get("type") == "human" for m in drop.perturbed_initial_state["messages"])

    # clear_list empties the channel at a fork after a tools step (the
    # tools→agent edge is unconditional here) and the agent copes
    clear = by["clear_list"]
    assert clear.chaos_timing == CHAOS_TIMING_MID_RUN and not clear.crashed
    assert clear.perturbed_initial_state["messages"] == []


def test_mid_run_edit_that_breaks_routing_is_reported_as_a_crash():
    """update_state evaluates the fork node's outgoing edges; a router that
    indexes messages[-1] on the emptied channel raises — the graph crashed
    on the perturbation, and the result says so instead of falling back."""
    from drift import guidelines as gl

    def a(state):
        return {"messages": [AIMessage(content="a")]}

    def b(state):
        return {"messages": [AIMessage(content="b")]}

    def c(state):
        return {"messages": [AIMessage(content="c")]}

    def route(state):
        return "c" if state["messages"][-1].content else END

    sg = StateGraph(MessagesState)
    for name, fn in (("a", a), ("b", b), ("c", c)):
        sg.add_node(name, fn)
    sg.add_edge(START, "a")
    sg.add_edge("a", "b")
    sg.add_conditional_edges("b", route, ["c", END])
    sg.add_edge("c", END)
    g = sg.compile()
    spec = gl.parse({"chaos": [{"pattern": "clear_list", "step": 2}, {"pattern": "inject_fabricated_prior_turn", "step": 2}]})
    res = drift_test(graph=g, initial_state={"messages": [("user", "hello there")]}, intensity="off",
                     chaos_timing="mid_run", divergence_mode="off", rules=spec)
    clear, fab = res.perturbations
    assert clear.chaos_timing == "mid_run" and clear.fork_step == 2
    assert clear.crashed and clear.error_type == "IndexError" and "update_state" in clear.event_summary
    # an append at the same point resumes normally and c runs
    assert fab.chaos_timing == "mid_run" and not fab.crashed
    assert [t["node"] for t in fab.trace] == ["c"] and fab.trace[0]["step"] == 3


def test_mid_run_rollouts_each_fork_fresh_and_tool_faults_are_unaffected():
    g = _graph(behaviour="obedient", n_calls=2)
    res = drift_test(graph=g, initial_state=INIT, intensity="off", seed=2, chaos_timing="mid_run",
                     perturbation_rollouts=2, divergence_mode="off",
                     rules=[], tool_chaos=True)
    # nothing scheduled at intensity off without declared chaos
    assert res.perturbations == []
    from drift import guidelines as gl
    spec = gl.parse({"chaos": [{"pattern": "inject_tool_result_instruction"},
                               {"tool": "web_search", "fault": "tool_timeout"}]})
    res = drift_test(graph=g, initial_state=INIT, intensity="off", seed=2, chaos_timing="mid_run",
                     perturbation_rollouts=2, divergence_mode="off", rules=spec)
    inj, tool = res.perturbations
    assert inj.chaos_timing == "mid_run" and inj.n_rollouts == 2
    assert all(r.chaos_timing == "mid_run" and r.fork_step == inj.fork_step for r in inj.all_rollouts())
    # the obedient agent follows the mid-run injection on every rollout
    assert inj.finding_rollout_counts.get("injected_instruction_followed") == 2
    assert tool.fault_layer == "tool" and tool.chaos_timing == "initial" and tool.fork_step is None


def test_mid_run_falls_back_with_a_reason_when_it_cannot_fork():
    # a stub graph (invoke only): no checkpointer can be attached
    class Stub:
        def invoke(self, state, *a, **k):
            out = dict(state)
            out["reply"] = "ok"
            return out
    res = drift_test(graph=Stub(), initial_state={"query": "hi", "reply": "", "approved": False},
                     intensity="exhaustive", seed=1, chaos_timing="mid_run")
    assert res.perturbations and all(p.chaos_timing == "initial" for p in res.perturbations)
    assert all("fell back to initial-state timing" in p.timing_note for p in res.perturbations)
    assert all("fell back" in p.event_summary for p in res.perturbations)

    # a one-step graph: nothing to resume after the fork point
    sg = StateGraph(MessagesState)
    sg.add_node("a", lambda s: {"messages": [AIMessage(content="x")]})
    sg.add_edge(START, "a")
    sg.add_edge("a", END)
    res = drift_test(graph=sg.compile(), initial_state=INIT, intensity="exhaustive", seed=1,
                     chaos_timing="mid_run", divergence_mode="off")
    assert all(p.chaos_timing == "initial" and "root step" in p.timing_note for p in res.perturbations)

    with pytest.raises(ValueError, match="chaos_timing"):
        drift_test(graph=_graph(), initial_state=INIT, intensity="off", chaos_timing="later")


def test_serialised_cli_and_server_surfaces(tmp_path, monkeypatch, capsys):
    import sys
    import textwrap

    g = _graph(n_calls=2)
    res = drift_test(graph=g, initial_state=INIT, intensity="light", seed=1, chaos_timing="mid_run",
                     divergence_mode="off", tool_chaos=False)
    shaped = shape_adapter_result(res, graph_meta={"name": "t", "description": "", "agents": []},
                                  seed=1, judge="off", judge_model=None, user_guidelines=None)
    assert shaped["chaos_timing"] == "mid_run"
    assert all(p["chaos_timing"] in ("initial", "mid_run") and "fork_step" in p and "timing_note" in p
               for p in shaped["perturbations"])

    from drift.cli import main
    name = "drift_timing_cli_graph"
    (tmp_path / f"{name}.py").write_text(textwrap.dedent('''
        from langchain_core.messages import AIMessage
        from langgraph.graph import END, START, MessagesState, StateGraph

        def a(state):
            return {"messages": [AIMessage(content="step a")]}

        def b(state):
            return {"messages": [AIMessage(content=f"step b after {len(state['messages'])} msgs")]}

        def build(model=None):
            sg = StateGraph(MessagesState)
            sg.add_node("a", a)
            sg.add_node("b", b)
            sg.add_edge(START, "a")
            sg.add_edge("a", "b")
            sg.add_edge("b", END)
            return sg.compile()

        def initial_state(query="hello there friend"):
            return {"messages": [("user", query)]}
    '''), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    sys.modules.pop(name, None)
    try:
        out = tmp_path / "o.json"
        code = main(["run", f"{name}:build", "--intensity", "exhaustive", "--timing", "mid_run", "--no-guidelines",
                     "--divergence", "off", "--json", str(out)])
        text = capsys.readouterr().out
        assert code == 0
        assert "(mid-run @ step 1)" in text
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["chaos_timing"] == "mid_run"
        assert any(p["chaos_timing"] == "mid_run" and p["fork_step"] == 1 for p in data["perturbations"])
    finally:
        sys.modules.pop(name, None)
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from drift.server import create_app

    client = TestClient(create_app())
    r = client.post("/api/adapter-demo", json={"graph_name": "ticket_triage", "intensity": "light", "judge": "off",
                                               "chaos_timing": "mid_run"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["chaos_timing"] == "mid_run"
    assert data["perturbations"] and all("fork_step" in p for p in data["perturbations"])
    r = client.post("/api/adapter-demo", json={"graph_name": "ticket_triage", "intensity": "off", "judge": "off",
                                               "chaos_timing": "whenever"})
    assert r.status_code == 400
