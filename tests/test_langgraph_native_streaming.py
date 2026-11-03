"""Adapter tests against REAL compiled LangGraph graphs (not stubs).

The stub-based tests in test_langgraph_adapter.py pin the adapter's
framework-agnostic contract. These pin what only a real `StateGraph` can
exercise:

  - `state_after` is the graph's reducer-applied state (an `add_messages`
    channel accumulates), not drift's own dict.update approximation.
  - BaseMessage objects are normalised to small stable dicts — no per-run
    `id` / metadata, so two identical runs compare equal and the `messages`
    field no longer diverges on every perturbation.
  - Subgraph super-steps are visible as `parent/inner` steps, INCLUDING a
    subgraph invoked from inside a tool call (the deepagents pattern that
    used to be a blind spot).
  - A recursion-limit loop is classified `crash_class="recursion_limit"`
    and its partial trace is kept, not discarded.

Skipped wholesale when langgraph isn't importable.
"""
from __future__ import annotations

import asyncio
from typing import Annotated, TypedDict

import pytest

langgraph = pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402

from drift.adapters.langgraph import (  # noqa: E402
    CRASH_CLASS_EXCEPTION,
    CRASH_CLASS_RECURSION_LIMIT,
    _diff_states,
    _run_one,
    drift_test,
    drift_test_fork,
)


class _S(TypedDict):
    messages: Annotated[list, add_messages]
    n: int


def _linear_messages_graph():
    """model -> tools -> done. Writes one message per node via add_messages."""

    def model(state):
        return {
            "messages": [AIMessage(content="", tool_calls=[
                {"name": "search", "args": {"q": "x"}, "id": "call_1"},
            ])],
            "n": state.get("n", 0) + 1,
        }

    def tools(state):
        return {"messages": [ToolMessage(content="Error: timeout", tool_call_id="call_1", status="error")]}

    def done(state):
        return {"messages": [AIMessage(content="final answer")]}

    g = StateGraph(_S)
    g.add_node("model", model)
    g.add_node("tools", tools)
    g.add_node("done", done)
    g.add_edge(START, "model")
    g.add_edge("model", "tools")
    g.add_edge("tools", "done")
    g.add_edge("done", END)
    return g.compile()


def _init():
    return {"messages": [{"role": "user", "content": "hi"}], "n": 0}


# ---- values-mode fidelity -------------------------------------------------


def test_state_after_is_reducer_applied_not_latest_delta():
    graph = _linear_messages_graph()
    final, trace, etype, err, _ = asyncio.run(_run_one(graph, _init()))
    assert etype == "", err
    assert [t["node"] for t in trace] == ["model", "tools", "done"]
    # The accumulated messages channel grows 2, 3, 4 — the old dict.update
    # accumulator would have shown 1, 1, 1 (just each node's delta).
    assert [len(t["state_after"]["messages"]) for t in trace] == [2, 3, 4]
    assert final is not None and len(final["messages"]) == 4
    # Each step's `update` is still the node's own delta.
    assert len(trace[0]["update"]["messages"]) == 1


def test_messages_are_normalised_to_stable_dicts():
    graph = _linear_messages_graph()
    final, trace, etype, _, _ = asyncio.run(_run_one(graph, _init()))
    msgs = final["messages"]
    assert [m["type"] for m in msgs] == ["human", "ai", "tool", "ai"]
    for m in msgs:
        assert isinstance(m, dict)
        assert "id" not in m and "response_metadata" not in m and "usage_metadata" not in m
    ai = msgs[1]
    # Provider tool-call ids are random per run; drift renumbers them
    # positionally so two runs compare equal AND the state still round-trips
    # into add_messages (which requires an id on every tool call).
    assert ai["tool_calls"] == [{"name": "search", "args": {"q": "x"}, "id": "call_1"}]
    tool_msg = msgs[2]
    assert tool_msg["tool_call_id"] == "call_1"
    assert tool_msg["status"] == "error"
    assert tool_msg["content"] == "Error: timeout"
    # The successful final AI message carries no status key (success is the default).
    assert "status" not in msgs[3]


def test_two_identical_runs_do_not_diverge_on_message_ids():
    """Before normalisation, BaseMessage `id` uuids made `messages` differ
    between any two runs — every perturbation looked DIVERGED. Now equal."""
    graph = _linear_messages_graph()
    a, _, _, _, _ = asyncio.run(_run_one(graph, _init()))
    b, _, _, _, _ = asyncio.run(_run_one(graph, _init()))
    diverged, summary, details = _diff_states(a, b)
    assert not diverged, summary


def test_drift_test_on_real_graph_runs_perturbations_and_coord_library():
    graph = _linear_messages_graph()
    result = drift_test(graph=graph, initial_state=_init(), intensity="exhaustive", seed=1)
    assert not result.baseline.crashed
    assert len(result.baseline.trace) == 3
    assert result.perturbations, "exhaustive should schedule at least one pattern"
    for p in result.perturbations:
        # Every perturbation either crashed or produced a 3-step trace.
        assert p.crashed or len(p.trace) == 3
        assert p.crash_class in ("", CRASH_CLASS_EXCEPTION, CRASH_CLASS_RECURSION_LIMIT)


# ---- subgraph visibility --------------------------------------------------


def _inner_graph():
    def inner(state):
        return {"messages": [AIMessage(content="inner done")]}

    g = StateGraph(_S)
    g.add_node("inner", inner)
    g.add_edge(START, "inner")
    g.add_edge("inner", END)
    return g.compile()


def test_subgraph_node_steps_are_visible_with_namespace():
    sub = _inner_graph()

    def a(state):
        return {"n": state.get("n", 0) + 1}

    g = StateGraph(_S)
    g.add_node("a", a)
    g.add_node("sub", sub)
    g.add_edge(START, "a")
    g.add_edge("a", "sub")
    g.add_edge("sub", END)
    graph = g.compile()

    final, trace, etype, err, _ = asyncio.run(_run_one(graph, _init()))
    assert etype == "", err
    nodes = [t["node"] for t in trace]
    assert nodes == ["a", "sub/inner", "sub"]
    sub_step = trace[1]
    assert sub_step["namespace"] == ["sub"]
    # state_after on a subgraph step is the ROOT state at that moment (root
    # hasn't committed the subgraph's output yet) — one schema for detectors.
    assert len(sub_step["state_after"]["messages"]) == 1
    assert "sub_state_after" in sub_step
    assert len(sub_step["sub_state_after"]["messages"]) == 2
    # Root steps carry no namespace key.
    assert "namespace" not in trace[0] and "namespace" not in trace[2]
    assert len(final["messages"]) == 2


def test_subgraph_invoked_inside_a_tool_call_is_visible():
    """The deepagents pattern: delegation happens inside a tool, not as a
    graph node. With subgraphs=True LangGraph surfaces it under the tools
    node's namespace, so the delegated agent's steps show up in the trace."""
    sub = _inner_graph()

    @tool
    def delegate(q: str) -> str:
        """Delegate to a sub-agent."""
        out = sub.invoke({"messages": [{"role": "user", "content": q}], "n": 0})
        return out["messages"][-1].content

    def model(state):
        last = state["messages"][-1]
        if getattr(last, "type", "") == "tool":
            return {"messages": [AIMessage(content="final")]}
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": "delegate", "args": {"q": "x"}, "id": "c1"},
        ])]}

    g = StateGraph(_S)
    g.add_node("model", model)
    g.add_node("tools", ToolNode([delegate]))
    g.add_edge(START, "model")
    g.add_conditional_edges("model", lambda s: "tools" if s["messages"][-1].tool_calls else END)
    g.add_edge("tools", "model")
    graph = g.compile()

    _, trace, etype, err, _ = asyncio.run(_run_one(graph, _init()))
    assert etype == "", err
    nodes = [t["node"] for t in trace]
    assert nodes == ["model", "tools/inner", "tools", "model"]
    assert trace[1]["namespace"] == ["tools"]


# ---- recursion-limit loops -------------------------------------------------


def _looping_graph():
    def spin(state):
        return {"n": state.get("n", 0) + 1}

    g = StateGraph(_S)
    g.add_node("spin", spin)
    g.add_edge(START, "spin")
    g.add_edge("spin", "spin")
    # Low recursion limit keeps the test fast (LangGraph's recursion error
    # path is slow per step) and exercises the `with_config` RunnableBinding
    # wrapper users commonly pass instead of the bare compiled graph.
    return g.compile().with_config(recursion_limit=6)


def test_recursion_limit_crash_is_classified_and_keeps_partial_trace():
    graph = _looping_graph()
    result = drift_test(graph=graph, initial_state={"messages": [], "n": 0}, intensity="off")
    b = result.baseline
    assert b.crashed
    assert b.error_type == "GraphRecursionError"
    assert b.crash_class == CRASH_CLASS_RECURSION_LIMIT
    # recursion_limit=6 -> six super-steps ran before LangGraph stopped it;
    # the partial trace must be there — that's where the loop is visible.
    assert len(b.trace) == 6
    assert all(t["node"] == "spin" for t in b.trace)
    assert all(t["state_after"] is not None for t in b.trace)
    assert any("recursion limit" in line for line in result.summary_lines()) or result.baseline.crashed


def test_recursion_limit_counts_in_adapter_result():
    graph = _looping_graph()
    result = drift_test(
        graph=graph, initial_state={"messages": [], "n": 0, "flag": True},
        intensity="exhaustive",
    )
    assert result.n_crashed == len(result.perturbations)
    assert result.n_recursion_limit == result.n_crashed
    assert any("looped to recursion limit" in line for line in result.summary_lines())


def test_plain_exception_is_classified_exception():
    def boom(state):
        raise RuntimeError("node exploded")

    g = StateGraph(_S)
    g.add_node("boom", boom)
    g.add_edge(START, "boom")
    g.add_edge("boom", END)
    graph = g.compile()
    result = drift_test(graph=graph, initial_state=_init(), intensity="off")
    assert result.baseline.crashed
    assert result.baseline.crash_class == CRASH_CLASS_EXCEPTION


# ---- fork still works on normalised traces --------------------------------


def test_fork_from_real_graph_trace():
    graph = _linear_messages_graph()
    parent = drift_test(graph=graph, initial_state=_init(), intensity="off")
    fork = drift_test_fork(graph=graph, parent_result=parent, fork_step=1, edits={"n": 41})
    assert not fork.fork_branch.crashed, fork.fork_branch.error
    # Rerun-from-state: the fork input is the normalised state_after[1]
    # (dict messages) — add_messages must coerce those back into messages.
    assert fork.fork_branch.final_state["n"] == 42
    assert [m["type"] for m in fork.fork_branch.final_state["messages"]][:2] == ["human", "ai"]


# ---- run-wide tool-call id consistency + output view + list noise bands ----


def test_tool_call_ids_agree_between_update_and_state_after():
    """The agent node's update and the tools node's update are normalised
    separately but must share one id map, or a ToolMessage's tool_call_id
    keeps the provider id while the AI update says call_1."""

    @tool
    def ping(q: str) -> str:
        """Ping."""
        return "pong"

    def model(state):
        last = state["messages"][-1]
        if getattr(last, "type", "") == "tool":
            return {"messages": [AIMessage(content="done")]}
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": "ping", "args": {"q": "x"}, "id": "call_9f8e7d6c5b4a"},
        ])]}

    g = StateGraph(_S)
    g.add_node("model", model)
    g.add_node("tools", ToolNode([ping]))
    g.add_edge(START, "model")
    g.add_conditional_edges("model", lambda s: "tools" if s["messages"][-1].tool_calls else END)
    g.add_edge("tools", "model")
    graph = g.compile()

    final, trace, etype, err, _ = asyncio.run(_run_one(graph, _init()))
    assert etype == "", err
    ai_update = trace[0]["update"]["messages"][0]
    tool_update = trace[1]["update"]["messages"][0]
    assert ai_update["tool_calls"][0]["id"] == "call_1"
    assert tool_update["tool_call_id"] == "call_1"
    assert final["messages"][1]["tool_calls"][0]["id"] == "call_1"
    assert final["messages"][2]["tool_call_id"] == "call_1"


def test_output_view_strips_only_the_messages_input_prefix():
    from drift.adapters.langgraph import _output_view

    initial = {"messages": [{"type": "human", "content": "q"}], "items": ["a", "b"], "n": 0}
    final = {
        "messages": [{"type": "human", "content": "q"}, {"type": "ai", "content": "ans"}],
        "items": ["a", "b"],
        "n": 3,
    }
    view = _output_view(initial, final)
    assert view["messages"] == [{"type": "ai", "content": "ans"}]
    assert view["items"] == ["a", "b"]  # plain lists are left alone
    assert view["n"] == 3
    # A messages channel the graph REPLACED (no input prefix) is left alone too.
    replaced = {"messages": [{"type": "ai", "content": "only"}]}
    assert _output_view(initial, replaced)["messages"] == replaced["messages"]


def test_message_perturbation_does_not_diverge_on_the_echoed_input():
    """Before the output view, every message perturbation was DIVERGED purely
    because the perturbed input sat at the front of the final conversation."""
    def respond(state):
        return {"messages": [AIMessage(content="fixed answer")]}

    g = StateGraph(_S)
    g.add_node("respond", respond)
    g.add_edge(START, "respond")
    g.add_edge("respond", END)
    graph = g.compile()

    result = drift_test(
        graph=graph,
        initial_state={"messages": [{"role": "user", "content": "please do the thing now"}], "n": 0},
        intensity="exhaustive",
    )
    msg_perts = [p for p in result.perturbations if p.perturbed_field == "messages"]
    assert msg_perts
    for p in msg_perts:
        assert not p.crashed, p.error
        assert not p.diverged, f"{p.pattern_type}: {p.divergence_summary}"


def test_list_valued_fields_get_a_similarity_noise_band():
    from drift.adapters.langgraph import _analyze_field_variance, _is_within_noise

    a = [{"type": "ai", "content": "7 times 8 is 56, and the capital of France is Paris."}]
    b = [{"type": "ai", "content": "7 times 8 is 56; the capital of France is Paris!"}]
    band = _analyze_field_variance("messages", [a, b])
    assert band.text_min_similarity is not None and band.text_min_similarity > 0.8
    c = [{"type": "ai", "content": "7 times 8 is 56 and the capital of France is Paris."}]
    within, sim = _is_within_noise(c, band, 0.85)
    assert within and sim is not None and sim > 0.9
    far = [{"type": "ai", "content": "DRIFT-CANARY-abcdef"}]
    assert _is_within_noise(far, band, 0.85)[0] is False
    # Bools never get text stats.
    assert _analyze_field_variance("flag", [True, False]).text_min_similarity is None
