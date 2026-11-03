"""Checkpointed LangGraph graphs: per-run thread ids and true checkpoint forks.

Before this, a graph compiled with a checkpointer crashed drift_test at the
baseline ("Checkpointer requires ... thread_id"), and if the user had wired a
fixed thread_id into the graph every perturbation would have run on top of
the previous one's state. Now every run gets its own thread, and
`drift_test_fork` resumes a real checkpoint when it can.

Skipped without langgraph.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402

from drift.adapters.langgraph import (  # noqa: E402
    FORK_MODE_CHECKPOINT,
    FORK_MODE_RERUN,
    drift_test,
    drift_test_fork,
)

S = TypedDict("S", {"messages": Annotated[list, add_messages], "n": int})


def _three_node_graph(checkpointer=None):
    def a(s):
        return {"messages": [AIMessage(content="a")], "n": s.get("n", 0) + 1}

    def b(s):
        return {"messages": [AIMessage(content="b")], "n": s["n"] + 10}

    def c(s):
        return {"messages": [AIMessage(content=f"c n={s['n']}")], "n": s["n"] + 100}

    g = StateGraph(S)
    for name, fn in (("a", a), ("b", b), ("c", c)):
        g.add_node(name, fn)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", END)
    return g.compile(checkpointer=checkpointer)


INIT = {"messages": [("user", "hi")], "n": 0}


def test_checkpointed_graph_runs_with_a_fresh_thread_per_run():
    graph = _three_node_graph(MemorySaver())
    result = drift_test(graph=graph, initial_state=INIT, intensity="exhaustive")
    assert not result.baseline.crashed, result.baseline.error
    assert result.baseline.thread_id.startswith("drift-")
    assert result.baseline.final_state["n"] == 111
    assert result.perturbations
    tids = {p.thread_id for p in result.perturbations}
    assert len(tids) == len(result.perturbations) and result.baseline.thread_id not in tids
    # No cross-run contamination: the untouched-field perturbations still end at the same n.
    for p in result.perturbations:
        if p.perturbed_field != "n" and not p.crashed:
            assert p.final_state["n"] == 111, p.event_name


def test_unchanged_graph_without_checkpointer_has_no_thread_id():
    result = drift_test(graph=_three_node_graph(), initial_state=INIT, intensity="off")
    assert result.baseline.thread_id == ""


def test_fork_resumes_a_real_checkpoint_when_available():
    graph = _three_node_graph(MemorySaver())
    parent = drift_test(graph=graph, initial_state=INIT, intensity="off")
    fork = drift_test_fork(graph=graph, parent_result=parent, fork_step=1, edits={"n": 41})
    br = fork.fork_branch
    assert not br.crashed, br.error
    assert br.execution_mode == FORK_MODE_CHECKPOINT
    assert fork.execution_note.startswith("checkpoint fork: resumed root step 1")
    assert br.thread_id == parent.baseline.thread_id
    # Only the CONTINUATION ran: b then c — a is not re-executed.
    assert [t["node"] for t in br.trace] == ["b", "c"]
    assert br.final_state["n"] == 41 + 10 + 100
    # The edited state at the fork is the checkpoint's values after update_state.
    assert fork.edited_state_at_fork["n"] == 41
    assert [m["content"] for m in br.final_state["messages"]] == ["hi", "a", "b", "c n=51"]


def test_fork_edits_to_a_reducer_channel_go_through_the_reducer():
    graph = _three_node_graph(MemorySaver())
    parent = drift_test(graph=graph, initial_state=INIT, intensity="off")
    fork = drift_test_fork(
        graph=graph, parent_result=parent, fork_step=2,
        edits={"messages": [("user", "an injected follow-up")]},
    )
    assert fork.fork_branch.execution_mode == FORK_MODE_CHECKPOINT
    contents = [m["content"] for m in fork.fork_branch.final_state["messages"]]
    # add_messages APPENDED the edit rather than replacing the channel.
    assert contents == ["hi", "a", "b", "an injected follow-up", "c n=11"]


def test_fork_falls_back_without_checkpointer():
    graph = _three_node_graph()
    parent = drift_test(graph=graph, initial_state=INIT, intensity="off")
    fork = drift_test_fork(graph=graph, parent_result=parent, fork_step=1, edits={"n": 41})
    assert fork.fork_branch.execution_mode == FORK_MODE_RERUN
    assert "no checkpointer" in fork.execution_note
    # Rerun-from-state re-executes from START: a runs again on the edited state.
    assert [t["node"] for t in fork.fork_branch.trace] == ["a", "b", "c"]
    assert fork.fork_branch.final_state["n"] == 41 + 1 + 10 + 100


def test_fork_falls_back_when_history_is_in_a_different_graph_object():
    """The web server re-imports the graph per request: a fresh MemorySaver
    has no history for the parent's thread. Must degrade, not crash."""
    g1 = _three_node_graph(MemorySaver())
    parent = drift_test(graph=g1, initial_state=INIT, intensity="off")
    g2 = _three_node_graph(MemorySaver())
    fork = drift_test_fork(graph=g2, parent_result=parent, fork_step=1, edits={"n": 41})
    assert fork.fork_branch.execution_mode == FORK_MODE_RERUN
    assert "no checkpoint for root step 1" in fork.execution_note
    assert not fork.fork_branch.crashed
    assert fork.fork_branch.thread_id.startswith("drift-fork-")


def test_fork_falls_back_for_a_subgraph_step():
    inner = _three_node_graph()

    def pre(s):
        return {"n": s.get("n", 0) + 1000}

    g = StateGraph(S)
    g.add_node("pre", pre)
    g.add_node("inner", inner)
    g.add_edge(START, "pre")
    g.add_edge("pre", "inner")
    g.add_edge("inner", END)
    graph = g.compile(checkpointer=MemorySaver())

    parent = drift_test(graph=graph, initial_state=INIT, intensity="off")
    nodes = [t["node"] for t in parent.baseline.trace]
    assert nodes == ["pre", "inner/a", "inner/b", "inner/c", "inner"]
    fork = drift_test_fork(graph=graph, parent_result=parent, fork_step=3, edits={"n": 5})
    assert fork.fork_branch.execution_mode == FORK_MODE_RERUN
    assert "inside a subgraph" in fork.execution_note
    # ...while a root step on the same graph forks for real.
    fork_root = drift_test_fork(graph=graph, parent_result=parent, fork_step=1, edits={"n": 5})
    assert fork_root.fork_branch.execution_mode == FORK_MODE_CHECKPOINT
    assert fork_root.fork_branch.final_state["n"] == 5 + 1 + 10 + 100


def test_top_edited_branch_runs_on_its_own_thread():
    graph = _three_node_graph(MemorySaver())
    parent = drift_test(graph=graph, initial_state=INIT, intensity="off")
    fork = drift_test_fork(
        graph=graph, parent_result=parent, fork_step=1, edits={"n": 41}, also_apply_at_initial=True,
    )
    top = fork.top_edited_branch
    assert top is not None and not top.crashed
    assert top.execution_mode == FORK_MODE_RERUN and top.thread_id.startswith("drift-fork-top-")
    assert top.final_state["n"] == 41 + 1 + 10 + 100
    # The checkpoint branch skipped node `a` (already executed before the fork
    # point); the top-edited branch ran it again. Same edit, different outcome:
    # exactly the path-dependence the compare exists to surface.
    assert fork.fork_branch.final_state["n"] == 41 + 10 + 100
    pd = fork.path_dependence
    assert pd["is_path_dependent"] is True
    assert "n" in {d["name"] for d in pd["diverged_fields"]}
