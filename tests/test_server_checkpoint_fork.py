"""Forks from the web UI take the real checkpoint path.

The fork endpoint rebuilds the graph per request. Before this, a graph
compiled without a checkpointer (every bundled demo, most user graphs)
could only ever be forked by rerun-from-state from the UI; the SDK path
had real LangGraph time travel since 11-04. Now the server attaches one
process-wide MemorySaver to compiled graphs that have none, the UI sends
`parent_thread_id`, and the fork resumes the parent's checkpoint.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from drift.server import create_app

pytest.importorskip("langgraph")

MODULE_SRC = textwrap.dedent('''
    from typing import TypedDict
    from langgraph.graph import END, START, StateGraph

    class S(TypedDict):
        query: str
        draft: str
        approved: bool
        final: str

    def draft(state):
        return {"draft": f"draft for {state['query']}"}

    def review(state):
        return {"approved": "urgent" not in state["draft"]}

    def publish(state):
        return {"final": state["draft"] if state["approved"] else "held for review"}

    def build(model=None):
        sg = StateGraph(S)
        sg.add_node("draft", draft)
        sg.add_node("review", review)
        sg.add_node("publish", publish)
        sg.add_edge(START, "draft")
        sg.add_edge("draft", "review")
        sg.add_edge("review", "publish")
        sg.add_edge("publish", END)
        return sg.compile()  # NO checkpointer

    def initial_state(query="release notes"):
        return {"query": query, "draft": "", "approved": False, "final": ""}
''')


@pytest.fixture
def module_name(tmp_path: Path) -> str:
    name = "drift_test_ckpt_graph"
    (tmp_path / f"{name}.py").write_text(MODULE_SRC, encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    sys.modules.pop(name, None)
    try:
        yield name
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(name, None)


def _run(client: TestClient, module_name: str) -> dict:
    resp = client.post("/api/custom-graph-run", json={
        "module": module_name, "build_fn": "build", "initial_state_fn": "initial_state",
        "intensity": "off", "judge": "off", "divergence_mode": "exact",
    })
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_custom_graph_runs_get_a_thread_id_from_the_server_saver(module_name: str) -> None:
    client = TestClient(create_app())
    run = _run(client, module_name)
    assert run["baseline"]["thread_id"].startswith("drift-"), run["baseline"]
    assert len(run["baseline"]["trace"]) == 3


def test_ui_fork_with_parent_thread_id_resumes_a_real_checkpoint(module_name: str) -> None:
    client = TestClient(create_app())
    run = _run(client, module_name)
    b = run["baseline"]
    # fork after "review" (step 2), flip the decision, let "publish" run
    resp = client.post("/api/adapter-fork", json={
        "module": module_name, "build_fn": "build", "initial_state_fn": "initial_state",
        "parent_baseline_trace": b["trace"],
        "parent_initial_state": b["initial_state"],
        "parent_thread_id": b["thread_id"],
        "fork_step": 2,
        "edits": {"approved": False},
    })
    assert resp.status_code == 200, resp.text
    fork = resp.json()
    br = fork["fork_branch"]
    assert br["execution_mode"] == "checkpoint", fork["execution_note"]
    assert "checkpoint fork" in fork["execution_note"]
    # only the continuation ran: one step (publish), and it saw the edit
    assert [s["node"] for s in br["trace"]] == ["publish"]
    assert br["final_state"]["final"] == "held for review"
    # LangGraph time travel stays on the parent's thread: update_state
    # writes a new checkpoint onto it and the resume continues from there.
    assert br["thread_id"] == b["thread_id"]


def test_ui_fork_without_parent_thread_id_falls_back_and_says_so(module_name: str) -> None:
    client = TestClient(create_app())
    run = _run(client, module_name)
    b = run["baseline"]
    resp = client.post("/api/adapter-fork", json={
        "module": module_name, "build_fn": "build", "initial_state_fn": "initial_state",
        "parent_baseline_trace": b["trace"],
        "parent_initial_state": b["initial_state"],
        "fork_step": 2,
        "edits": {"approved": False},
    })
    assert resp.status_code == 200, resp.text
    fork = resp.json()
    assert fork["fork_branch"]["execution_mode"] == "rerun_from_state"
    assert "no thread_id" in fork["execution_note"]
    # rerun-from-state replays every node from the edited state
    assert len(fork["fork_branch"]["trace"]) == 3


def test_stub_demo_graph_is_left_alone() -> None:
    """ticket_triage is a hand-rolled stub (no get_state_history); the
    server must not try to bolt a checkpointer onto it."""
    client = TestClient(create_app())
    resp = client.post("/api/adapter-demo", json={
        "graph_name": "ticket_triage", "intensity": "off", "judge": "off",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["baseline"]["thread_id"] == ""
