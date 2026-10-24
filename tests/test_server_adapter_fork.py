"""Server-level test for /api/adapter-fork — zero prior test coverage.

Exercises the real chain: run /api/adapter-demo to get a genuine baseline
trace (no LLM needed, bundled ticket_triage), then fork at step 1 and
confirm the fork actually re-ran the graph and applied the edit. Doesn't
unit-test drift_test_fork_async in isolation (that's adapters/langgraph's
own job) — this proves the server's shim (AdapterResult/BaselineResult
wrapping of client-supplied parent data) and response shaping work on a
real, not hand-crafted, parent trace.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from drift.server import create_app


def _run_baseline(client: TestClient) -> dict:
    resp = client.post(
        "/api/adapter-demo",
        json={
            "graph_name": "ticket_triage",
            "intensity": "off",
            "max_perturbations": 1,
            "judge": "off",
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_fork_at_step_one_reruns_the_graph() -> None:
    client = TestClient(create_app())
    run = _run_baseline(client)
    trace = run["baseline"]["trace"]
    assert trace, "ticket_triage should stream at least one super-step"

    resp = client.post(
        "/api/adapter-fork",
        json={
            "graph_name": "ticket_triage",
            "parent_baseline_trace": trace,
            "parent_initial_state": run["baseline"]["initial_state"],
            "fork_step": 1,
            "edits": {},
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["fork_step"] == 1
    assert data["fork_branch"]["crashed"] is False
    assert data["fork_branch"]["final_state"] is not None
    assert data["top_edited_branch"] is None  # also_apply_at_initial defaulted off


def test_fork_also_apply_at_initial_returns_both_branches() -> None:
    client = TestClient(create_app())
    run = _run_baseline(client)
    trace = run["baseline"]["trace"]

    resp = client.post(
        "/api/adapter-fork",
        json={
            "graph_name": "ticket_triage",
            "parent_baseline_trace": trace,
            "parent_initial_state": run["baseline"]["initial_state"],
            "fork_step": 1,
            "edits": {},
            "also_apply_at_initial": True,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["top_edited_branch"] is not None
    assert data["top_edited_branch"]["crashed"] is False


def test_fork_step_out_of_range_is_a_400() -> None:
    client = TestClient(create_app())
    run = _run_baseline(client)
    trace = run["baseline"]["trace"]

    resp = client.post(
        "/api/adapter-fork",
        json={
            "graph_name": "ticket_triage",
            "parent_baseline_trace": trace,
            "parent_initial_state": run["baseline"]["initial_state"],
            "fork_step": len(trace) + 50,
            "edits": {},
        },
    )
    assert resp.status_code == 400
