"""Server-level test for POST /api/trace-ask — zero prior test coverage.

Backs the Ask tab. Uses judge="mock" (ScriptedMockJudge, no network) for
the happy-path plumbing tests — same philosophy as
test_server_coordination_roles.py: a contract test proving the request
reaches the judge and the response is shaped correctly, not a test of
answer quality (that depends on a real LLM and isn't this endpoint's
job to guarantee). Error-path tests (empty question, no data, judge=off)
need no judge call at all and are asserted precisely.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from drift.server import create_app


def test_ask_requires_a_question() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/api/trace-ask",
        json={"question": "   ", "summary": {"n_runs": 1}},
    )
    assert resp.status_code == 400


def test_ask_requires_trace_or_summary() -> None:
    client = TestClient(create_app())
    resp = client.post("/api/trace-ask", json={"question": "what happened?"})
    assert resp.status_code == 400


def test_ask_requires_a_real_judge() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/api/trace-ask",
        json={"question": "what happened?", "summary": {"n_runs": 1}, "judge": "off"},
    )
    assert resp.status_code == 400


def test_ask_with_summary_only_returns_an_answer() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/api/trace-ask",
        json={
            "question": "did anything crash?",
            "summary": {"n_runs": 3, "n_crashed": 0, "agents": ["a", "b"]},
            "judge": "mock",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["question"] == "did anything crash?"
    assert isinstance(data["answer"], str) and data["answer"]


def test_ask_with_full_trace_returns_an_answer() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/api/trace-ask",
        json={
            "question": "what did node a do?",
            "trace": [
                {"step": 1, "agent": "a", "update": {"note": "did something"}},
            ],
            "initial_state": {"task": "x"},
            "judge": "mock",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["question"] == "what did node a do?"
