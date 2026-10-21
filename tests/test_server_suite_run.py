"""Tests for GET /api/targets and POST /api/suite-run.

Exercises against the real examples/targets/ registry rather than mocking
it — these endpoints are thin wiring over drift.adapters.langgraph and
the registry, both already covered elsewhere; what needs proving here is
that the wiring itself (registry listing shape, per-target skip/error
branching, one-bad-target-doesn't-stop-the-rest) behaves correctly.

No test here exercises the "ok" (successful run) branch: every target
currently in the registry needs a real OpenAI call even at intensity=off
(the baseline still invokes the graph once), and this suite — like the
rest of the project's tests — never makes real paid API calls. The ok
branch was verified manually against supervisor_basic during
development (one real run, n_crashed=0, result shaped correctly); the
skip/error paths below cover the branching logic itself.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from drift.server import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_list_targets_returns_the_real_registry() -> None:
    resp = _client().get("/api/targets")
    assert resp.status_code == 200
    targets = resp.json()["targets"]
    names = {t["name"] for t in targets}
    assert "supervisor_basic" in names
    assert "tradingagents_debate" in names
    for t in targets:
        assert t["module"].startswith("targets.")
        assert "importable" in t
        assert "missing_deps" in t


def test_suite_run_skips_a_target_with_missing_deps() -> None:
    resp = _client().post("/api/suite-run", json={
        "target_names": ["tradingagents_debate"],
        "intensity": "off",
    })
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 1
    assert results[0]["status"] == "skipped"
    assert "tradingagents" in results[0]["error"]
    assert results[0]["result"] is None


def test_suite_run_reports_unknown_target_as_error_not_crash() -> None:
    resp = _client().post("/api/suite-run", json={
        "target_names": ["this_target_does_not_exist"],
        "intensity": "off",
    })
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results[0]["status"] == "error"
    assert "this_target_does_not_exist" in results[0]["error"]


def test_suite_run_one_bad_target_does_not_stop_the_rest() -> None:
    resp = _client().post("/api/suite-run", json={
        "target_names": ["nonexistent_a", "tradingagents_debate", "nonexistent_b"],
        "intensity": "off",
    })
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 3
    assert [r["target_name"] for r in results] == [
        "nonexistent_a", "tradingagents_debate", "nonexistent_b",
    ]
    assert results[0]["status"] == "error"
    assert results[1]["status"] == "skipped"
    assert results[2]["status"] == "error"


def test_suite_run_requires_at_least_one_target() -> None:
    resp = _client().post("/api/suite-run", json={"target_names": []})
    assert resp.status_code == 422  # pydantic min_length violation
