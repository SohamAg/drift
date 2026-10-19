"""Server-level test for regression tracking on /api/adapter-demo.

Exercises the real endpoint (bundled ticket_triage graph, no LLM needed)
end-to-end rather than unit-testing drift.regression in isolation — proves
the wiring in _shape_adapter_result actually persists and returns a
regression block, not just that the module works standalone.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import drift.server as server_module
from drift.server import create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Redirect regression history to a tmp dir so the test suite never
    # writes into the repo's real results/_history/.
    monkeypatch.setattr(server_module, "HISTORY_DIR", tmp_path / "_history")
    return TestClient(create_app())


def _run(client: TestClient, track_history: bool) -> dict:
    resp = client.post(
        "/api/adapter-demo",
        json={
            "graph_name": "ticket_triage",
            "intensity": "light",
            "seed": 7,
            "max_perturbations": 3,
            "judge": "off",
            "divergence_mode": "exact",
            "track_history": track_history,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_regression_omitted_by_default(client: TestClient) -> None:
    data = _run(client, track_history=False)
    assert data["regression"] is None


def test_first_tracked_run_has_no_previous(client: TestClient) -> None:
    data = _run(client, track_history=True)
    assert data["regression"] is not None
    assert data["regression"]["has_previous"] is False


def test_second_tracked_run_compares_against_first(client: TestClient) -> None:
    _run(client, track_history=True)
    second = _run(client, track_history=True)
    assert second["regression"]["has_previous"] is True
    # Same graph, same seed, no judge — two runs of a deterministic bundled
    # demo graph shouldn't newly regress against themselves.
    assert second["regression"]["is_regression"] is False


def test_history_persists_to_the_redirected_dir(
    client: TestClient, tmp_path: Path
) -> None:
    _run(client, track_history=True)
    history_files = list((tmp_path / "_history").glob("*.jsonl"))
    assert len(history_files) == 1
    assert "ticket_triage" in history_files[0].name
