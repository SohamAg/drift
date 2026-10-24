"""Server-level test for POST /api/report — zero prior test coverage.

Backs the "Evidence pack" button in the Results browser and Overview
tabs. Uses the real adapter-demo -> save-run -> report chain against the
bundled ticket_triage graph (no LLM needed), same pattern as
test_server_overview.py, rather than hand-crafting a run JSON.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import drift.server as server_module
from drift.server import create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(server_module, "RESULTS_DIR", tmp_path / "results")
    return TestClient(create_app())


def _save_a_run(client: TestClient) -> str:
    resp = client.post(
        "/api/adapter-demo",
        json={
            "graph_name": "ticket_triage",
            "intensity": "light",
            "seed": 7,
            "max_perturbations": 2,
            "judge": "off",
        },
    )
    assert resp.status_code == 200, resp.text
    run = resp.json()

    resp = client.post(
        "/api/save-run",
        json={"data": run, "group": "report_check", "name": "r.json"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["path"]


def test_report_renders_a_self_contained_html_page(client: TestClient) -> None:
    rel_path = _save_a_run(client)
    resp = client.post("/api/report", json={"path": rel_path})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "<html" in data["html"].lower()
    assert data["filename"].endswith(".report.html")
    assert "n_runs" in data["summary"]


def test_report_honors_custom_title(client: TestClient) -> None:
    rel_path = _save_a_run(client)
    resp = client.post(
        "/api/report", json={"path": rel_path, "title": "My Custom Evidence Pack"},
    )
    assert resp.status_code == 200, resp.text
    assert "My Custom Evidence Pack" in resp.json()["html"]


def test_report_404s_on_missing_file(client: TestClient) -> None:
    resp = client.post("/api/report", json={"path": "nope/does_not_exist.json"})
    assert resp.status_code == 404


def test_report_rejects_path_traversal(client: TestClient) -> None:
    resp = client.post("/api/report", json={"path": "../../etc/passwd"})
    assert resp.status_code in (400, 403, 404)


def test_report_400s_on_non_drift_run_json(
    client: TestClient, tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "junk.json").write_text('{"hello": "world"}', encoding="utf-8")

    resp = client.post("/api/report", json={"path": "junk.json"})
    assert resp.status_code == 400
