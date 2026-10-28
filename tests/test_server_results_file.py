"""Server-level test for GET/DELETE /api/results/{relpath} — the single
mutating endpoint in an otherwise read-only results-browsing surface.

GET had no dedicated test before (only indirect coverage via
test_server_overview.py's click-through check). DELETE is new.
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


def _save_a_run(client: TestClient, group: str = "del_check", name: str = "r.json") -> str:
    resp = client.post(
        "/api/adapter-demo",
        json={
            "graph_name": "ticket_triage", "intensity": "off",
            "max_perturbations": 1, "judge": "off",
        },
    )
    assert resp.status_code == 200, resp.text
    resp = client.post(
        "/api/save-run",
        json={"data": resp.json(), "group": group, "name": name},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["path"]


def test_get_returns_the_saved_json(client: TestClient) -> None:
    rel_path = _save_a_run(client)
    resp = client.get(f"/api/results/{rel_path}")
    assert resp.status_code == 200
    assert resp.json()["graph_name"] == "ticket_triage"


def test_get_404s_on_missing_file(client: TestClient) -> None:
    resp = client.get("/api/results/nope.json")
    assert resp.status_code == 404


def test_get_rejects_path_traversal(client: TestClient) -> None:
    resp = client.get("/api/results/../../etc/passwd")
    assert resp.status_code in (400, 403, 404)


def test_get_rejects_non_json_extension(
    client: TestClient, tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "notes.txt").write_text("hello", encoding="utf-8")
    resp = client.get("/api/results/notes.txt")
    assert resp.status_code == 400


def test_delete_removes_the_file(client: TestClient, tmp_path: Path) -> None:
    rel_path = _save_a_run(client)
    full = (tmp_path / "results" / rel_path)
    assert full.exists()

    resp = client.delete(f"/api/results/{rel_path}")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"deleted": True, "path": rel_path}
    assert not full.exists()


def test_delete_is_reflected_in_the_results_index(client: TestClient) -> None:
    rel_path = _save_a_run(client)
    resp = client.get("/api/results")
    assert any(
        e["path"] == rel_path for entries in resp.json()["groups"].values() for e in entries
    )

    client.delete(f"/api/results/{rel_path}")

    resp = client.get("/api/results")
    assert not any(
        e["path"] == rel_path for entries in resp.json()["groups"].values() for e in entries
    )


def test_delete_404s_on_missing_file(client: TestClient) -> None:
    resp = client.delete("/api/results/nope.json")
    assert resp.status_code == 404


def test_delete_is_not_idempotent_second_call_404s(client: TestClient) -> None:
    rel_path = _save_a_run(client)
    first = client.delete(f"/api/results/{rel_path}")
    assert first.status_code == 200
    second = client.delete(f"/api/results/{rel_path}")
    assert second.status_code == 404


def test_delete_rejects_path_traversal(client: TestClient) -> None:
    resp = client.delete("/api/results/../../etc/passwd")
    assert resp.status_code in (400, 403, 404)


def test_delete_rejects_non_json_extension(
    client: TestClient, tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    target = results_dir / "notes.txt"
    target.write_text("hello", encoding="utf-8")
    resp = client.delete("/api/results/notes.txt")
    assert resp.status_code == 400
    assert target.exists()  # rejected, not deleted
