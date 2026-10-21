"""Tests for POST /api/build-target and POST /api/save-target.

/api/build-target is pure codegen (drift.target_builder, already
exhaustively tested) exposed over HTTP — these tests cover the HTTP
wiring (success/error status codes, request validation) rather than
re-testing generation logic. /api/save-target tests write into a
monkeypatched USER_GENERATED_DIR (tmp_path) so the real
examples/targets/user_generated/ is never touched by the test suite.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import drift.server as server_module
from drift.server import create_app

PIPELINE_AGENTS = [
    {"name": "draft", "role": "Draft a response."},
    {"name": "finalize", "role": "Finalize the response."},
]


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(server_module, "USER_GENERATED_DIR", tmp_path / "user_generated")
    return TestClient(create_app())


def test_build_target_returns_generated_code(client: TestClient) -> None:
    resp = client.post("/api/build-target", json={
        "topology": "pipeline", "agents": PIPELINE_AGENTS,
    })
    assert resp.status_code == 200
    code = resp.json()["code"]
    assert "def build(" in code
    assert "def initial_state(" in code


def test_build_target_rejects_unknown_topology(client: TestClient) -> None:
    resp = client.post("/api/build-target", json={
        "topology": "not_a_topology", "agents": PIPELINE_AGENTS,
    })
    assert resp.status_code == 400
    assert "not_a_topology" in resp.json()["detail"]


def test_build_target_rejects_too_few_agents(client: TestClient) -> None:
    resp = client.post("/api/build-target", json={
        "topology": "pipeline", "agents": [{"name": "solo", "role": "x"}],
    })
    assert resp.status_code == 400


def test_build_target_requires_at_least_one_agent_entry(client: TestClient) -> None:
    resp = client.post("/api/build-target", json={"topology": "pipeline", "agents": []})
    assert resp.status_code == 422  # pydantic min_length violation


def test_save_target_writes_file_and_returns_module_path(
    client: TestClient, tmp_path: Path,
) -> None:
    code = "def build(model=None):\n    return None\n\n\ndef initial_state(query=''):\n    return {}\n"
    resp = client.post("/api/save-target", json={"name": "My Cool System!", "code": code})
    assert resp.status_code == 200
    data = resp.json()
    assert data["module"] == "targets.user_generated.my_cool_system"
    assert data["path"].endswith("my_cool_system.py")

    saved = tmp_path / "user_generated" / "my_cool_system.py"
    assert saved.exists()
    assert saved.read_text(encoding="utf-8") == code


def test_save_target_creates_init_file_if_missing(client: TestClient, tmp_path: Path) -> None:
    client.post("/api/save-target", json={"name": "x_system", "code": "x = 1\n"})
    assert (tmp_path / "user_generated" / "__init__.py").exists()


def test_save_target_rejects_bad_name(client: TestClient) -> None:
    resp = client.post("/api/save-target", json={"name": "!!!", "code": "x = 1\n"})
    assert resp.status_code == 400


def test_save_target_refuses_code_with_syntax_error(client: TestClient, tmp_path: Path) -> None:
    resp = client.post("/api/save-target", json={"name": "broken", "code": "def build(:\n"})
    assert resp.status_code == 400
    assert "syntax error" in resp.json()["detail"]
    assert not (tmp_path / "user_generated" / "broken.py").exists()


def test_save_target_accepts_hand_edited_code_not_from_build_target(
    client: TestClient, tmp_path: Path,
) -> None:
    """The preview is editable — code need not match what /api/build-target
    returned, as long as it parses."""
    edited = "# a human wrote this\ndef build(model=None):\n    return 'not a real graph'\n"
    resp = client.post("/api/save-target", json={"name": "hand_edited", "code": edited})
    assert resp.status_code == 200
    saved = (tmp_path / "user_generated" / "hand_edited.py").read_text(encoding="utf-8")
    assert saved == edited
