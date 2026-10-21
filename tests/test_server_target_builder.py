"""Tests for POST /api/propose-target, POST /api/build-target, and
POST /api/save-target.

/api/build-target and /api/propose-target are thin HTTP wrapping over
drift.target_builder (already exhaustively tested at the module level,
including propose_target's LLM-call mocking) — these tests cover the
HTTP wiring (status codes, request validation, error propagation) only.
/api/propose-target tests monkeypatch drift.target_builder.propose_target
directly so no real LLM call happens here either.
/api/save-target tests write into a monkeypatched USER_GENERATED_DIR
(tmp_path) so the real examples/targets/user_generated/ is never
touched by the test suite.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import drift.server as server_module
import drift.target_builder as target_builder_module
from drift.server import create_app
from drift.target_builder import ProposedTarget, ScenarioProposalError

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


def test_propose_target_returns_the_draft(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_propose(scenario, *, model):
        return ProposedTarget(
            topology="fanout",
            agents=[
                {"name": "dispatcher", "role": "split the task"},
                {"name": "a", "role": "angle a"},
                {"name": "b", "role": "angle b"},
                {"name": "merge", "role": "combine"},
            ],
            suggested_name="movie_success",
            suggested_query="how would a sci-fi movie do?",
            rationale="parallel angles fit fanout",
        )
    monkeypatch.setattr(target_builder_module, "propose_target", fake_propose)

    resp = client.post("/api/propose-target", json={"scenario": "how would a movie do"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["topology"] == "fanout"
    assert len(data["agents"]) == 4
    assert data["suggested_name"] == "movie_success"


def test_propose_target_surfaces_proposal_errors_as_400(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_propose(scenario, *, model):
        raise ScenarioProposalError("model did not return valid JSON")
    monkeypatch.setattr(target_builder_module, "propose_target", failing_propose)

    resp = client.post("/api/propose-target", json={"scenario": "anything"})
    assert resp.status_code == 400
    assert "valid JSON" in resp.json()["detail"]


def test_propose_target_requires_a_scenario_field(client: TestClient) -> None:
    resp = client.post("/api/propose-target", json={})
    assert resp.status_code == 422  # pydantic: scenario is required, no default
