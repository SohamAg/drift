"""Contract test: required_followups/policy_field reach drift_test_async
from /api/adapter-demo.

Mirrors test_server_coordination_roles.py's shape. Library-level detector
logic (premature_termination, policy_inconsistency_across_steps) already
has exhaustive unit coverage in test_library_detectors.py — this only
proves the server plumbing, using the real bundled ticket_triage graph
(no LLM needed).
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from drift.server import create_app


def test_adapter_demo_required_followups_triggers_premature_termination() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/api/adapter-demo",
        json={
            "graph_name": "ticket_triage",
            "intensity": "off",
            "max_perturbations": 1,
            "judge": "off",
            "required_followups": {"escalate": "human_review"},
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # ticket_triage's "escalate" node never hands off to a "human_review"
    # node, so this isn't just a plumbing check — it's a real fire.
    types = [f["failure_type"] for f in data["baseline"]["coordination_findings"]]
    assert "premature_termination" in types


def test_adapter_demo_accepts_policy_field_without_crashing() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/api/adapter-demo",
        json={
            "graph_name": "ticket_triage",
            "intensity": "off",
            "max_perturbations": 1,
            "judge": "off",
            "policy_field": "policy_version",
        },
    )
    assert resp.status_code == 200, resp.text
    # ticket_triage's state never carries a policy_version field, so no
    # finding is expected — this just confirms the field reaches the
    # detector without breaking request validation or the call.
    assert resp.json()["graph_name"] == "ticket_triage"


def test_custom_graph_run_request_model_accepts_the_new_fields() -> None:
    # custom-graph-run's real contract needs an importable module on
    # PYTHONPATH, which isn't worth setting up for a plumbing-only check —
    # this instead confirms CustomGraphRequest itself accepts the two new
    # fields (doesn't 422 on them), same as AdapterDemoRequest above.
    from drift.server import CustomGraphRequest

    req = CustomGraphRequest(
        module="nonexistent.module",
        required_followups={"escalate": "human_review"},
        policy_field="policy_version",
    )
    assert req.required_followups == {"escalate": "human_review"}
    assert req.policy_field == "policy_version"
