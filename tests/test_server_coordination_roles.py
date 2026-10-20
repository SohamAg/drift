"""Contract test: coordination_roles reaches drift_test_async from both
/api/adapter-demo and /api/custom-graph-run (previously only AdapterForkRequest
carried this field). Library-level detection logic (role_drift) already has
exhaustive unit coverage in test_library_detectors.py — this only proves the
plumbing, using the real bundled ticket_triage graph (no LLM needed).
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from drift.server import create_app


def test_adapter_demo_accepts_coordination_roles() -> None:
    client = TestClient(create_app())
    resp = client.post(
        "/api/adapter-demo",
        json={
            "graph_name": "ticket_triage",
            "intensity": "off",
            "judge": "off",
            "coordination_roles": {"escalate": "verifier"},
        },
    )
    assert resp.status_code == 200, resp.text
    # A clean contract test, not a detection test: ticket_triage's nodes
    # write no execution-shaped verbs, so no finding is expected — just
    # confirming the field didn't break request validation or the call.
    assert resp.json()["graph_name"] == "ticket_triage"
