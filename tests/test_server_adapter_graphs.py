"""Server-level test for GET /api/adapter-graphs — zero prior test coverage
despite backing the Test tab's graph picker since early in the project.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from drift.server import create_app


def test_lists_bundled_graphs_with_required_fields() -> None:
    client = TestClient(create_app())
    resp = client.get("/api/adapter-graphs")
    assert resp.status_code == 200
    graphs = resp.json()["graphs"]
    names = {g["name"] for g in graphs}
    assert "ticket_triage" in names
    assert "langgraph_supervisor" in names

    required = {
        "name", "label", "description", "agents", "available",
        "unavailable_reason", "supports_query_override",
        "query_field_label", "query_default", "needs_openai",
    }
    for g in graphs:
        assert required <= g.keys(), f"{g['name']} missing fields: {required - g.keys()}"


def test_ticket_triage_is_always_available() -> None:
    client = TestClient(create_app())
    resp = client.get("/api/adapter-graphs")
    graphs = {g["name"]: g for g in resp.json()["graphs"]}
    assert graphs["ticket_triage"]["available"] is True
    assert graphs["ticket_triage"]["needs_openai"] is False
