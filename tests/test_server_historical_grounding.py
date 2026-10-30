"""Tests for GET /api/historical-events, POST /api/simulate-scenario, and
POST /api/calibrate.

Each underlying piece (target_builder.propose_target, historical_grounding
.retrieve_similar_events, calibration.run_calibration) already has
thorough unit coverage with LLM calls mocked at their own layer. These
tests cover the HTTP wiring: request validation, how the pieces compose,
and error propagation — mocking propose_target/retrieve_similar_events
directly (same pattern as test_server_target_builder.py) and
_load_generated_module (server.py's own helper) so no real LangGraph
graph is built or invoked, matching the rest of this test suite's
"never make real paid API calls" convention.

Real end-to-end proof (propose -> ground -> generate -> run -> verdict,
and a full calibration run) was done manually during development: a
real scenario ("a major chip export ban affecting an AI chipmaker") ran
3 times via TestClient with no 500s, each producing a plausible
grounded verdict citing the real SNAP/META precedents; a real
calibration run against all 7 seed events with a cheap 2-agent roster
scored 7/7 direction matches.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import drift.calibration as calibration_module
import drift.historical_grounding as grounding_module
import drift.server as server_module
import drift.target_builder as target_builder_module
from drift.calibration import CalibrationReport, CalibrationResult
from drift.historical_grounding import HistoricalEvent
from drift.server import create_app
from drift.target_builder import ProposedTarget


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


SAMPLE_EVENT = HistoricalEvent(
    "NVDA", "2023-05-24", "2023-05-25", "sample event", 100.0, 124.0, 24.0,
)


class _FakeGraph:
    def __init__(self, verdict: dict):
        self._verdict = verdict

    async def ainvoke(self, state):
        return {**state, "verdict": self._verdict, "notes": {"a": "note"}}


class _FakeModule:
    def __init__(self, verdict: dict):
        self._verdict = verdict

    def build(self, model=None):
        return _FakeGraph(self._verdict)

    def initial_state(self, query):
        return {"task": query, "notes": {}}


def test_list_historical_events_returns_the_real_dataset(client: TestClient) -> None:
    resp = client.get("/api/historical-events")
    assert resp.status_code == 200
    data = resp.json()
    events = data["events"]
    assert len(events) >= 5
    tickers = {e["ticker"] for e in events}
    assert "META" in tickers


def test_list_historical_events_includes_the_default_fleet(client: TestClient) -> None:
    from drift.historical_grounding import (
        DEFAULT_STOCK_REACTION_AGENTS,
        DEFAULT_STOCK_REACTION_TOPOLOGY,
    )

    data = client.get("/api/historical-events").json()
    assert data["default_topology"] == DEFAULT_STOCK_REACTION_TOPOLOGY
    assert data["default_agents"] == list(DEFAULT_STOCK_REACTION_AGENTS)
    assert all({"name", "role"} <= a.keys() for a in data["default_agents"])


def test_simulate_scenario_end_to_end_wiring(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_propose(scenario, *, model):
        return ProposedTarget(
            topology="fanout",
            agents=[
                {"name": "dispatcher", "role": "split"},
                {"name": "a", "role": "angle a"},
                {"name": "b", "role": "angle b"},
                {"name": "merge", "role": "synth"},
            ],
            rationale="fanout fits",
        )
    monkeypatch.setattr(target_builder_module, "propose_target", fake_propose)

    async def fake_retrieve(scenario, events=None, *, k=3, model="gpt-4o-mini"):
        return [SAMPLE_EVENT]
    monkeypatch.setattr(grounding_module, "retrieve_similar_events", fake_retrieve)

    monkeypatch.setattr(
        server_module, "_load_generated_module",
        lambda code, label: _FakeModule({"direction": "up", "confidence": "high"}),
    )

    resp = client.post("/api/simulate-scenario", json={"scenario": "something happens"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["topology"] == "fanout"
    assert data["verdict"] == {"direction": "up", "confidence": "high"}
    assert data["grounded_on"][0]["ticker"] == "NVDA"
    assert "code" in data


def test_simulate_scenario_with_user_configured_fleet_skips_proposal(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the caller supplies both topology and agents (the fleet-setup
    override), propose_target is never called — the configured fleet runs
    exactly as given, and the response echoes it back with no rationale."""
    def fail_if_called(*a, **k):
        raise AssertionError("propose_target should not be called")
    monkeypatch.setattr(target_builder_module, "propose_target", fail_if_called)

    async def fake_retrieve(scenario, events=None, *, k=3, model="gpt-4o-mini"):
        return [SAMPLE_EVENT]
    monkeypatch.setattr(grounding_module, "retrieve_similar_events", fake_retrieve)

    monkeypatch.setattr(
        server_module, "_load_generated_module",
        lambda code, label: _FakeModule({"direction": "flat"}),
    )

    custom_agents = [
        {"name": "macro_analyst", "role": "assess macro conditions"},
        {"name": "verdict_synth", "role": "synthesize a verdict"},
    ]
    resp = client.post("/api/simulate-scenario", json={
        "scenario": "something happens",
        "topology": "pipeline",
        "agents": custom_agents,
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["topology"] == "pipeline"
    assert data["agents"] == custom_agents
    assert data["rationale"] == ""


def test_simulate_scenario_passes_k_to_retrieval(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_propose(scenario, *, model):
        return ProposedTarget(topology="pipeline", agents=[
            {"name": "a", "role": "x"}, {"name": "b", "role": "y"},
        ])
    monkeypatch.setattr(target_builder_module, "propose_target", fake_propose)

    captured = {}

    async def fake_retrieve(scenario, events=None, *, k=3, model="gpt-4o-mini"):
        captured["k"] = k
        return [SAMPLE_EVENT]
    monkeypatch.setattr(grounding_module, "retrieve_similar_events", fake_retrieve)

    monkeypatch.setattr(
        server_module, "_load_generated_module",
        lambda code, label: _FakeModule({"direction": "flat"}),
    )

    resp = client.post("/api/simulate-scenario", json={"scenario": "x", "k": 7})
    assert resp.status_code == 200, resp.text
    assert captured["k"] == 7


def test_simulate_scenario_without_grounding_skips_retrieval(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_propose(scenario, *, model):
        return ProposedTarget(topology="pipeline", agents=[
            {"name": "a", "role": "x"}, {"name": "b", "role": "y"},
        ])
    monkeypatch.setattr(target_builder_module, "propose_target", fake_propose)

    def fail_if_called(*a, **k):
        raise AssertionError("retrieve_similar_events should not be called")
    monkeypatch.setattr(grounding_module, "retrieve_similar_events", fail_if_called)

    monkeypatch.setattr(
        server_module, "_load_generated_module",
        lambda code, label: _FakeModule({"direction": "flat"}),
    )

    resp = client.post("/api/simulate-scenario", json={"scenario": "x", "ground": False})
    assert resp.status_code == 200
    assert resp.json()["grounded_on"] == []


def test_simulate_scenario_propagates_proposal_errors_as_400(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drift.target_builder import ScenarioProposalError

    async def failing_propose(scenario, *, model):
        raise ScenarioProposalError("model did not return valid JSON")
    monkeypatch.setattr(target_builder_module, "propose_target", failing_propose)

    resp = client.post("/api/simulate-scenario", json={"scenario": "x"})
    assert resp.status_code == 400


def test_simulate_scenario_surfaces_run_failures_as_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_propose(scenario, *, model):
        return ProposedTarget(topology="pipeline", agents=[
            {"name": "a", "role": "x"}, {"name": "b", "role": "y"},
        ])
    monkeypatch.setattr(target_builder_module, "propose_target", fake_propose)
    monkeypatch.setattr(grounding_module, "retrieve_similar_events", lambda *a, **k: _async([]))

    def raise_on_load(code, label):
        raise RuntimeError("boom")
    monkeypatch.setattr(server_module, "_load_generated_module", raise_on_load)

    resp = client.post("/api/simulate-scenario", json={"scenario": "x"})
    assert resp.status_code == 502


def _async(value):
    async def _inner():
        return value
    return _inner()


def test_calibrate_uses_default_roster_when_agents_omitted(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_generate(topology, agents, *, model, verdict_schema):
        captured["topology"] = topology
        captured["agents"] = agents
        return "# generated"
    monkeypatch.setattr(target_builder_module, "generate_target_source", fake_generate)
    monkeypatch.setattr(
        server_module, "_load_generated_module",
        lambda code, label: _FakeModule({"direction": "up"}),
    )

    async def fake_run_calibration(build_fn, initial_state_fn, *, model, ground):
        return CalibrationReport(results=[
            CalibrationResult(event=SAMPLE_EVENT, predicted_direction="up", matched=True),
        ])
    monkeypatch.setattr(calibration_module, "run_calibration", fake_run_calibration)

    resp = client.post("/api/calibrate", json={})
    assert resp.status_code == 200
    assert captured["topology"] == "fanout"
    assert len(captured["agents"]) == 5  # DEFAULT_STOCK_REACTION_AGENTS


def test_calibrate_returns_report_shape(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        target_builder_module, "generate_target_source",
        lambda topology, agents, *, model, verdict_schema: "# generated",
    )
    monkeypatch.setattr(
        server_module, "_load_generated_module",
        lambda code, label: _FakeModule({"direction": "up"}),
    )

    async def fake_run_calibration(build_fn, initial_state_fn, *, model, ground):
        return CalibrationReport(results=[
            CalibrationResult(event=SAMPLE_EVENT, predicted_direction="up", matched=True),
        ])
    monkeypatch.setattr(calibration_module, "run_calibration", fake_run_calibration)

    resp = client.post("/api/calibrate", json={"topology": "pipeline", "agents": [
        {"name": "a", "role": "x"}, {"name": "b", "role": "y"},
    ]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_total"] == 1
    assert data["accuracy"] == 1.0
    assert "code" in data


def test_calibrate_surfaces_insufficient_events_as_400(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drift.calibration import CalibrationError

    monkeypatch.setattr(
        target_builder_module, "generate_target_source",
        lambda topology, agents, *, model, verdict_schema: "# generated",
    )
    monkeypatch.setattr(
        server_module, "_load_generated_module",
        lambda code, label: _FakeModule({"direction": "up"}),
    )

    async def failing_run_calibration(build_fn, initial_state_fn, *, model, ground):
        raise CalibrationError("need at least 2 historical events")
    monkeypatch.setattr(calibration_module, "run_calibration", failing_run_calibration)

    resp = client.post("/api/calibrate", json={})
    assert resp.status_code == 400
