"""Server-level test for GET /api/detectors — zero prior test coverage.

Backs the new in-product Detectors reference view. Asserts the endpoint
reads live from drift.failures.library.ALL_DETECTORS and
drift.failures.judge.JUDGE_FAMILY_DESCRIPTIONS rather than a
hand-maintained list, so it can't silently drift out of sync with what
actually ships.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from drift.failures.judge import JUDGE_FAMILY_DESCRIPTIONS, USER_GUIDELINE_FAMILY
from drift.failures.library import ALL_DETECTORS
from drift.server import create_app


def test_lists_every_detector_with_required_fields() -> None:
    client = TestClient(create_app())
    resp = client.get("/api/detectors")
    assert resp.status_code == 200
    detectors = resp.json()["detectors"]
    assert len(detectors) == len(ALL_DETECTORS)

    names = {d["name"] for d in detectors}
    assert names == {mod.NAME for mod in ALL_DETECTORS}

    required = {"name", "summary", "source", "mast_modes", "needs_config"}
    for d in detectors:
        assert required <= d.keys()
        assert isinstance(d["mast_modes"], list)
        assert isinstance(d["needs_config"], bool)


def test_provisional_detectors_are_flagged_needs_config() -> None:
    client = TestClient(create_app())
    detectors = {d["name"]: d for d in client.get("/api/detectors").json()["detectors"]}
    assert detectors["premature_termination"]["needs_config"] is True
    assert detectors["policy_inconsistency_across_steps"]["needs_config"] is True
    assert detectors["role_drift"]["needs_config"] is False


def test_lists_every_judge_family_with_required_fields() -> None:
    client = TestClient(create_app())
    families = client.get("/api/detectors").json()["judge_families"]
    assert len(families) == len(JUDGE_FAMILY_DESCRIPTIONS)

    names = {f["name"] for f in families}
    assert names == set(JUDGE_FAMILY_DESCRIPTIONS.keys())

    required = {"name", "description", "is_user_guideline"}
    for f in families:
        assert required <= f.keys()
        assert isinstance(f["description"], str) and f["description"]


def test_user_guideline_family_is_flagged() -> None:
    client = TestClient(create_app())
    families = {f["name"]: f for f in client.get("/api/detectors").json()["judge_families"]}
    assert families[USER_GUIDELINE_FAMILY]["is_user_guideline"] is True
    assert families["state_drift"]["is_user_guideline"] is False
