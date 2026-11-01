"""Tests for drift.historical_grounding.

The LLM call is mocked via monkeypatching _call_retrieval_llm, never a
real API call. End-to-end proof that retrieval + grounding work with a
real LLM was done manually during development (scenario: "a major chip
export ban affecting an AI chipmaker" correctly retrieved NVDA as the
top analogue from the real seed dataset).
"""
from __future__ import annotations

import asyncio
import json

import pytest

import drift.historical_grounding as grounding_module
from drift.historical_grounding import (
    HISTORICAL_EVENTS,
    GroundingError,
    HistoricalEvent,
    build_grounded_query,
    load_events,
    retrieve_similar_events,
    weighted_baseline,
)


def test_seed_dataset_has_at_least_a_handful_of_events():
    events = load_events()
    assert len(events) >= 5


def test_seed_dataset_is_internally_consistent():
    """actual_move_pct must match what baseline/reaction close imply —
    guards against a typo in the hand-maintained seed data diverging
    from the numbers actually fetched from yfinance."""
    for e in HISTORICAL_EVENTS:
        expected = (e.reaction_close - e.baseline_close) / e.baseline_close * 100
        assert abs(expected - e.actual_move_pct) < 0.01, e.ticker


def test_seed_dataset_covers_both_directions():
    """Not just a pile of crashes — the calibration set should have real
    up-moves too, or a model that always predicts "down" would score
    deceptively well."""
    directions = {e.actual_direction for e in HISTORICAL_EVENTS}
    assert "up" in directions
    assert "down" in directions


def test_actual_direction_thresholds():
    up = HistoricalEvent("X", "d1", "d2", "desc", 100.0, 102.0, 2.0)
    down = HistoricalEvent("X", "d1", "d2", "desc", 100.0, 98.0, -2.0)
    flat = HistoricalEvent("X", "d1", "d2", "desc", 100.0, 100.5, 0.5)
    assert up.actual_direction == "up"
    assert down.actual_direction == "down"
    assert flat.actual_direction == "flat"


def test_load_events_returns_independent_copies():
    a = load_events()
    b = load_events()
    assert a == b
    assert a is not b  # mutating one list must not affect the other
    a.pop()
    assert len(load_events()) == len(b)


def test_retrieve_similar_events_rejects_empty_scenario():
    with pytest.raises(GroundingError):
        asyncio.run(retrieve_similar_events("   "))


def test_retrieve_similar_events_parses_llm_indices(monkeypatch: pytest.MonkeyPatch):
    async def fake_call(system, user, model):
        return json.dumps({"indices": [1, 0]})
    monkeypatch.setattr(grounding_module, "_call_retrieval_llm", fake_call)

    result = asyncio.run(retrieve_similar_events("some scenario", k=3))
    assert result == [HISTORICAL_EVENTS[1], HISTORICAL_EVENTS[0]]


def test_retrieve_similar_events_respects_k(monkeypatch: pytest.MonkeyPatch):
    async def fake_call(system, user, model):
        return json.dumps({"indices": [0, 1, 2, 3, 4]})
    monkeypatch.setattr(grounding_module, "_call_retrieval_llm", fake_call)

    result = asyncio.run(retrieve_similar_events("some scenario", k=2))
    assert len(result) == 2


def test_retrieve_similar_events_ignores_out_of_range_indices(monkeypatch: pytest.MonkeyPatch):
    async def fake_call(system, user, model):
        return json.dumps({"indices": [999, 0, -1]})
    monkeypatch.setattr(grounding_module, "_call_retrieval_llm", fake_call)

    result = asyncio.run(retrieve_similar_events("some scenario"))
    assert result == [HISTORICAL_EVENTS[0]]


def test_retrieve_similar_events_rejects_malformed_json(monkeypatch: pytest.MonkeyPatch):
    async def fake_call(system, user, model):
        return "not json"
    monkeypatch.setattr(grounding_module, "_call_retrieval_llm", fake_call)

    with pytest.raises(GroundingError):
        asyncio.run(retrieve_similar_events("some scenario"))


def test_retrieve_similar_events_rejects_no_valid_indices(monkeypatch: pytest.MonkeyPatch):
    async def fake_call(system, user, model):
        return json.dumps({"indices": [999]})
    monkeypatch.setattr(grounding_module, "_call_retrieval_llm", fake_call)

    with pytest.raises(GroundingError):
        asyncio.run(retrieve_similar_events("some scenario"))


def test_retrieve_similar_events_can_use_a_custom_pool():
    custom = [HISTORICAL_EVENTS[0]]

    async def fake_call(system, user, model):
        return json.dumps({"indices": [0]})

    import unittest.mock
    with unittest.mock.patch.object(grounding_module, "_call_retrieval_llm", fake_call):
        result = asyncio.run(retrieve_similar_events("x", events=custom))
    assert result == custom


def test_build_grounded_query_includes_scenario_and_analogues():
    query = build_grounded_query("Will the stock move?", [HISTORICAL_EVENTS[0]])
    assert "Will the stock move?" in query
    assert HISTORICAL_EVENTS[0].ticker in query
    assert f"{HISTORICAL_EVENTS[0].actual_move_pct:+.1f}%" in query


def test_build_grounded_query_with_no_analogues_still_includes_scenario():
    query = build_grounded_query("Will the stock move?", [])
    assert "Will the stock move?" in query
    assert "Relevant historical precedents" not in query


# ---------------------------------------------------------------- weighted_baseline

UP_EVENT = HistoricalEvent("NVDA", "d1", "d2", "up event", 100.0, 124.0, 24.0)
DOWN_EVENT = HistoricalEvent("META", "d1", "d2", "down event", 100.0, 74.0, -26.0)


def test_weighted_baseline_with_no_analogues_is_none():
    baseline = weighted_baseline([])
    assert baseline.direction is None
    assert baseline.magnitude_pct is None
    assert baseline.contributions == []


def test_weighted_baseline_single_analogue_matches_it_exactly():
    baseline = weighted_baseline([UP_EVENT])
    assert baseline.direction == "up"
    assert baseline.magnitude_pct == pytest.approx(24.0)
    assert len(baseline.contributions) == 1
    assert baseline.contributions[0]["weight"] == pytest.approx(1.0)


def test_weighted_baseline_default_rank_decay_favors_first_analogue():
    # Default weights are 1/(i+1): the first (most-relevant) analogue
    # gets more than twice the second's weight, so its direction wins
    # even though it's a 1-1 tie by raw count.
    baseline = weighted_baseline([UP_EVENT, DOWN_EVENT])
    assert baseline.direction == "up"
    # weighted average: (1*24 + 0.5*-26) / 1.5
    assert baseline.magnitude_pct == pytest.approx((1.0 * 24.0 + 0.5 * -26.0) / 1.5)


def test_weighted_baseline_custom_weights_can_flip_the_vote():
    baseline = weighted_baseline([UP_EVENT, DOWN_EVENT], weights=[1.0, 5.0])
    assert baseline.direction == "down"


def test_weighted_baseline_rejects_mismatched_weight_length():
    with pytest.raises(ValueError):
        weighted_baseline([UP_EVENT, DOWN_EVENT], weights=[1.0])


def test_weighted_baseline_rejects_non_positive_weight_sum():
    with pytest.raises(ValueError):
        weighted_baseline([UP_EVENT, DOWN_EVENT], weights=[0.0, 0.0])


def test_weighted_baseline_to_json_shape():
    baseline = weighted_baseline([UP_EVENT])
    d = baseline.to_json()
    assert d["direction"] == "up"
    assert d["contributions"][0]["event"]["ticker"] == "NVDA"
