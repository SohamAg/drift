"""Tests for drift.calibration.

All graph execution and retrieval calls are mocked — no real LLM calls,
no real graph builds. Real end-to-end proof was done manually during
development: a generated 2-agent pipeline with a verdict node, run
against 2 real held-out historical events, correctly predicted META's
"down" direction (matched=True) and correctly isolated a retrieval
failure on the second event into that event's own error field without
corrupting the accuracy calculation for the first.
"""
from __future__ import annotations

import asyncio

import pytest

import drift.calibration as calibration_module
from drift.calibration import (
    CalibrationError,
    CalibrationReport,
    CalibrationResult,
    run_calibration,
)
from drift.historical_grounding import HistoricalEvent

UP_EVENT = HistoricalEvent("NVDA", "d1", "d2", "nvda event", 100.0, 124.0, 24.0)
DOWN_EVENT = HistoricalEvent("META", "d1", "d2", "meta event", 100.0, 74.0, -26.0)
FLAT_EVENT = HistoricalEvent("XYZ", "d1", "d2", "flat event", 100.0, 100.3, 0.3)


class _FakeGraph:
    def __init__(self, verdict: dict | None, raises: Exception | None = None):
        self._verdict = verdict
        self._raises = raises

    async def ainvoke(self, state):
        if self._raises:
            raise self._raises
        return {**state, "verdict": self._verdict}


@pytest.fixture(autouse=True)
def no_real_retrieval(monkeypatch: pytest.MonkeyPatch):
    """Default: grounding retrieval returns no analogues and never makes
    a real call, unless a test overrides it. Keeps tests fast/offline."""
    async def fake_retrieve(scenario, events=None, *, k=3, model="gpt-4o-mini"):
        return []
    monkeypatch.setattr(calibration_module, "retrieve_similar_events", fake_retrieve)


def test_run_calibration_requires_at_least_two_events():
    with pytest.raises(CalibrationError):
        asyncio.run(run_calibration(
            lambda model: _FakeGraph({"direction": "up"}),
            lambda query: {"task": query},
            events=[UP_EVENT],
        ))


def test_run_calibration_scores_a_correct_match():
    def build_fn(model):
        return _FakeGraph({"direction": "up", "magnitude_pct": 10.0, "confidence": "high"})

    report = asyncio.run(run_calibration(
        build_fn, lambda query: {"task": query}, events=[UP_EVENT, DOWN_EVENT], ground=False,
    ))
    up_result = next(r for r in report.results if r.event is UP_EVENT)
    assert up_result.predicted_direction == "up"
    assert up_result.matched is True


def test_run_calibration_scores_an_incorrect_match():
    def build_fn(model):
        return _FakeGraph({"direction": "down"})

    report = asyncio.run(run_calibration(
        build_fn, lambda query: {"task": query}, events=[UP_EVENT, DOWN_EVENT], ground=False,
    ))
    up_result = next(r for r in report.results if r.event is UP_EVENT)
    assert up_result.predicted_direction == "down"
    assert up_result.matched is False


def test_run_calibration_treats_unparseable_direction_as_unscored():
    def build_fn(model):
        return _FakeGraph({"direction": "sideways-ish, hard to say"})  # not a valid direction

    report = asyncio.run(run_calibration(
        build_fn, lambda query: {"task": query}, events=[UP_EVENT, DOWN_EVENT], ground=False,
    ))
    assert all(r.predicted_direction is None for r in report.results)
    assert report.n_scored == 0
    assert report.accuracy is None  # not 0.0 — nothing was scorable


def test_run_calibration_treats_missing_verdict_key_as_unscored():
    def build_fn(model):
        return _FakeGraph(None)  # no verdict at all

    report = asyncio.run(run_calibration(
        build_fn, lambda query: {"task": query}, events=[UP_EVENT, DOWN_EVENT], ground=False,
    ))
    assert all(r.predicted_direction is None for r in report.results)


def test_run_calibration_isolates_one_event_failure_from_the_rest():
    calls = {"n": 0}

    def build_fn(model):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeGraph(None, raises=RuntimeError("boom"))
        return _FakeGraph({"direction": "up"})

    report = asyncio.run(run_calibration(
        build_fn, lambda query: {"task": query}, events=[UP_EVENT, DOWN_EVENT], ground=False,
    ))
    errored = [r for r in report.results if r.error is not None]
    ok = [r for r in report.results if r.error is None]
    assert len(errored) == 1
    assert "RuntimeError" in errored[0].error
    assert len(ok) == 1
    assert report.n_total == 2


def test_run_calibration_accuracy_excludes_errored_and_unscored_from_denominator():
    # Route each event's query to a graph whose behavior depends on which
    # event is being scored, via a closure over initial_state_fn's query.
    built_graphs = {
        "nvda event": _FakeGraph({"direction": "up"}),
        "meta event": _FakeGraph(None, raises=RuntimeError("x")),
        "flat event": _FakeGraph({"direction": "not-a-real-direction"}),
    }

    class _Router:
        def __call__(self, model):
            return self

        async def ainvoke(self, state):
            graph = built_graphs[state["task"]]
            return await graph.ainvoke(state)

    report = asyncio.run(run_calibration(
        _Router(), lambda query: {"task": query},
        events=[UP_EVENT, DOWN_EVENT, FLAT_EVENT], ground=False,
    ))
    assert report.n_total == 3
    assert report.n_scored == 1  # only the "up" one produced a valid direction
    assert report.n_matched == 1
    assert report.accuracy == 1.0


def test_run_calibration_ground_false_skips_retrieval_entirely(monkeypatch: pytest.MonkeyPatch):
    def fail_if_called(*a, **k):
        raise AssertionError("retrieve_similar_events should not be called when ground=False")
    monkeypatch.setattr(calibration_module, "retrieve_similar_events", fail_if_called)

    def build_fn(model):
        return _FakeGraph({"direction": "up"})

    report = asyncio.run(run_calibration(
        build_fn, lambda query: {"task": query}, events=[UP_EVENT, DOWN_EVENT], ground=False,
    ))
    assert report.n_total == 2


def test_run_calibration_ground_true_excludes_the_scored_event_itself(monkeypatch: pytest.MonkeyPatch):
    seen_pools = []

    async def fake_retrieve(scenario, events=None, *, k=3, model="gpt-4o-mini"):
        seen_pools.append(list(events or []))
        return []
    monkeypatch.setattr(calibration_module, "retrieve_similar_events", fake_retrieve)

    def build_fn(model):
        return _FakeGraph({"direction": "up"})

    asyncio.run(run_calibration(
        build_fn, lambda query: {"task": query},
        events=[UP_EVENT, DOWN_EVENT, FLAT_EVENT], ground=True,
    ))
    assert len(seen_pools) == 3
    for pool in seen_pools:
        # whichever event is currently being scored must not appear in
        # its own grounding pool (leave-one-out, not circular).
        assert len(pool) == 2


def test_calibration_report_to_json_shape():
    report = CalibrationReport(results=[
        CalibrationResult(event=UP_EVENT, predicted_direction="up", matched=True),
    ])
    d = report.to_json()
    assert d["n_total"] == 1
    assert d["n_scored"] == 1
    assert d["accuracy"] == 1.0
    assert d["results"][0]["event"]["ticker"] == "NVDA"
