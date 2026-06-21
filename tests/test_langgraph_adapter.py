"""Tests for the LangGraph adapter.

We don't depend on the langgraph package — the adapter contract is
"anything with .invoke / .ainvoke / __call__", so tests use small stub
graphs. Coverage:
  - Baseline + perturbations actually run, and per-perturbation results
    track which chaos pattern was applied.
  - Crash detection: an exception inside the graph surfaces as
    crashed=True, error_type populated, final_state=None.
  - Divergence detection: same input + chaos -> different output gets
    flagged; unchanged behavior under chaos stays diverged=False.
  - Schema dispatch: bool / dict / list / string / numeric fields each
    produce at least one applicable chaos perturbation.
  - Exclusion filter passes through to plan_auto_chaos.
  - Async variant runs from inside an existing event loop.
  - state_factory hook is invoked per perturbation when supplied.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from drift.adapters.langgraph import (
    AdapterResult,
    BaselineResult,
    PerturbationResult,
    drift_test,
    drift_test_async,
)


# ---- stub graphs ---------------------------------------------------------


class _PassthroughGraph:
    """Echoes the input state back unchanged."""

    def invoke(self, state: dict) -> dict:
        return dict(state)


class _DivergingGraph:
    """Records the perturbed input verbatim in a `seen` key so any chaos
    mutation produces a visibly different final state."""

    def invoke(self, state: dict) -> dict:
        out = dict(state)
        out["seen"] = dict(state)
        return out


class _CrashOnFlagGraph:
    """Raises when `is_admin` is False — designed to be tripped by
    flip_bool[is_admin] when the baseline state has is_admin=True."""

    def invoke(self, state: dict) -> dict:
        if state.get("is_admin") is False:
            raise RuntimeError("non-admin not allowed here")
        return {**state, "ok": True}


class _AsyncGraph:
    """Async-only graph — exercises the ainvoke branch."""

    async def ainvoke(self, state: dict) -> dict:
        await asyncio.sleep(0)
        return {**state, "async_seen": True}


def _callable_graph(state: dict) -> dict:
    """Plain function-as-graph — exercises the callable fallback."""
    return {**state, "via_call": True}


# ---- baseline + perturbation execution ----------------------------------


def test_baseline_runs_and_records_final_state():
    result = drift_test(
        graph=_PassthroughGraph(),
        initial_state={"flag": True, "items": ["a", "b"]},
        intensity="off",  # no perturbations — just baseline
        seed=1,
    )
    assert isinstance(result, AdapterResult)
    assert isinstance(result.baseline, BaselineResult)
    assert result.baseline.crashed is False
    assert result.baseline.final_state == {"flag": True, "items": ["a", "b"]}
    assert result.perturbations == []
    assert result.intensity == "off"


def test_perturbations_run_and_track_chaos_pattern():
    # Moderate intensity over a multi-field state should schedule at least
    # one perturbation. Each PerturbationResult should know which pattern
    # was applied and to which field.
    result = drift_test(
        graph=_PassthroughGraph(),
        initial_state={
            "is_admin": True,
            "open_cases": {"CASE-1": {"id": "CASE-1"}},
            "queue": ["a", "b"],
            "label": "hello",
            "load": 0.5,
        },
        intensity="aggressive",
        seed=7,
    )
    assert result.patterns_total > 0
    assert len(result.perturbations) > 0
    for p in result.perturbations:
        assert isinstance(p, PerturbationResult)
        assert p.event_name.startswith("AutoChaos.")
        assert p.pattern_type
        assert p.perturbed_field
        # Each perturbation either crashed or returned a final_state;
        # never both, never neither.
        if p.crashed:
            assert p.final_state is None
            assert p.error_type
        else:
            assert p.final_state is not None


def test_diverging_graph_flags_diverged_perturbations():
    # Passthrough echoes its input verbatim into a `seen` field, so every
    # chaos mutation must show up as a divergence vs baseline.
    result = drift_test(
        graph=_DivergingGraph(),
        initial_state={"is_admin": True, "queue": ["x", "y"]},
        intensity="aggressive",
        seed=3,
    )
    assert len(result.perturbations) > 0
    diverged = [p for p in result.perturbations if p.diverged]
    assert len(diverged) == len(result.perturbations), (
        "every perturbation should diverge for the diverging graph"
    )
    # Divergence summary should mention something from the state.
    assert all(p.divergence_summary for p in diverged)


def test_passthrough_graph_yields_no_divergence():
    # Passthrough returns the input as-is. Even when chaos mutates the
    # input, the final state still equals that mutated input — but
    # baseline's final state also equals baseline input. So baseline-vs-
    # perturbed always shows divergence on whichever field was mutated.
    # The exception: if chaos was a no-op (field not present at runtime),
    # divergence should be False. We verify by checking the per-event
    # `event_summary` and divergence flag stay consistent.
    result = drift_test(
        graph=_PassthroughGraph(),
        initial_state={"flag": True, "items": ["a", "b"]},
        intensity="aggressive",
        seed=11,
    )
    for p in result.perturbations:
        applied = "no-op" not in p.event_summary
        assert p.diverged is applied, (
            f"event {p.event_name} summary={p.event_summary!r} "
            f"diverged={p.diverged} applied={applied}"
        )


# ---- crash detection -----------------------------------------------------


def test_crash_under_chaos_is_captured():
    # Baseline has is_admin=True so the graph happily returns ok=True.
    # flip_bool[is_admin] will turn it False and trip the RuntimeError.
    result = drift_test(
        graph=_CrashOnFlagGraph(),
        initial_state={"is_admin": True},
        intensity="aggressive",
        seed=42,
        # Ensure at least one is_admin flip is in the schedule by leaving
        # the schema small — only one fuzzable field exists.
    )
    assert result.baseline.crashed is False
    assert result.baseline.final_state == {"is_admin": True, "ok": True}
    crashed = [p for p in result.perturbations if p.crashed]
    assert len(crashed) >= 1
    assert all(p.error_type == "RuntimeError" for p in crashed)
    assert all("non-admin" in p.error for p in crashed)
    assert all(p.final_state is None for p in crashed)
    assert result.n_crashed == len(crashed)


def test_baseline_crash_surfaces_in_result():
    class _AlwaysCrashes:
        def invoke(self, state: dict) -> dict:
            raise ValueError("nope")

    result = drift_test(
        graph=_AlwaysCrashes(),
        initial_state={"x": 1},
        intensity="off",
        seed=0,
    )
    assert result.baseline.crashed is True
    assert result.baseline.error_type == "ValueError"
    assert "nope" in result.baseline.error
    assert result.baseline.final_state is None
    # summary_lines mentions baseline crash so users see it
    lines = result.summary_lines()
    assert any("baseline itself crashed" in line for line in lines)


# ---- schema dispatch + exclusion ----------------------------------------


def test_each_supported_type_produces_at_least_one_pattern():
    # discover_field_patterns is the chaos primitive; we go through the
    # adapter to confirm the wiring + patterns_total reflects what we'd
    # expect from a varied schema.
    result = drift_test(
        graph=_PassthroughGraph(),
        initial_state={
            "a_bool": True,
            "a_int": 7,
            "a_float": 0.5,
            "a_str": "hello",
            "a_dict": {"k": 1},
            "a_list": [1, 2, 3],
        },
        intensity="off",
        seed=0,
    )
    # 6 fields, each producing >= 1 spec.
    assert result.patterns_total >= 6


def test_exclude_filters_perturbations():
    # Excluding flip_bool prevents any perturbation against the bool field.
    result = drift_test(
        graph=_PassthroughGraph(),
        initial_state={
            "switch": True,
            "items": ["x"],
        },
        intensity="aggressive",
        seed=5,
        auto_chaos_exclude=["flip_bool"],
    )
    for p in result.perturbations:
        assert p.pattern_type != "flip_bool"


def test_max_perturbations_clamps():
    # Asking for aggressive intensity over a wide schema while clamping
    # to 2 should cap perturbations at 2.
    state = {
        "f1": True, "f2": False, "f3": True, "f4": False,
        "d1": {"a": 1}, "d2": {"b": 2}, "d3": {"c": 3},
        "l1": [1, 2, 3], "l2": ["x", "y"],
    }
    result = drift_test(
        graph=_PassthroughGraph(),
        initial_state=state,
        intensity="aggressive",
        seed=99,
        max_perturbations=2,
    )
    assert len(result.perturbations) <= 2


# ---- graph-shape compatibility ------------------------------------------


def test_async_only_graph_is_supported():
    result = drift_test(
        graph=_AsyncGraph(),
        initial_state={"flag": True},
        intensity="off",
        seed=0,
    )
    assert result.baseline.final_state == {"flag": True, "async_seen": True}


def test_plain_callable_is_supported():
    result = drift_test(
        graph=_callable_graph,
        initial_state={"q": 1},
        intensity="off",
        seed=0,
    )
    assert result.baseline.final_state == {"q": 1, "via_call": True}


def test_non_graph_object_raises_with_clear_message():
    with pytest.raises(TypeError, match="no .invoke/.ainvoke"):
        drift_test(graph=object(), initial_state={"x": 1}, intensity="off")


def test_initial_state_must_be_dict():
    with pytest.raises(TypeError, match="initial_state must be a dict"):
        drift_test(graph=_PassthroughGraph(), initial_state=[1, 2, 3])  # type: ignore[arg-type]


# ---- async API -----------------------------------------------------------


def test_drift_test_async_runs_inside_event_loop():
    async def _main() -> AdapterResult:
        return await drift_test_async(
            graph=_AsyncGraph(),
            initial_state={"flag": True, "items": ["a"]},
            intensity="moderate",
            seed=4,
        )

    result = asyncio.run(_main())
    assert result.baseline.final_state is not None
    assert result.intensity == "moderate"


# ---- state_factory hook -------------------------------------------------


def test_state_factory_is_called_per_run():
    calls: list[int] = []

    def fresh() -> dict:
        calls.append(len(calls) + 1)
        return {"flag": True, "items": [1, 2]}

    result = drift_test(
        graph=_PassthroughGraph(),
        initial_state={"_unused": True},  # ignored when state_factory supplied
        intensity="moderate",
        seed=2,
        state_factory=fresh,
    )
    # 1 baseline + 1 per perturbation
    assert len(calls) == 1 + len(result.perturbations)


# ---- result accounting --------------------------------------------------


def test_counters_partition_perturbations():
    # crashed + diverged + unchanged should sum to len(perturbations).
    result = drift_test(
        graph=_DivergingGraph(),
        initial_state={"flag": True, "items": ["x", "y"]},
        intensity="aggressive",
        seed=8,
    )
    total = result.n_crashed + result.n_diverged + result.n_unchanged
    assert total == len(result.perturbations)


def test_summary_lines_describe_run():
    result = drift_test(
        graph=_PassthroughGraph(),
        initial_state={"flag": True},
        intensity="moderate",
        seed=1,
    )
    lines = result.summary_lines()
    assert any("perturbation" in line for line in lines)
    assert any("crashed" in line for line in lines)
