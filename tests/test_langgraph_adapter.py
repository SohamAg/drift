"""Tests for the LangGraph adapter.

We don't depend on the langgraph package — the adapter contract is
"anything with .invoke / .ainvoke / .stream / .astream / __call__", so
tests use small stub graphs. Coverage:
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
  - Streaming graphs produce per-super-step traces; invoke-only graphs do not.
  - Judge plumbing: supplying a judge attaches findings per perturbation;
    user_guidelines flow through to the judge prompt.
"""
from __future__ import annotations

import asyncio
import json
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


# ---- streaming / trace capture -----------------------------------------


class _StreamingGraph:
    """Stub graph that supports `.stream()` (sync) with langgraph-shaped chunks.

    Two-node pipeline: `classify` writes intent, `respond` writes reply.
    Each yielded chunk mimics langgraph's `stream_mode="updates"` shape:
    `{node_name: partial_dict}`.
    """

    def stream(self, state: dict):
        running = dict(state)
        # node 1
        update1 = {"intent": "refund" if "refund" in (state.get("text") or "") else "other"}
        running.update(update1)
        yield {"classify": update1}
        # node 2
        update2 = {"reply": f"handled {running['intent']}"}
        yield {"respond": update2}

    # Also expose invoke so the existing crash/diverge tests keep working
    # if they ever pick this stub up — but the adapter prefers stream when
    # present.
    def invoke(self, state: dict) -> dict:
        out = dict(state)
        for chunk in self.stream(state):
            for upd in chunk.values():
                out.update(upd)
        return out


class _AsyncStreamingGraph:
    """Stub graph that supports `.astream()` (async)."""

    async def astream(self, state: dict):
        await asyncio.sleep(0)
        update1 = {"priority": "high"}
        yield {"classify": update1}
        await asyncio.sleep(0)
        yield {"respond": {"reply": "ok"}}


def test_streaming_graph_captures_per_super_step_trace():
    result = drift_test(
        graph=_StreamingGraph(),
        initial_state={"text": "I want a refund", "intent": "", "reply": ""},
        intensity="off",
        seed=0,
    )
    # Baseline trace has 2 super-steps, one per node.
    assert len(result.baseline.trace) == 2
    nodes = [entry["node"] for entry in result.baseline.trace]
    assert nodes == ["classify", "respond"]
    # Each trace entry carries the update + accumulated state_after.
    for entry in result.baseline.trace:
        assert "update" in entry and isinstance(entry["update"], dict)
        assert "state_after" in entry and isinstance(entry["state_after"], dict)
        assert "step" in entry


def test_async_streaming_graph_captures_trace():
    result = drift_test(
        graph=_AsyncStreamingGraph(),
        initial_state={"priority": "", "reply": ""},
        intensity="off",
        seed=0,
    )
    assert len(result.baseline.trace) == 2
    assert [e["node"] for e in result.baseline.trace] == ["classify", "respond"]
    # Final state reflects both updates merged in order.
    assert result.baseline.final_state == {"priority": "high", "reply": "ok"}


def test_invoke_only_graph_produces_empty_trace():
    # _PassthroughGraph has .invoke but no .stream/.astream
    result = drift_test(
        graph=_PassthroughGraph(),
        initial_state={"flag": True},
        intensity="off",
        seed=0,
    )
    assert result.baseline.trace == []


def test_perturbation_traces_independent_of_baseline():
    result = drift_test(
        graph=_StreamingGraph(),
        initial_state={"text": "I want a refund", "intent": "", "reply": ""},
        intensity="aggressive",
        seed=3,
    )
    assert result.perturbations, "test needs at least one perturbation"
    # Every non-crashed perturbation should have a 2-step trace
    # (StreamingGraph always yields 2 chunks unless it raised).
    for p in result.perturbations:
        if not p.crashed:
            assert len(p.trace) == 2, f"{p.event_name} should have 2 trace entries"


# ---- judge plumbing ----------------------------------------------------


class _CannedJudge:
    """Judge stub that returns a fixed payload and records every prompt
    it received so tests can assert what was sent.
    """

    def __init__(self, payload_obj: dict) -> None:
        self.payload = json.dumps(payload_obj)
        self.calls: list[dict] = []

    async def judge(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        return self.payload


def test_judge_findings_attach_to_perturbations():
    judge = _CannedJudge({
        "failures": [
            {
                "family": "coordination_contradiction",
                "summary": "test finding",
                "evidence_action_ids": [],
                "agents_involved": [],
            }
        ]
    })
    result = drift_test(
        graph=_StreamingGraph(),
        initial_state={"text": "I want a refund", "intent": "", "reply": ""},
        intensity="moderate",
        seed=5,
        judge_llm=judge,
    )
    # Judge fires once for baseline + once per perturbation that has a trace.
    runs_with_trace = 1 + sum(1 for p in result.perturbations if p.trace)
    assert len(judge.calls) == runs_with_trace

    # Findings attached: baseline + each traced perturbation.
    assert len(result.baseline.judge_findings) == 1
    assert result.baseline.judge_findings[0]["failure_type"] == "llm:coordination_contradiction"
    for p in result.perturbations:
        if p.trace:
            assert len(p.judge_findings) == 1


def test_judge_skipped_when_graph_has_no_stream():
    judge = _CannedJudge({"failures": []})
    result = drift_test(
        graph=_PassthroughGraph(),     # invoke-only stub, no .stream
        initial_state={"flag": True, "items": [1, 2]},
        intensity="moderate",
        seed=2,
        judge_llm=judge,
    )
    # No trace -> no judge calls anywhere.
    assert judge.calls == []
    assert result.baseline.judge_findings == []
    for p in result.perturbations:
        assert p.judge_findings == []


def test_user_guidelines_reach_the_judge():
    judge = _CannedJudge({"failures": []})
    drift_test(
        graph=_StreamingGraph(),
        initial_state={"text": "I want a refund", "intent": "", "reply": ""},
        intensity="off",  # baseline only, single judge call
        seed=0,
        judge_llm=judge,
        user_guidelines=[
            "classify must not skip the refund branch",
            "respond must echo the intent",
        ],
    )
    assert len(judge.calls) == 1
    system = judge.calls[0]["system"]
    assert "classify must not skip the refund branch" in system
    assert "respond must echo the intent" in system


def test_judge_findings_skipped_for_crashed_step_zero():
    """A graph that crashes before yielding any super-step has no trace,
    so the judge has nothing to read and should be silently skipped."""

    class _CrashOnEntry:
        def stream(self, state: dict):
            raise RuntimeError("nothing to see here")
            yield  # make it a generator

    judge = _CannedJudge({"failures": [{"family": "state_drift", "summary": "x"}]})
    result = drift_test(
        graph=_CrashOnEntry(),
        initial_state={"flag": True},
        intensity="off",
        seed=0,
        judge_llm=judge,
    )
    assert result.baseline.crashed is True
    assert result.baseline.trace == []
    assert result.baseline.judge_findings == []
    # Judge was never called.
    assert judge.calls == []


def test_n_judge_findings_counts_across_baseline_and_perturbations():
    judge = _CannedJudge({
        "failures": [{"family": "grounding_failure", "summary": "y"}]
    })
    result = drift_test(
        graph=_StreamingGraph(),
        initial_state={"text": "x", "intent": "", "reply": ""},
        intensity="moderate",
        seed=7,
        judge_llm=judge,
    )
    expected = len(result.baseline.judge_findings) + sum(
        len(p.judge_findings) for p in result.perturbations
    )
    assert result.n_judge_findings == expected
    # Summary line should mention judge findings when there are any.
    if expected > 0:
        assert any("judge findings" in line for line in result.summary_lines())


# ---- phase 2: tiered divergence cascade -----------------------------------


from drift.adapters.langgraph import (   # noqa: E402
    FieldDivergence,
    FieldNoiseBand,
    _analyze_field_variance,
    _diff_states_tiered,
    _is_within_noise,
    _text_similarity,
    _tier0_structural,
    _tier1_exact,
)


# ---- tier 0 + 1 (free deterministic tiers) --------------------------------


def test_tier0_detects_added_removed_keys_and_type_changes():
    baseline = {"a": 1, "b": "x", "shared": True}
    perturbed = {"a": 1, "shared": "now a string", "c": "new"}
    diffs = _tier0_structural(baseline, perturbed)
    by_name = {d.name: d for d in diffs}
    assert "-b" in by_name["b"].summary
    assert "+c" in by_name["c"].summary
    assert "type" in by_name["shared"].summary
    assert all(d.tier == 0 for d in diffs)


def test_tier1_ignores_canonical_equality():
    # Same dict, different key order — should not register as divergence.
    baseline = {"x": {"a": 1, "b": 2}}
    perturbed = {"x": {"b": 2, "a": 1}}
    diffs = _tier1_exact(baseline, perturbed, skip_fields=set())
    assert diffs == []


def test_tier1_flags_actual_value_differences():
    baseline = {"answer": "yes", "score": 0.5}
    perturbed = {"answer": "no", "score": 0.5}
    diffs = _tier1_exact(baseline, perturbed, skip_fields=set())
    assert [d.name for d in diffs] == ["answer"]
    assert diffs[0].tier == 1


def test_tier1_skips_fields_already_handled_by_tier0():
    baseline = {"shared": True, "other": "a"}
    perturbed = {"shared": "now str", "other": "b"}
    tier0 = _tier0_structural(baseline, perturbed)
    tier1 = _tier1_exact(baseline, perturbed, skip_fields={d.name for d in tier0})
    # "shared" handled by tier 0; tier 1 should only report "other"
    assert [d.name for d in tier1] == ["other"]


# ---- noise band analysis --------------------------------------------------


def test_analyze_field_variance_for_enum_field():
    band = _analyze_field_variance("priority", ["high", "low", "high", "normal"])
    assert band.sample_count == 4
    assert set(band.distinct_values) == {"high", "low", "normal"}
    assert band.value_frequencies['"high"'] == 2


def test_analyze_field_variance_for_numeric_field():
    band = _analyze_field_variance("score", [0.5, 0.7, 0.9])
    assert band.numeric_min == 0.5
    assert band.numeric_max == 0.9


def test_analyze_field_variance_for_text_field_records_similarity():
    band = _analyze_field_variance("reply", [
        "Your refund has been processed",
        "We've refunded your order",
        "Refund processed successfully",
    ])
    assert band.text_min_similarity is not None
    assert band.text_mean_similarity is not None
    assert 0.0 <= band.text_min_similarity <= band.text_mean_similarity <= 1.0


def test_analyze_field_variance_skips_text_similarity_for_bool_field():
    band = _analyze_field_variance("is_premium", [True, False, True])
    assert band.text_min_similarity is None
    assert band.numeric_min is None  # bool excluded from numeric path


# ---- tier 2 noise filtering -----------------------------------------------


def test_is_within_noise_exact_match_to_observed_value():
    band = FieldNoiseBand(
        name="priority", sample_count=3,
        distinct_values=["high", "normal", "low"],
        value_frequencies={'"high"': 1, '"normal"': 1, '"low"': 1},
    )
    within, sim = _is_within_noise("normal", band, similarity_threshold=0.85)
    assert within is True
    assert sim is None  # exact-match path doesn't compute similarity


def test_is_within_noise_text_similar_to_baseline_passes():
    band = _analyze_field_variance("reply", [
        "Your refund has been processed",
        "We've refunded your order",
    ])
    # Highly similar text should be within noise.
    within, sim = _is_within_noise(
        "Your refund was processed", band, similarity_threshold=0.5,
    )
    assert within is True
    assert sim is not None and sim > 0.5


def test_is_within_noise_substantively_different_text_fails():
    band = _analyze_field_variance("reply", [
        "Approved for refund of $49.99",
        "Refund of $49.99 approved",
    ])
    within, sim = _is_within_noise(
        "Cannot process refund at this time", band, similarity_threshold=0.85,
    )
    assert within is False


def test_is_within_noise_numeric_in_range():
    band = _analyze_field_variance("score", [0.5, 0.6, 0.7])
    assert _is_within_noise(0.55, band, similarity_threshold=0.85)[0] is True
    assert _is_within_noise(0.9, band, similarity_threshold=0.85)[0] is False


def test_is_within_noise_no_band_means_not_within():
    # Without a band (e.g., single-rollout baseline), we can't say "within noise".
    within, sim = _is_within_noise("anything", None, similarity_threshold=0.85)
    assert within is False
    assert sim is None


# ---- tier 3 judge equivalence + budget ------------------------------------


def test_tiered_cascade_filters_noise_keeps_real_divergence():
    """Field with noise should be filtered out; field that exceeds noise stays."""
    baseline = {"reply": "Refund processed", "answer": "approved"}
    perturbed = {"reply": "Refund has been processed", "answer": "denied"}
    noise = {
        "reply": _analyze_field_variance("reply", [
            "Refund processed",
            "Refund completed",
            "Refund done",
        ]),
        "answer": _analyze_field_variance("answer", ["approved", "approved"]),
    }
    diverged, _summary, details, judge_used = asyncio.run(_diff_states_tiered(
        baseline, perturbed, noise_band=noise, judge_llm=None,
        similarity_threshold=0.5, judge_calls_remaining=0,
    ))
    # reply variants are similar enough to be within noise -> filtered.
    # answer changed approved->denied (not in noise) -> retained.
    assert diverged is True
    names = [d.name for d in details]
    assert "answer" in names
    assert "reply" not in names
    assert judge_used == 0  # no judge configured


def test_tiered_cascade_calls_judge_for_survivors_and_respects_budget():
    """When a judge is supplied, tier-3 fires on survivors; budget caps calls."""

    # Judge that always says "not equivalent" so survivors stay reported.
    class _JudgeNotEquiv:
        def __init__(self):
            self.calls = 0

        async def judge(self, *, system: str, user: str) -> str:
            self.calls += 1
            return '{"equivalent": false, "reasoning": "different"}'

    baseline = {"a": "one", "b": "two", "c": "three"}
    perturbed = {"a": "ONE", "b": "TWO", "c": "THREE"}
    # No noise band -> tier 2 fails closed -> tier 3 fires for each field.
    judge = _JudgeNotEquiv()
    diverged, _summary, details, judge_used = asyncio.run(_diff_states_tiered(
        baseline, perturbed, noise_band={}, judge_llm=judge,
        similarity_threshold=0.85, judge_calls_remaining=2,
    ))
    assert diverged is True
    # 3 differing fields, but budget is 2 -> judge called exactly 2x; the
    # 3rd field surfaces without a judge verdict.
    assert judge.calls == 2
    assert judge_used == 2
    # All 3 fields end up in details (tier-3-judged ones marked as different,
    # the budget-exhausted one falls through as a plain tier-1 divergence).
    assert len(details) == 3


def test_tiered_cascade_judge_clears_equivalent_fields():
    """Judge saying 'equivalent' should drop the divergence from details."""

    class _JudgeEquiv:
        async def judge(self, *, system: str, user: str) -> str:
            return '{"equivalent": true, "reasoning": "same meaning"}'

    baseline = {"reply": "Approved"}
    perturbed = {"reply": "Yes, approved."}
    diverged, _summary, details, judge_used = asyncio.run(_diff_states_tiered(
        baseline, perturbed, noise_band={}, judge_llm=_JudgeEquiv(),
        similarity_threshold=0.85, judge_calls_remaining=5,
    ))
    assert diverged is False
    assert details == []
    assert judge_used == 1


def test_tiered_cascade_judge_error_surfaces_divergence():
    """If the judge throws, we should NOT silently drop the divergence."""

    class _JudgeBroken:
        async def judge(self, *, system: str, user: str) -> str:
            raise RuntimeError("API down")

    baseline = {"x": "a"}
    perturbed = {"x": "b"}
    diverged, _, details, judge_used = asyncio.run(_diff_states_tiered(
        baseline, perturbed, noise_band={}, judge_llm=_JudgeBroken(),
        similarity_threshold=0.85, judge_calls_remaining=5,
    ))
    assert diverged is True
    assert len(details) == 1
    assert "judge error" in details[0].summary.lower()
    assert judge_used == 1


def test_tiered_cascade_passes_through_tier0_structural_always():
    """Type changes and key add/remove must be reported regardless of noise."""
    baseline = {"x": 1, "y": "a"}
    perturbed = {"x": "now str", "z": "added"}  # x type-changed, y removed, z added
    # Even with a huge noise band that "permits anything," structural changes
    # are unfilterable.
    noise = {
        "x": _analyze_field_variance("x", [1, "str_form", 2]),
        "y": _analyze_field_variance("y", ["a", "a", "a"]),
        "z": _analyze_field_variance("z", ["added", "added"]),
    }
    diverged, _, details, _ = asyncio.run(_diff_states_tiered(
        baseline, perturbed, noise_band=noise, judge_llm=None,
        similarity_threshold=0.85, judge_calls_remaining=0,
    ))
    assert diverged is True
    tier0_names = {d.name for d in details if d.tier == 0}
    assert {"x", "y", "z"}.issubset(tier0_names)


# ---- end-to-end with divergence_mode + baseline_rollouts ------------------


def test_divergence_mode_off_skips_divergence_detection():
    result = drift_test(
        graph=_DivergingGraph(),
        initial_state={"flag": True, "items": ["x", "y"]},
        intensity="moderate",
        seed=4,
        divergence_mode="off",
    )
    # No divergence reported even though graph clearly diverges.
    assert all(p.diverged is False for p in result.perturbations)
    assert all(p.divergence_details == [] for p in result.perturbations)


def test_divergence_mode_tiered_with_rollouts_measures_noise_band():
    """With baseline_rollouts > 1, the result carries a non-empty noise_band."""
    result = drift_test(
        graph=_StreamingGraph(),
        initial_state={"text": "I want a refund", "intent": "", "reply": ""},
        intensity="off",                  # no perturbations — just measure noise
        seed=0,
        divergence_mode="tiered",
        baseline_rollouts=3,
    )
    assert result.divergence_mode == "tiered"
    assert result.baseline_rollouts == 3
    # _StreamingGraph is deterministic so each field has 1 distinct value;
    # we still get a band recorded (just with sample_count=3, no variance).
    assert result.noise_band  # non-empty
    for name, band in result.noise_band.items():
        assert band.sample_count == 3


def test_divergence_mode_tiered_judge_budget_is_tracked():
    """judge_calls_used reports how many tier-3 calls fired."""

    class _Judge:
        def __init__(self):
            self.calls = 0

        async def judge(self, *, system: str, user: str) -> str:
            self.calls += 1
            # Real coordination judge AND divergence judge use the same protocol;
            # this judge stub answers both. We use the "failures: []" shape for
            # the coord-judge calls and "equivalent: false" for divergence calls.
            if '"equivalent"' in system or "semantically equivalent" in system:
                return '{"equivalent": false, "reasoning": "different"}'
            return '{"failures": []}'

    judge = _Judge()
    result = drift_test(
        graph=_DivergingGraph(),
        initial_state={"flag": True, "items": ["x", "y"]},
        intensity="moderate",
        seed=4,
        divergence_mode="tiered",
        judge_llm=judge,
        max_judge_calls=3,
    )
    # Cost telemetry surfaces.
    assert result.judge_calls_budget == 3
    assert result.judge_calls_used <= 3


# ---------------------------------------------------------------------------
# Phase 3: coordination-detector library integration
# ---------------------------------------------------------------------------


class _VerifierLoopGraph:
    """Streams 8 super-steps of planner+verifier alternation; verifier always
    emits verdict='approve'. With all state keys pre-seeded, no progress
    occurs across the loop — fires both verifier_always_approves AND
    infinite_handoff from the library."""

    def stream(self, state: dict):
        for _ in range(4):
            u = {"rationale": "(thinking)"}
            yield {"planner": u}
            u = {"verdict": "approve"}
            yield {"verifier": u}


def test_coordination_library_fires_through_adapter():
    result = drift_test(
        graph=_VerifierLoopGraph(),
        initial_state={
            "task": "review feature x",
            "rationale": "(thinking)",
            "verdict": "approve",
        },
        intensity="off",        # baseline-only path
        seed=1,
    )
    types_baseline = {f["failure_type"] for f in result.baseline.coordination_findings}
    assert "verifier_always_approves" in types_baseline
    assert "infinite_handoff" in types_baseline
    # Aggregate counter tracks them.
    assert result.n_coordination_findings >= 2


def test_coordination_library_can_be_disabled():
    result = drift_test(
        graph=_VerifierLoopGraph(),
        initial_state={
            "task": "x",
            "rationale": "(thinking)",
            "verdict": "approve",
        },
        intensity="off",
        seed=1,
        run_coordination_detectors=False,
    )
    assert result.baseline.coordination_findings == []
    assert result.n_coordination_findings == 0


def test_coordination_library_empty_when_no_trace():
    """Plain .invoke()-only graph: no trace, library has nothing to scan."""

    class _NoStream:
        def invoke(self, state: dict) -> dict:
            return dict(state)

    result = drift_test(
        graph=_NoStream(),
        initial_state={"x": 1, "y": "hi"},
        intensity="off",
        seed=1,
    )
    # No trace means detector library is silent — not an error.
    assert result.baseline.coordination_findings == []


def test_coordination_library_explicit_roles_passthrough():
    """User-declared roles let detectors fire on agents whose names don't
    match the default verifier regex."""

    class _GenericGraph:
        def stream(self, state: dict):
            for _ in range(4):
                yield {"agent_x": {"rationale": "..."}}
                yield {"agent_y": {"verdict": "approve"}}

    result = drift_test(
        graph=_GenericGraph(),
        initial_state={"task": "x", "rationale": "...", "verdict": "approve"},
        intensity="off",
        seed=1,
        coordination_roles={"agent_y": "verifier"},
    )
    types = {f["failure_type"] for f in result.baseline.coordination_findings}
    assert "verifier_always_approves" in types
