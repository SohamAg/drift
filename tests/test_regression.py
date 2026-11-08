"""Tests for drift.regression — snapshot/compare/persistence logic.

Uses lightweight stand-ins for AdapterResult/BaselineResult/PerturbationResult
rather than the real dataclasses: snapshot_from_result only reads a handful
of attributes, and duck-typing keeps these tests from needing a real graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from drift.regression import (
    RegressionFlag,
    RegressionReport,
    RunSnapshot,
    append_snapshot,
    compare,
    load_history,
    snapshot_from_result,
    track,
)


@dataclass
class _Baseline:
    duration_s: float = 1.0
    crashed: bool = False
    coordination_findings: list[dict] = field(default_factory=list)


@dataclass
class _Perturbation:
    duration_s: float = 1.0
    coordination_findings: list[dict] = field(default_factory=list)


@dataclass
class _Result:
    intensity: str = "light"
    patterns_total: int = 10
    n_crashed: int = 0
    n_diverged: int = 0
    n_unchanged: int = 10
    n_judge_findings: int = 0
    n_coordination_findings: int = 0
    judge_calls_used: int = 0
    baseline: _Baseline = field(default_factory=_Baseline)
    perturbations: list[_Perturbation] = field(default_factory=list)


def _finding(failure_type: str) -> dict:
    return {"failure_type": failure_type, "summary": "x", "timestep": 1, "snapshot_timestep": 1}


def test_snapshot_from_result_counts_finding_types() -> None:
    result = _Result(
        baseline=_Baseline(coordination_findings=[_finding("infinite_handoff")]),
        perturbations=[
            _Perturbation(duration_s=2.0, coordination_findings=[_finding("infinite_handoff")]),
            _Perturbation(duration_s=4.0, coordination_findings=[_finding("verifier_always_approves")]),
        ],
    )
    snap = snapshot_from_result(result, "my_graph")
    assert snap.graph_signature == "my_graph"
    assert snap.coordination_finding_types == {"infinite_handoff": 2, "verifier_always_approves": 1}
    assert snap.mean_perturbation_duration_s == 3.0


def test_snapshot_from_result_empty_perturbations_no_crash() -> None:
    snap = snapshot_from_result(_Result(), "my_graph")
    assert snap.mean_perturbation_duration_s == 0.0
    assert snap.coordination_finding_types == {}


def test_snapshot_round_trips_through_json() -> None:
    snap = snapshot_from_result(
        _Result(baseline=_Baseline(coordination_findings=[_finding("role_drift")])),
        "g",
    )
    restored = RunSnapshot.from_json(snap.to_json())
    assert restored == snap


def test_compare_no_history_has_no_previous() -> None:
    current = snapshot_from_result(_Result(), "g")
    report = compare(current, [])
    assert report.has_previous is False
    assert report.flags == []
    assert report.is_regression is False


def test_compare_flags_new_baseline_crash() -> None:
    previous = snapshot_from_result(_Result(), "g")
    previous.timestamp = 100.0
    current = snapshot_from_result(_Result(baseline=_Baseline(crashed=True)), "g")
    current.timestamp = 200.0

    report = compare(current, [previous])
    assert report.has_previous is True
    assert any(f.severity == "crash" for f in report.flags)
    assert report.is_regression is True


def test_compare_flags_new_finding_type() -> None:
    previous = snapshot_from_result(_Result(), "g")
    previous.timestamp = 100.0
    current = snapshot_from_result(
        _Result(perturbations=[_Perturbation(coordination_findings=[_finding("infinite_handoff")])]),
        "g",
    )
    current.timestamp = 200.0

    report = compare(current, [previous])
    assert report.new_finding_types == ["infinite_handoff"]
    assert report.resolved_finding_types == []
    assert report.is_regression is True


def test_compare_flags_resolved_finding_as_info_not_regression() -> None:
    previous = snapshot_from_result(
        _Result(perturbations=[_Perturbation(coordination_findings=[_finding("infinite_handoff")])]),
        "g",
    )
    previous.timestamp = 100.0
    current = snapshot_from_result(_Result(), "g")
    current.timestamp = 200.0

    report = compare(current, [previous])
    assert report.resolved_finding_types == ["infinite_handoff"]
    assert all(f.severity != "regression" for f in report.flags)
    assert report.is_regression is False


def test_compare_flags_crash_rate_increase_above_threshold() -> None:
    previous = snapshot_from_result(_Result(patterns_total=10, n_crashed=0), "g")
    previous.timestamp = 100.0
    # 2/10 = 20% crash rate, previous was 0% — well above the 5% threshold.
    current = snapshot_from_result(_Result(patterns_total=10, n_crashed=2), "g")
    current.timestamp = 200.0

    report = compare(current, [previous])
    assert report.crash_rate_delta > 0.05
    assert any("crash rate up" in f.summary for f in report.flags)
    assert report.is_regression is True


def test_compare_ignores_small_crash_rate_jitter() -> None:
    previous = snapshot_from_result(_Result(patterns_total=100, n_crashed=1), "g")
    previous.timestamp = 100.0
    # 2/100 vs 1/100 — a 1pp move, under the 5pp threshold.
    current = snapshot_from_result(_Result(patterns_total=100, n_crashed=2), "g")
    current.timestamp = 200.0

    report = compare(current, [previous])
    assert not any(f.severity == "regression" for f in report.flags)


def test_compare_flags_duration_regression() -> None:
    previous = snapshot_from_result(_Result(perturbations=[_Perturbation(duration_s=1.0)]), "g")
    previous.timestamp = 100.0
    current = snapshot_from_result(_Result(perturbations=[_Perturbation(duration_s=3.0)]), "g")
    current.timestamp = 200.0

    report = compare(current, [previous])
    assert report.duration_delta_pct == 2.0
    assert any("runtime up" in f.summary for f in report.flags)


def test_compare_picks_most_recent_prior_not_oldest() -> None:
    old = snapshot_from_result(_Result(), "g")
    old.timestamp = 50.0
    mid = snapshot_from_result(
        _Result(perturbations=[_Perturbation(coordination_findings=[_finding("x")])]), "g"
    )
    mid.timestamp = 100.0
    current = snapshot_from_result(_Result(), "g")
    current.timestamp = 200.0

    report = compare(current, [old, mid])
    # "x" existed in `mid` (most recent prior) but not `old` — resolved,
    # not new, proving comparison used `mid` as the baseline.
    assert report.resolved_finding_types == ["x"]


def test_history_persistence_round_trip(tmp_path: Path) -> None:
    snap1 = snapshot_from_result(_Result(), "g")
    snap1.timestamp = 100.0
    append_snapshot(snap1, history_dir=tmp_path)

    snap2 = snapshot_from_result(_Result(n_crashed=1, patterns_total=10), "g")
    snap2.timestamp = 200.0
    append_snapshot(snap2, history_dir=tmp_path)

    loaded = load_history("g", history_dir=tmp_path)
    assert len(loaded) == 2
    assert loaded[0].timestamp == 100.0
    assert loaded[1].timestamp == 200.0


def test_history_is_isolated_per_graph_signature(tmp_path: Path) -> None:
    snap_a = snapshot_from_result(_Result(), "graph_a")
    append_snapshot(snap_a, history_dir=tmp_path)
    snap_b = snapshot_from_result(_Result(), "graph_b")
    append_snapshot(snap_b, history_dir=tmp_path)

    assert len(load_history("graph_a", history_dir=tmp_path)) == 1
    assert len(load_history("graph_b", history_dir=tmp_path)) == 1


def test_load_history_never_run_before_returns_empty(tmp_path: Path) -> None:
    assert load_history("never_seen", history_dir=tmp_path) == []


def test_load_history_skips_corrupt_lines(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    snap = snapshot_from_result(_Result(), "g")
    path.write_text("not json\n" + __import__("json").dumps(snap.to_json()) + "\n", encoding="utf-8")
    loaded = load_history("g", history_dir=tmp_path)
    assert len(loaded) == 1


def test_track_appends_and_compares_in_one_call(tmp_path: Path) -> None:
    report1 = track(_Result(), "g", history_dir=tmp_path)
    assert report1.has_previous is False

    report2 = track(_Result(baseline=_Baseline(crashed=True)), "g", history_dir=tmp_path)
    assert report2.has_previous is True
    assert report2.is_regression is True

    assert len(load_history("g", history_dir=tmp_path)) == 2


def test_regression_flag_and_report_are_plain_dataclasses() -> None:
    flag = RegressionFlag(severity="info", summary="x")
    assert flag.severity == "info"
    report = RegressionReport(has_previous=False, previous_timestamp=None, flags=[flag])
    d = report.to_json()
    assert d["flags"][0]["summary"] == "x"


def test_report_to_json_includes_is_regression_property() -> None:
    # is_regression is a @property, not a dataclass field — asdict() alone
    # would silently drop it from the API response. Guard the fix.
    report = RegressionReport(
        has_previous=True, previous_timestamp=1.0,
        flags=[RegressionFlag("regression", "x")],
    )
    d = report.to_json()
    assert d["is_regression"] is True


def test_rates_use_perturbations_run_not_pattern_catalogue():
    """max_perturbations=4 on a 14-pattern schema: 2 crashes is 50%, not 14%."""
    from drift.regression import RunSnapshot, compare

    def snap(n_crashed, ts):
        return RunSnapshot(
            graph_signature="g", timestamp=ts, intensity="light", patterns_total=14,
            n_crashed=n_crashed, n_diverged=0, n_unchanged=4 - n_crashed,
            n_judge_findings=0, n_coordination_findings=0, n_perturbations=4,
        )

    report = compare(snap(2, 2.0), [snap(0, 1.0)])
    assert abs(report.crash_rate_delta - 0.5) < 1e-9
    assert any("crash rate up +50%" in f.summary for f in report.flags)


def test_old_snapshots_without_new_fields_still_load_and_compare():
    from drift.regression import RunSnapshot, compare

    old = RunSnapshot.from_json({
        "graph_signature": "g", "timestamp": 1.0, "intensity": "light", "patterns_total": 10,
        "n_crashed": 1, "n_diverged": 0, "n_unchanged": 9, "n_judge_findings": 0,
        "n_coordination_findings": 0,
    })
    assert old.n_perturbations == 0 and old.rate_denominator == 10 and old.perturbation_rollouts == 1
    new = RunSnapshot(
        graph_signature="g", timestamp=2.0, intensity="light", patterns_total=10,
        n_crashed=1, n_diverged=0, n_unchanged=9, n_judge_findings=0, n_coordination_findings=0,
        n_perturbations=10, perturbation_rollouts=3, n_intermittent=2,
    )
    report = compare(new, [old])
    assert any("2 perturbation(s) now give different outcomes across 3 rollouts" in f.summary for f in report.flags)
    assert any("rollouts per perturbation changed 1 → 3" in f.summary for f in report.flags)


def test_snapshot_from_result_records_rollout_fields():
    from drift.adapters.langgraph import drift_test
    from drift.regression import snapshot_from_result

    class _G:
        def invoke(self, state, *a, **k):
            return {**state, "reply": "ok"}

    res = drift_test(graph=_G(), initial_state={"query": "q", "reply": "", "n": 1},
                     intensity="light", seed=1, perturbation_rollouts=2)
    snap = snapshot_from_result(res, "stub")
    assert snap.n_perturbations == len(res.perturbations)
    assert snap.perturbation_rollouts == 2 and snap.n_intermittent == 0
    assert snap.rate_denominator == len(res.perturbations)
