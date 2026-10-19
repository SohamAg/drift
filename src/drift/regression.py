"""Regression tracking — diff one drift_test run against the same graph's history.

Drift today answers "is this graph robust right now" (one-shot). This module
adds the over-time question: "did the LAST change make it worse" — the thing
flagged in FUTURE_DIRECTIONS.md as the single highest-value missing piece,
because it's what actually catches "I tweaked a prompt and broke something
subtle" instead of requiring the user to notice on their own.

Each `drift_test` / `drift_test_async` call can be condensed into a
`RunSnapshot` (small, comparable stats — never the full trace, so history
accumulates cheaply across hundreds of runs) and appended to a per-graph
JSONL history file. `compare()` diffs the newest snapshot against the most
recent prior one and flags what moved: new coordination-finding types,
finding types that stopped firing, crash/divergence rate deltas, and
perturbation runtime deltas.

Deliberately NOT built here (out of scope for this pass):
  - cross-run caching / skip-unchanged (FUTURE_DIRECTIONS calls this a
    separate "side effect" win, not the regression signal itself)
  - token/cost tracking (AdapterResult doesn't capture token counts yet)
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_HISTORY_DIR = Path("results") / "_history"

# Thresholds below which a rate/duration change is noise, not signal. Chosen
# to be well above single-run jitter (one more/fewer crash out of ~10-25
# perturbations is a few percentage points) without requiring calibration.
CRASH_RATE_REGRESSION_THRESHOLD = 0.05
DIVERGE_RATE_REGRESSION_THRESHOLD = 0.10
DURATION_REGRESSION_THRESHOLD_PCT = 0.50


@dataclass
class RunSnapshot:
    """Condensed, JSON-serializable stats from one AdapterResult.

    `graph_signature` is caller-supplied and must be stable across runs of
    "the same graph" for history to mean anything — e.g. the bundled
    `graph_name` ("ticket_triage") or `"{module}.{build_fn}"` for an
    imported graph. Different signatures never get compared against each
    other.
    """

    graph_signature: str
    timestamp: float
    intensity: str
    patterns_total: int
    n_crashed: int
    n_diverged: int
    n_unchanged: int
    n_judge_findings: int
    n_coordination_findings: int
    coordination_finding_types: dict[str, int] = field(default_factory=dict)
    judge_calls_used: int = 0
    baseline_duration_s: float = 0.0
    mean_perturbation_duration_s: float = 0.0
    baseline_crashed: bool = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "RunSnapshot":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def snapshot_from_result(result: Any, graph_signature: str) -> RunSnapshot:
    """Build a RunSnapshot from an AdapterResult (duck-typed — any object
    exposing the same attributes/properties works, so tests don't need the
    real dataclass)."""
    finding_types: Counter[str] = Counter()
    for f in result.baseline.coordination_findings:
        finding_types[f.get("failure_type", "unknown")] += 1
    for p in result.perturbations:
        for f in p.coordination_findings:
            finding_types[f.get("failure_type", "unknown")] += 1

    durations = [p.duration_s for p in result.perturbations]
    mean_duration = sum(durations) / len(durations) if durations else 0.0

    return RunSnapshot(
        graph_signature=graph_signature,
        timestamp=time.time(),
        intensity=result.intensity,
        patterns_total=result.patterns_total,
        n_crashed=result.n_crashed,
        n_diverged=result.n_diverged,
        n_unchanged=result.n_unchanged,
        n_judge_findings=result.n_judge_findings,
        n_coordination_findings=result.n_coordination_findings,
        coordination_finding_types=dict(finding_types),
        judge_calls_used=result.judge_calls_used,
        baseline_duration_s=result.baseline.duration_s,
        mean_perturbation_duration_s=mean_duration,
        baseline_crashed=result.baseline.crashed,
    )


def _history_path(graph_signature: str, history_dir: Path = DEFAULT_HISTORY_DIR) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", graph_signature).strip("_") or "graph"
    return Path(history_dir) / f"{safe}.jsonl"


def load_history(
    graph_signature: str, history_dir: Path = DEFAULT_HISTORY_DIR
) -> list[RunSnapshot]:
    """All past snapshots for this graph, oldest first. Empty if never run before."""
    path = _history_path(graph_signature, history_dir)
    if not path.exists():
        return []
    out: list[RunSnapshot] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(RunSnapshot.from_json(json.loads(line)))
        except (json.JSONDecodeError, TypeError):
            continue
    out.sort(key=lambda s: s.timestamp)
    return out


def append_snapshot(snapshot: RunSnapshot, history_dir: Path = DEFAULT_HISTORY_DIR) -> Path:
    """Append one snapshot to its graph's history file. Append-only — history
    is never rewritten or pruned, since each line is a few hundred bytes and
    the comparison only ever needs the most recent prior entry."""
    path = _history_path(snapshot.graph_signature, history_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(snapshot.to_json(), default=str) + "\n")
    return path


@dataclass
class RegressionFlag:
    severity: str  # "crash" | "regression" | "info"
    summary: str


@dataclass
class RegressionReport:
    has_previous: bool
    previous_timestamp: float | None
    new_finding_types: list[str] = field(default_factory=list)
    resolved_finding_types: list[str] = field(default_factory=list)
    crash_rate_delta: float = 0.0
    diverge_rate_delta: float = 0.0
    judge_calls_delta: int = 0
    duration_delta_pct: float | None = None
    flags: list[RegressionFlag] = field(default_factory=list)

    @property
    def is_regression(self) -> bool:
        return any(f.severity in ("crash", "regression") for f in self.flags)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def compare(current: RunSnapshot, history: list[RunSnapshot]) -> RegressionReport:
    """Diff `current` against the most recent snapshot in `history` that's
    older than it. `history` should be the snapshots loaded BEFORE appending
    `current` — callers append after comparing so `current` never ends up
    comparing against itself."""
    prior = [s for s in history if s.timestamp < current.timestamp]
    if not prior:
        return RegressionReport(has_previous=False, previous_timestamp=None)
    previous = max(prior, key=lambda s: s.timestamp)

    cur_types = set(current.coordination_finding_types)
    prev_types = set(previous.coordination_finding_types)
    new_types = sorted(cur_types - prev_types)
    resolved_types = sorted(prev_types - cur_types)

    def _rate(n: int, total: int) -> float:
        return (n / total) if total else 0.0

    crash_delta = _rate(current.n_crashed, current.patterns_total) - _rate(
        previous.n_crashed, previous.patterns_total
    )
    diverge_delta = _rate(current.n_diverged, current.patterns_total) - _rate(
        previous.n_diverged, previous.patterns_total
    )
    judge_delta = current.judge_calls_used - previous.judge_calls_used

    duration_pct: float | None = None
    if previous.mean_perturbation_duration_s > 0:
        duration_pct = (
            current.mean_perturbation_duration_s - previous.mean_perturbation_duration_s
        ) / previous.mean_perturbation_duration_s

    flags: list[RegressionFlag] = []
    if current.baseline_crashed and not previous.baseline_crashed:
        flags.append(RegressionFlag("crash", "baseline now crashes — it didn't on the last run"))
    for t in new_types:
        flags.append(RegressionFlag("regression", f"new coordination finding type: {t}"))
    for t in resolved_types:
        flags.append(RegressionFlag("info", f"coordination finding type no longer firing: {t}"))
    if crash_delta > CRASH_RATE_REGRESSION_THRESHOLD:
        flags.append(RegressionFlag("regression", f"crash rate up {crash_delta:+.0%} vs last run"))
    if diverge_delta > DIVERGE_RATE_REGRESSION_THRESHOLD:
        flags.append(
            RegressionFlag("regression", f"divergence rate up {diverge_delta:+.0%} vs last run")
        )
    if duration_pct is not None and duration_pct > DURATION_REGRESSION_THRESHOLD_PCT:
        flags.append(
            RegressionFlag(
                "regression", f"mean perturbation runtime up {duration_pct:+.0%} vs last run"
            )
        )

    return RegressionReport(
        has_previous=True,
        previous_timestamp=previous.timestamp,
        new_finding_types=new_types,
        resolved_finding_types=resolved_types,
        crash_rate_delta=crash_delta,
        diverge_rate_delta=diverge_delta,
        judge_calls_delta=judge_delta,
        duration_delta_pct=duration_pct,
        flags=flags,
    )


def track(
    result: Any, graph_signature: str, history_dir: Path = DEFAULT_HISTORY_DIR
) -> RegressionReport:
    """One-call convenience: load history, compare, append the new snapshot,
    return the report. Call once per drift_test call you want tracked."""
    history = load_history(graph_signature, history_dir)
    current = snapshot_from_result(result, graph_signature)
    report = compare(current, history)
    append_snapshot(current, history_dir)
    return report
