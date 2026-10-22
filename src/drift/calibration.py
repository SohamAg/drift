"""Calibration — the proof, not just the vibe, that historical grounding
actually helps.

Runs a generated scenario simulation (a target built with
target_builder.generate_target_source(..., verdict_schema=...)) against
held-out historical events whose real outcomes are already known (see
drift.historical_grounding), and checks whether each run's structured
verdict direction (up/down/flat) matches what actually happened.
Without this, "grounded in historical data" is just a claim; this is
what makes it checkable.

Leave-one-out by construction: when scoring event i, the historical
analogues retrieved for grounding exclude event i itself — otherwise
"predicting" an event using itself as context would be circular, not a
real test.

This makes real LLM calls — one run per event, each run being the full
multi-agent graph (several LLM calls itself) plus one retrieval call if
grounding is on. For the 7-event seed dataset that's realistically
20-35 LLM calls total. Not free, not instant; callers should say so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from drift.historical_grounding import (
    DEFAULT_MODEL,
    HistoricalEvent,
    build_grounded_query,
    load_events,
    retrieve_similar_events,
)

_VALID_DIRECTIONS = ("up", "down", "flat")


class CalibrationError(ValueError):
    """Raised for setup problems (not enough events, bad graph callables)
    — never for a single event's run failing, which is captured per-
    result instead so one bad run doesn't block scoring the rest."""


@dataclass
class CalibrationResult:
    event: HistoricalEvent
    predicted_direction: str | None  # None if unparseable or the run errored
    predicted_magnitude_pct: float | None = None
    predicted_confidence: str | None = None
    matched: bool = False
    error: str | None = None  # set only if the run itself failed

    def to_json(self) -> dict[str, Any]:
        return {
            "event": self.event.to_json(),
            "predicted_direction": self.predicted_direction,
            "predicted_magnitude_pct": self.predicted_magnitude_pct,
            "predicted_confidence": self.predicted_confidence,
            "matched": self.matched,
            "error": self.error,
        }


@dataclass
class CalibrationReport:
    results: list[CalibrationResult] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_scored(self) -> int:
        """Results where a direction was actually predicted — excludes
        errored/unparseable runs, which is the honest denominator for
        accuracy (an accuracy computed over n_total would silently treat
        "the run crashed" the same as "it predicted wrong")."""
        return sum(1 for r in self.results if r.predicted_direction is not None)

    @property
    def n_matched(self) -> int:
        return sum(1 for r in self.results if r.matched)

    @property
    def accuracy(self) -> float | None:
        """Fraction of SCORED results whose direction matched reality.
        None (not 0.0) when nothing was scorable — a 0% accuracy and "we
        couldn't score anything" are different findings and must stay
        distinguishable."""
        if self.n_scored == 0:
            return None
        return self.n_matched / self.n_scored

    def to_json(self) -> dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_scored": self.n_scored,
            "n_matched": self.n_matched,
            "accuracy": self.accuracy,
            "results": [r.to_json() for r in self.results],
        }


async def run_calibration(
    build_fn: Callable[[str | None], Any],
    initial_state_fn: Callable[[str], dict],
    *,
    events: list[HistoricalEvent] | None = None,
    model: str | None = None,
    ground: bool = True,
    retrieval_k: int = 2,
    retrieval_model: str = DEFAULT_MODEL,
) -> CalibrationReport:
    """Score a verdict-producing target against held-out historical
    events. `build_fn`/`initial_state_fn` are a generated target's own
    build()/initial_state() — same callables Import-graph and Test Suite
    already invoke, so this works on anything saved via the Build tab's
    verdict_schema path, not a special calibration-only target shape.

    One event failing to run (a bad LLM response, a network error) is
    captured in that event's CalibrationResult.error, not raised — the
    report still covers every other event.
    """
    pool = events if events is not None else load_events()
    if len(pool) < 2:
        raise CalibrationError(
            "need at least 2 historical events to calibrate (one to score, "
            "at least one left over to ground it against)"
        )

    results: list[CalibrationResult] = []
    for event in pool:
        try:
            if ground:
                others = [e for e in pool if e is not event]
                analogues = await retrieve_similar_events(
                    event.description, events=others, k=retrieval_k, model=retrieval_model,
                )
                query = build_grounded_query(event.description, analogues)
            else:
                query = event.description

            graph = build_fn(model)
            state = initial_state_fn(query)
            final_state = await graph.ainvoke(state)
        except Exception as e:  # noqa: BLE001 — one event's failure must not sink the report
            results.append(CalibrationResult(
                event=event, predicted_direction=None,
                error=f"{type(e).__name__}: {e}",
            ))
            continue

        verdict = final_state.get("verdict") or {}
        direction = verdict.get("direction")
        if direction not in _VALID_DIRECTIONS:
            direction = None
        results.append(CalibrationResult(
            event=event,
            predicted_direction=direction,
            predicted_magnitude_pct=verdict.get("magnitude_pct"),
            predicted_confidence=verdict.get("confidence"),
            matched=bool(direction and direction == event.actual_direction),
        ))

    return CalibrationReport(results=results)
