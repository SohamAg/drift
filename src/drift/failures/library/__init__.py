"""Coordination-failure detector library.

Curated, source-cited detectors for named MAS coordination failures.

Each detector module exposes:
  - `NAME` (str): unique failure_type the detector emits
  - `SUMMARY` (str): one-line description
  - `MAST_MODES` (list[str]): MAST mode IDs this detector targets (for the
    validation runner to align predictions with ground truth)
  - `SOURCE` (str): primary source citation
  - `detect(ctx)`: structured detector over `CoordinationDetectorContext`
  - `detect_from_text(transcript)`: raw-text variant for unstructured traces

To run the whole library over an adapter trace:

    from drift.failures.library import run_all_on_trace
    findings = run_all_on_trace(trace, baseline_state=..., initial_state=...)

To run text-only over a raw transcript (e.g. MAST validation):

    from drift.failures.library import run_all_on_text
    findings = run_all_on_text(transcript_string)
"""
from __future__ import annotations

from typing import Any

from drift.failures.base import FailureRecord
from drift.failures.library import (
    contradictory_decisions,
    hallucinated_reference,
    infinite_handoff,
    subagent_fanout_excess,
    verifier_always_approves,
)
from drift.failures.library.base import (
    CoordinationDetector,
    CoordinationDetectorContext,
    TraceStep,
    from_adapter_trace,
    from_native,
)


ALL_DETECTORS = [
    verifier_always_approves,
    infinite_handoff,
    subagent_fanout_excess,
    hallucinated_reference,
    contradictory_decisions,
]


def run_all_on_trace(
    trace: list[dict],
    *,
    initial_state: dict | None = None,
    baseline_state: dict | None = None,
    roles_by_agent: dict[str, str] | None = None,
) -> list[FailureRecord]:
    """Run every library detector over an adapter trace, return concatenated findings."""
    ctx = from_adapter_trace(
        trace,
        initial_state=initial_state,
        baseline_state=baseline_state,
        roles_by_agent=roles_by_agent,
    )
    out: list[FailureRecord] = []
    for module in ALL_DETECTORS:
        try:
            out.extend(module.detect(ctx))
        except Exception as exc:  # noqa: BLE001
            # Library detectors must never crash the calling adapter run. If
            # one blows up on a malformed trace, swallow + continue so the
            # other detectors still fire and the adapter still returns.
            out.append(FailureRecord(
                timestep=0,
                failure_type=f"{module.NAME}:error",
                agents_involved=[],
                evidence_action_ids=[],
                summary=f"detector crashed: {type(exc).__name__}: {exc}",
                snapshot_timestep=0,
            ))
    return out


def run_all_on_text(transcript: str) -> list[FailureRecord]:
    """Run every library detector's text-variant over an unstructured transcript."""
    out: list[FailureRecord] = []
    for module in ALL_DETECTORS:
        try:
            out.extend(module.detect_from_text(transcript))
        except Exception as exc:  # noqa: BLE001
            out.append(FailureRecord(
                timestep=0,
                failure_type=f"{module.NAME}:error",
                agents_involved=[],
                evidence_action_ids=[],
                summary=f"detector crashed: {type(exc).__name__}: {exc}",
                snapshot_timestep=0,
            ))
    return out


def detector_names() -> list[str]:
    return [m.NAME for m in ALL_DETECTORS]


def mast_mode_map() -> dict[str, list[str]]:
    """Detector name -> list of MAST mode IDs it targets."""
    return {m.NAME: list(m.MAST_MODES) for m in ALL_DETECTORS}


__all__ = [
    "ALL_DETECTORS",
    "CoordinationDetector",
    "CoordinationDetectorContext",
    "TraceStep",
    "contradictory_decisions",
    "detector_names",
    "from_adapter_trace",
    "from_native",
    "hallucinated_reference",
    "infinite_handoff",
    "mast_mode_map",
    "run_all_on_text",
    "run_all_on_trace",
    "subagent_fanout_excess",
    "verifier_always_approves",
]
