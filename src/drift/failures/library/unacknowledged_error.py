"""Detector: an agent's output signals an error and nothing downstream
acknowledges, retries, or resolves it — the system just keeps going as
if nothing happened.

Failure: step K writes a structured error-shaped field (`error`,
`exception`, `traceback`, `failed=True`, `success=False`, …) with a
real (non-empty, non-False-as-absence) value. None of the next
`within_steps` steps — from any agent — contain a recognized
acknowledgment signal (a retry, an escalation, a fallback, or the same
error field being explicitly cleared). The error is simply dropped on
the floor; downstream agents proceed as though the step that failed
succeeded.

Anthropic's own engineering blog is the primary source here: silently
swallowing tool/agent errors and continuing is named as one of the
harder bugs to catch in production multi-agent systems, precisely
because nothing in the trace LOOKS wrong after the error — the system
just keeps producing output. MAST doesn't enumerate this failure mode
directly; it's tagged under 2.3 (Task derailment) as the closest fit,
since the system's forward progress after an unacknowledged error is
no longer grounded in a correct prior step.
"""
from __future__ import annotations

import re
from typing import Any

from drift.failures.base import FailureRecord
from drift.failures.library.base import (
    CoordinationDetectorContext,
    TraceStep,
    any_token,
    lowercased_text_blob,
)

NAME = "unacknowledged_error"
SUMMARY = "An agent signaled an error and nothing downstream acknowledged or resolved it."
MAST_MODES = ["2.3"]  # Task derailment — MAST doesn't enumerate silent error-swallowing directly
SOURCE = "Anthropic multi-agent research postmortem (errors silently treated as success)"

DEFAULT_WITHIN_STEPS = 3

# Structured fields whose presence with a real value signals an error.
ERROR_FIELD_NAMES = ("error", "exception", "traceback", "stack_trace", "failure_reason", "err")
# Boolean-flag fields: (field_name, value_that_signals_error).
ERROR_BOOL_FIELDS = (("failed", True), ("success", False), ("ok", False))

# Structured fields whose truthy presence counts as acknowledgment.
ACK_FIELD_NAMES = (
    "retry", "retrying", "retried", "recovered", "resolved", "resolved_error",
    "fallback", "handled_error", "error_handled", "acknowledged", "escalated",
)
# Free-text tokens that count as acknowledgment when no structured field is set.
ACK_TEXT_TOKENS = (
    "retry", "retrying", "retried", "recovered", "resolved", "fallback",
    "handling error", "handled the error", "escalating", "escalated",
)


def _extract_error_signal(update: dict[str, Any]) -> tuple[str, str] | None:
    """Return (field_name, display_value) for the first recognized error
    signal in `update`, or None."""
    for k in ERROR_FIELD_NAMES:
        if k in update:
            v = update[k]
            if v not in (None, "", False):
                return (k, repr(v)[:80])
    for field, bad_value in ERROR_BOOL_FIELDS:
        if update.get(field) == bad_value:
            return (field, repr(bad_value))
    return None


def _field_cleared(field_name: str, update: dict[str, Any]) -> bool:
    """True if `update` explicitly rewrites `field_name` to a non-error value."""
    if field_name not in update:
        return False
    if field_name in ERROR_FIELD_NAMES:
        return update[field_name] in (None, "", False)
    for field, bad_value in ERROR_BOOL_FIELDS:
        if field == field_name:
            return update[field_name] != bad_value
    return False


def _looks_like_acknowledgment(step: TraceStep) -> bool:
    for k in ACK_FIELD_NAMES:
        if step.update.get(k):
            return True
    blob = lowercased_text_blob(step.update) + " " + (step.rationale or "").lower()
    return any_token(blob, ACK_TEXT_TOKENS) is not None


def detect(
    ctx: CoordinationDetectorContext, *, within_steps: int = DEFAULT_WITHIN_STEPS
) -> list[FailureRecord]:
    steps = ctx.steps
    out: list[FailureRecord] = []
    for i, step in enumerate(steps):
        sig = _extract_error_signal(step.update)
        if sig is None:
            continue
        field_name, display_value = sig
        window = steps[i + 1: i + 1 + within_steps]
        acknowledged = any(
            _looks_like_acknowledgment(w) or _field_cleared(field_name, w.update)
            for w in window
        )
        if acknowledged:
            continue
        out.append(FailureRecord(
            timestep=step.step,
            failure_type=NAME,
            agents_involved=[step.agent],
            evidence_action_ids=[],
            summary=(
                f"agent {step.agent!r} signaled an error at step {step.step} "
                f"({field_name}={display_value}) with no acknowledgment, retry, "
                f"or resolution in the next {within_steps} step(s)"
            ),
            snapshot_timestep=step.step,
        ))
    return out


# ---------------------------------------------------------------------------
# Raw-text variant — for MAST validation.
# ---------------------------------------------------------------------------


_ERROR_LINE = re.compile(r"\b(error|exception|traceback|failed|failure)\b", re.I)


def detect_from_text(
    transcript: str, *, within_steps: int = DEFAULT_WITHIN_STEPS
) -> list[FailureRecord]:
    """Text-only variant: a line matching an error token with no
    acknowledgment token in the next `within_steps` lines. Looser than the
    structured path — no per-field tracking, just line-level proximity."""
    if not transcript:
        return []
    lines = [ln for ln in transcript.splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        if not _ERROR_LINE.search(line):
            continue
        window = lines[i + 1: i + 1 + within_steps]
        if any(any_token(w.lower(), ACK_TEXT_TOKENS) for w in window):
            continue
        return [FailureRecord(
            timestep=0,
            failure_type=NAME,
            agents_involved=[],
            evidence_action_ids=[],
            summary=f"text scan: error line with no acknowledgment nearby: {line.strip()[:140]!r}",
            snapshot_timestep=0,
        )]
    return []


__all__ = [
    "DEFAULT_WITHIN_STEPS",
    "MAST_MODES",
    "NAME",
    "SOURCE",
    "SUMMARY",
    "detect",
    "detect_from_text",
]
