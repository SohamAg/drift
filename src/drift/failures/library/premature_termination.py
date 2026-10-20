"""Detector: an agent acted without a required follow-up agent acting
afterward.

Failure: the user declares `required_followups={"trigger_agent":
"required_agent"}` — "whenever trigger_agent acts, required_agent must act
within N steps afterward." When trigger_agent acts and required_agent never
follows within the window, that's an orphaned action: work that should have
triggered a next step (escalation, verification, cleanup) but didn't.

Generic form of two native-sim patterns retired in the 2026-06-29 cleanup:
`silent_remediation` (remediate without communicate) and `comms_lag`
(triage without communicate) — this detector is the trace-shape-agnostic
version of "action X happened, required action Y didn't."

PROVISIONAL CONFIG: `required_followups` is a single flat dict, not a
general rule DSL (multiple required follow-ups per trigger, conditional
rules, etc. aren't expressible yet — see this project's user-guideline
DSL, not yet built). Without it declared, this detector always returns []
— it has no way to guess what "required" means for your graph.
"""
from __future__ import annotations

from drift.failures.base import FailureRecord
from drift.failures.library.base import CoordinationDetectorContext

NAME = "premature_termination"
SUMMARY = "An agent acted but the required follow-up agent never acted afterward."
# "4.1" per this project's detector_mining_findings.md ("MAST 4.1,
# termination errors") — kept consistent with that doc's citation.
MAST_MODES = ["4.1"]
SOURCE = "MAST 4.1 (termination errors)"

DEFAULT_WITHIN_STEPS = 5


def detect(
    ctx: CoordinationDetectorContext, *, within_steps: int = DEFAULT_WITHIN_STEPS
) -> list[FailureRecord]:
    pairs = ctx.required_followups
    if not pairs:
        return []

    steps = ctx.steps
    out: list[FailureRecord] = []
    flagged_triggers: set[str] = set()
    for i, step in enumerate(steps):
        trigger = step.agent
        if trigger not in pairs or trigger in flagged_triggers:
            continue
        required = pairs[trigger]
        window = steps[i + 1: i + 1 + within_steps]
        if any(s.agent == required for s in window):
            continue
        out.append(FailureRecord(
            timestep=step.step,
            failure_type=NAME,
            agents_involved=[trigger],
            evidence_action_ids=[],
            summary=(
                f"agent {trigger!r} acted at step {step.step} but the required "
                f"follow-up {required!r} did not occur within {within_steps} "
                f"step(s) afterward"
            ),
            snapshot_timestep=step.step,
        ))
        flagged_triggers.add(trigger)  # one finding per trigger agent, not per occurrence
    return out


def detect_from_text(transcript: str) -> list[FailureRecord]:
    """No-op by design: required_followups is caller-supplied config with
    no channel in the raw-text path (run_all_on_text calls this with only
    a transcript, no config), so there's nothing this variant could check
    without guessing at structure nobody declared. Returns [] always."""
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
