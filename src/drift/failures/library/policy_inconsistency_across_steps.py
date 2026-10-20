"""Detector: an agent cites a stale version of a config/policy field the
state has since moved past.

Failure: the user declares `policy_field="policy_version"` (or whatever
the state's version field is named). The detector tracks that field's
authoritative value as it's written across the trace — the most recent
write is "current." If a later step's text content (rationale or any
string field) mentions a PRIOR value of that field, that's a stale
citation: the agent is acting on — or at least talking about — a version
of the policy/config that's no longer in effect. MAST 1.5 (configuration
drift), per this project's detector-mining notes.

PROVISIONAL CONFIG: `policy_field` is a single field-name hint, not a
general schema for "what counts as policy." Without it declared, this
detector always returns [] — it has no way to know which field is the
version key for your graph.
"""
from __future__ import annotations

from drift.failures.base import FailureRecord
from drift.failures.library.base import (
    CoordinationDetectorContext,
    any_token,
    lowercased_text_blob,
)

NAME = "policy_inconsistency_across_steps"
SUMMARY = "An agent referenced a stale value of a tracked policy/config field."
MAST_MODES = ["1.5"]  # configuration drift, per detector_mining_findings.md
SOURCE = "MAST 1.5 (configuration drift)"

# Below this length a "version value" is too likely to collide with
# unrelated text (e.g. a bare "1" or "v1" inside an unrelated word) to be
# a trustworthy citation match — require something more specific.
_MIN_VERSION_LEN = 2


def detect(ctx: CoordinationDetectorContext) -> list[FailureRecord]:
    field_name = ctx.policy_field
    if not field_name:
        return []

    seen_versions: list[str] = []
    current: str | None = None
    initial_value = (ctx.initial_state or {}).get(field_name)
    if initial_value is not None:
        current = str(initial_value)
        seen_versions.append(current)

    out: list[FailureRecord] = []
    flagged_agents: set[str] = set()
    for step in ctx.steps:
        if field_name in step.update:
            new_value = str(step.update[field_name])
            if new_value != current:
                current = new_value
                if current not in seen_versions:
                    seen_versions.append(current)
            continue  # the step that DEFINES a version isn't itself a violator

        if current is None or step.agent in flagged_agents:
            continue

        stale_candidates = [
            v for v in seen_versions if v != current and len(v) >= _MIN_VERSION_LEN
        ]
        if not stale_candidates:
            continue

        blob = lowercased_text_blob(step.update)
        if step.rationale:
            blob = f"{blob} {step.rationale.lower()}"
        tok = any_token(blob, stale_candidates)
        if tok is None:
            continue

        out.append(FailureRecord(
            timestep=step.step,
            failure_type=NAME,
            agents_involved=[step.agent],
            evidence_action_ids=[],
            summary=(
                f"agent {step.agent!r} referenced stale {field_name}={tok!r} "
                f"at step {step.step}; current value is {current!r}"
            ),
            snapshot_timestep=step.step,
        ))
        flagged_agents.add(step.agent)
    return out


def detect_from_text(transcript: str) -> list[FailureRecord]:
    """No-op by design: policy_field is caller-supplied config with no
    channel in the raw-text path (run_all_on_text calls this with only a
    transcript, no config), so there's nothing to check without guessing
    at a field name nobody declared. Returns [] always."""
    return []


__all__ = [
    "MAST_MODES",
    "NAME",
    "SOURCE",
    "SUMMARY",
    "detect",
    "detect_from_text",
]
