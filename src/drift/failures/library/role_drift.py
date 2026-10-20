"""Detector: an agent performs an action outside its declared or inferred role.

Failure: an agent whose role is "verifier" (reviewer/critic/judge) performs
an irreversible EXECUTION action — merges, deploys, closes, pays, deletes —
instead of rendering a verdict. Or the inverse: an agent whose role is
"executor" (worker/actor/implementer) renders a VERIFICATION verdict
(approved/rejected/verified) instead of just doing the work. MAST 1.2
(Disobey Role Specification) — both are the same underlying failure in
opposite directions: the system's separation-of-concerns broke down and one
agent started doing another agent's job.

Why it matters: verify-then-execute is the whole point of having a separate
verifier agent. If the verifier itself executes, there's no check left — a
bad decision ships with nothing that would have caught it. The inverse
(executor renders verdicts) means a "verified" claim in the trace didn't
actually come from anything resembling review.

Role resolution: explicit declaration (`coordination_roles={"agent_name":
"verifier"}`, same dict already used by fork-edit-replay) takes precedence;
falls back to a name-pattern match (`verifier`, `review`, `critic`, `judge`
/ `execut`, `worker`, `actor`, `implement`) so this fires with zero config
on conventionally-named agents.

Fires at most once per agent (the first violation), not once per
occurrence — a verifier that executes on every step is one finding, not N.
"""
from __future__ import annotations

import re

from drift.failures.base import FailureRecord
from drift.failures.library.base import (
    CoordinationDetectorContext,
    TraceStep,
    any_token,
    lowercased_text_blob,
)

NAME = "role_drift"
SUMMARY = "An agent performed an action outside what its role should do."
MAST_MODES = ["1.2"]  # Disobey Role Specification
SOURCE = "MAST 1.2 (Disobey Role Specification)"

_VERIFIER_NAME_PATTERN = re.compile(r"verif|review|critic|judge", re.I)
_EXECUTOR_NAME_PATTERN = re.compile(r"execut|worker|\bactor\b|implement", re.I)

# A verifier/reviewer rendering these means it stopped checking and started
# acting — irreversible execution verbs, not verdicts.
_VERIFIER_FORBIDDEN_VERBS = (
    "merged", "deployed", "executed", "closed", "resolved", "shipped",
    "committed", "paid", "refunded", "cancelled", "deleted",
)
# An executor/worker rendering these means it's claiming review authority
# nothing upstream actually exercised.
_EXECUTOR_FORBIDDEN_VERBS = (
    "approved", "rejected", "verified", "validated", "lgtm",
)

_ROLE_RULES: dict[str, tuple[re.Pattern[str], tuple[str, ...]]] = {
    "verifier": (_VERIFIER_NAME_PATTERN, _VERIFIER_FORBIDDEN_VERBS),
    "executor": (_EXECUTOR_NAME_PATTERN, _EXECUTOR_FORBIDDEN_VERBS),
}


def _blob_for(step: TraceStep) -> str:
    blob = lowercased_text_blob(step.update)
    if step.rationale:
        blob = f"{blob} {step.rationale.lower()}"
    return blob


def _candidate_rules(
    agent: str, roles_by_agent: dict[str, str]
) -> list[tuple[str, tuple[re.Pattern[str], tuple[str, ...]]]]:
    """Which role rule(s) apply to this agent.

    A DECLARED role is authoritative and exclusive — an agent explicitly
    declared "verifier" is never also checked against the executor rule,
    even if its name happens to match the executor name pattern (e.g. a
    verifier literally named "worker_bot"). Only when nothing is declared
    do we fall back to matching the agent's name against every rule's
    pattern — role_matches()'s per-rule fallback isn't enough here since
    we check multiple rules per agent, not one.
    """
    declared = roles_by_agent.get(agent, "").lower()
    if declared:
        rule = _ROLE_RULES.get(declared)
        return [(declared, rule)] if rule else []
    return [
        (role_name, rule)
        for role_name, rule in _ROLE_RULES.items()
        if rule[0].search(agent or "")
    ]


def detect(ctx: CoordinationDetectorContext) -> list[FailureRecord]:
    out: list[FailureRecord] = []
    flagged_agents: set[str] = set()
    for step in ctx.steps:
        if step.agent in flagged_agents:
            continue
        blob = _blob_for(step)
        if not blob:
            continue
        for role_name, (pattern, forbidden) in _candidate_rules(step.agent, ctx.roles_by_agent):
            tok = any_token(blob, forbidden)
            if tok is None:
                continue
            out.append(FailureRecord(
                timestep=step.step,
                failure_type=NAME,
                agents_involved=[step.agent],
                evidence_action_ids=[],
                summary=(
                    f"agent {step.agent!r} (role={role_name}) performed a "
                    f"{tok!r}-shaped action at step {step.step}, outside what a "
                    f"{role_name} should do"
                ),
                snapshot_timestep=step.step,
            ))
            flagged_agents.add(step.agent)
            break  # one rule match is enough for this agent
    return out


# ---------------------------------------------------------------------------
# Raw-text variant — for MAST validation.
# ---------------------------------------------------------------------------

_SPEAKER_LINE = re.compile(
    r"^[\[\<]?\s*(?P<name>[A-Za-z][A-Za-z0-9_ ]{1,40}?)\s*[:\]>]\s*(?P<body>.*)$",
    re.M,
)


def detect_from_text(transcript: str) -> list[FailureRecord]:
    """Text-only variant: per speaker-labelled line, resolve role from the
    speaker's NAME only (no declared-roles channel in raw text) and check
    for the same forbidden-verb vocabulary. Lower precision than the
    structured path by design — ships for apples-to-apples MAST F1."""
    if not transcript:
        return []
    out: list[FailureRecord] = []
    flagged_names: set[str] = set()
    for m in _SPEAKER_LINE.finditer(transcript):
        name = re.sub(r"\s+", " ", m.group("name").strip())
        if name.lower() in flagged_names:
            continue
        body = m.group("body").lower()
        if not body:
            continue
        for role_name, (pattern, forbidden) in _ROLE_RULES.items():
            if not pattern.search(name):
                continue
            tok = any_token(body, forbidden)
            if tok is None:
                continue
            out.append(FailureRecord(
                timestep=0,
                failure_type=NAME,
                agents_involved=[name],
                evidence_action_ids=[],
                summary=(
                    f"text scan: speaker {name!r} (name matches {role_name}) used "
                    f"{tok!r}-shaped language, outside what a {role_name} should do"
                ),
                snapshot_timestep=0,
            ))
            flagged_names.add(name.lower())
            break
    return out


__all__ = [
    "MAST_MODES",
    "NAME",
    "SOURCE",
    "SUMMARY",
    "detect",
    "detect_from_text",
]
