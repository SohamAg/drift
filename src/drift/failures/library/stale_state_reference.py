"""Detector: agent references an entity that was closed/removed earlier.

Failure pattern: agent B references entity X at step N; X had `status=closed`
(or `resolved`, `archived`, `deleted`, `cancelled`, ...) written by some
earlier step K < N and was not reopened between K and N. B is operating
on stale state — the entity existed, but its current lifecycle position
means B should not be acting on it.

Sources:
- MAST 1.5 — parallel-agent state race / staleness
- Cognition "Don't Build Multi-Agents" open problem #2 — cross-sibling
  discovery sharing; parallel branches with divergent state views
- MAST 4.1 — termination-order errors

Why it matters: distinct from `hallucinated_reference` (entity never
existed). Here the entity DID exist and the reference LOOKS valid — but
the entity's lifecycle has moved on. Damage compounds: the acting agent
believes it's operating on a live entity; downstream nodes trust its
output; state ends up representing operations against a closed record.

Common in LangGraph `Send()` fan-outs and in supervisor patterns where
one branch closes a case and another branch continues processing it in
parallel without seeing the close.

Detection strategy (structured / adapter trace):

  1. Walk the trace. Track a per-entity lifecycle: entities gain
     `closed_at=(step, agent)` when a step's structured update carries an
     entity id plus a closure-signal status token. `reopened_at` clears
     the closure.

  2. On each step, if any structured entity id in the step's update is
     currently closed (closed at step K < this step) AND this step does
     not reopen it, fire — the agent is targeting a closed entity.

  3. Fire once per (entity_id, referencing_step) pair. Multiple stale
     references on the same trace to the same entity all fire (each is a
     distinct downstream propagation event).

Detection strategy (raw-text variant):

  Precision-loose fallback: pair id mentions with close/open verbs in
  speaker utterances. Fire when a later speaker's utterance mentions an
  id whose most recent context was a "closed"-style signal from a prior
  speaker. Recall-conservative — false negatives are honest here.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from drift.failures.base import FailureRecord
from drift.failures.library.base import (
    CoordinationDetectorContext,
    TraceStep,
)

NAME = "stale_state_reference"
SUMMARY = (
    "Agent referenced an entity that had been closed/removed in an earlier "
    "step and not reopened."
)
MAST_MODES = ["1.5", "4.1"]  # parallel-agent state race; termination order
SOURCE = (
    "MAST 1.5 (parallel-agent state race); Cognition open problem #2 "
    "(cross-sibling discovery sharing); MAST 4.1 (termination-order errors)"
)


# ---------------------------------------------------------------------------
# Vocabulary — smart defaults; overridable via detect() kwargs.
# ---------------------------------------------------------------------------

DEFAULT_ENTITY_ID_FIELDS: frozenset[str] = frozenset({
    "case_id", "ticket_id", "pr_id", "task_id", "order_id", "issue_id",
    "entity_id", "target_case_id", "target_id", "target_ticket_id",
    "target_task_id", "context_id", "id", "acting_on", "operating_on",
    "processing_id",
})

DEFAULT_STATUS_FIELDS: frozenset[str] = frozenset({
    "status", "state", "lifecycle", "case_status", "ticket_status",
    "resolution", "outcome",
})

CLOSED_STATUS_TOKENS: frozenset[str] = frozenset({
    "closed", "close", "resolved", "resolve", "done", "completed",
    "complete", "finished", "finish", "archived", "archive", "cancelled",
    "canceled", "cancel", "deleted", "delete", "removed", "remove",
    "terminated", "terminate", "abandoned", "abandon", "shipped",
    "shut", "shutdown", "expired",
})

REOPEN_STATUS_TOKENS: frozenset[str] = frozenset({
    "reopened", "reopen", "reactivated", "reactivate", "restored",
    "restore", "unclosed", "reissued", "reissue", "revived", "revive",
    "escalated_back",
})


def _canonicalize_id(id_text: str) -> str:
    return id_text.strip().lower().replace("_", "-").replace(" ", "-")


def _classify_status(text: str) -> str:
    """Return 'closed', 'reopened', or '' for a status-field value."""
    t = str(text).strip().lower()
    if t in CLOSED_STATUS_TOKENS:
        return "closed"
    if t in REOPEN_STATUS_TOKENS:
        return "reopened"
    return ""


def _extract_entity_ids(
    update: dict[str, Any],
    id_fields: frozenset[str],
) -> list[str]:
    """Extract canonical entity ids from structured update fields.

    Walks nested dicts + lists to find id-shaped fields. Preserves
    encounter order so callers can prefer the first id when needed.
    """
    out: list[str] = []
    seen: set[str] = set()

    def _walk(v: Any, key_hint: str = "", depth: int = 0) -> None:
        if depth > 8:
            return
        if isinstance(v, dict):
            for k, sub in v.items():
                _walk(sub, key_hint=str(k), depth=depth + 1)
        elif isinstance(v, (list, tuple, set)):
            for sub in v:
                _walk(sub, key_hint=key_hint, depth=depth + 1)
        elif isinstance(v, str):
            if key_hint.lower() in id_fields and v.strip():
                cid = _canonicalize_id(v)
                if cid not in seen:
                    seen.add(cid)
                    out.append(cid)

    _walk(update)
    return out


def _extract_status_updates(
    update: dict[str, Any],
    status_fields: frozenset[str],
) -> list[tuple[str, str]]:
    """Extract (status_class, raw_value) pairs for status-shaped fields.

    Returns list of ('closed' | 'reopened', raw_value) — the raw value
    is preserved for evidence strings. Neutral/unknown statuses are dropped.
    """
    out: list[tuple[str, str]] = []

    def _walk(v: Any, key_hint: str = "", depth: int = 0) -> None:
        if depth > 8:
            return
        if isinstance(v, dict):
            for k, sub in v.items():
                _walk(sub, key_hint=str(k), depth=depth + 1)
        elif isinstance(v, (list, tuple, set)):
            for sub in v:
                _walk(sub, key_hint=key_hint, depth=depth + 1)
        elif isinstance(v, str):
            if key_hint.lower() in status_fields:
                cls = _classify_status(v)
                if cls:
                    out.append((cls, v))

    _walk(update)
    return out


# ---------------------------------------------------------------------------
# Structured detector
# ---------------------------------------------------------------------------


def detect(
    ctx: CoordinationDetectorContext,
    *,
    entity_id_fields: Iterable[str] | None = None,
    status_fields: Iterable[str] | None = None,
) -> list[FailureRecord]:
    """Fire on each stale reference (entity id + referencing step)."""
    efields = frozenset(f.lower() for f in (entity_id_fields or DEFAULT_ENTITY_ID_FIELDS))
    sfields = frozenset(f.lower() for f in (status_fields or DEFAULT_STATUS_FIELDS))

    closed_at: dict[str, tuple[int, str, str]] = {}  # eid -> (step, agent, evidence)
    findings: list[FailureRecord] = []
    already_flagged: set[tuple[str, int]] = set()

    for step in ctx.steps:
        step_eids = _extract_entity_ids(step.update, efields)
        step_statuses = _extract_status_updates(step.update, sfields)

        # 1. Process REOPEN signals first — a step that reopens then references
        #    the entity is legitimate.
        reopened_this_step: set[str] = set()
        if step_statuses and step_eids:
            for cls, raw in step_statuses:
                if cls == "reopened":
                    for eid in step_eids:
                        closed_at.pop(eid, None)
                        reopened_this_step.add(eid)

        # 2. Check for stale references. Fire when a structured entity id in
        #    this step's update is currently closed AND wasn't reopened this
        #    same step.
        for eid in step_eids:
            if eid in reopened_this_step:
                continue
            if eid not in closed_at:
                continue
            closed_step, closed_by, closed_evidence = closed_at[eid]
            if closed_step >= step.step:
                continue  # same-step close; agent may be describing the close
            key = (eid, step.step)
            if key in already_flagged:
                continue
            already_flagged.add(key)
            agents = sorted({closed_by, step.agent} - {""})
            findings.append(FailureRecord(
                timestep=step.step,
                failure_type=NAME,
                agents_involved=agents,
                evidence_action_ids=[],
                summary=(
                    f"agent {step.agent!r} referenced entity {eid!r} at step "
                    f"{step.step}, but it was closed at step {closed_step} by "
                    f"{closed_by!r} (status={closed_evidence!r})"
                ),
                snapshot_timestep=step.step,
            ))

        # 3. Now process NEW closures — these take effect starting NEXT step.
        if step_statuses and step_eids:
            for cls, raw in step_statuses:
                if cls == "closed":
                    for eid in step_eids:
                        # If reopened this step, ignore closures on it too —
                        # net effect is "still open" for our purposes.
                        if eid in reopened_this_step:
                            continue
                        # Don't overwrite an earlier closure — the first
                        # closure is the most diagnostic origin.
                        closed_at.setdefault(eid, (step.step, step.agent, raw))

    return findings


# ---------------------------------------------------------------------------
# Raw-text variant (precision-loose)
# ---------------------------------------------------------------------------


_SPEAKER_LINE = re.compile(
    r"^\s*(?:\[|\<)?\s*"
    r"(?P<name>[A-Za-z][A-Za-z0-9_ ]{1,40}?)"
    r"\s*(?:\]|\>)?\s*:\s*"
    r"(?P<body>.*?)\s*$",
    re.M,
)

_ID_PATTERNS = [
    re.compile(r"\b([A-Z]{2,10}-\d{1,10})\b"),
    re.compile(r"(?<![\w#])(#\d{2,10})\b"),
    re.compile(
        r"\b(case|ticket|pr|issue|order|task|entity|session|job)"
        r"[\s_\-]*([A-Za-z0-9]*\d[A-Za-z0-9]*)\b",
        re.I,
    ),
]

_CLOSE_VERB_RE = re.compile(
    r"\b(?:closed|closing|resolved|resolving|archived|archiving|"
    r"deleted|deleting|removed|removing|cancelled|canceling|"
    r"completed|completing|shipped|shutting|terminated|"
    r"marked (?:as )?(?:closed|resolved|done|complete|archived))\b",
    re.I,
)

_REOPEN_VERB_RE = re.compile(
    r"\b(?:reopened|reopening|reactivated|reactivating|restored|"
    r"restoring|unclosed|revived|reviving)\b",
    re.I,
)


def _ids_in_text(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for pat in _ID_PATTERNS:
        for m in pat.finditer(text or ""):
            if len(m.groups()) == 2:
                token = f"{m.group(1)}-{m.group(2)}"
            else:
                token = m.group(1) or m.group(0)
            seen.setdefault(_canonicalize_id(token), None)
    return list(seen.keys())


def detect_from_text(transcript: str) -> list[FailureRecord]:
    """Fire when an id appears in a later utterance after a close verb, unless
    a reopen verb appeared between.

    Recall-conservative: real staleness expressed via implication (`we'll
    handle it next week`) rather than a close verb will be missed. False
    positives on "we mentioned that CASE-42 was closed" are possible.
    """
    if not transcript:
        return []
    findings: list[FailureRecord] = []
    closed_at: dict[str, tuple[int, str]] = {}
    already_flagged: set[tuple[str, int]] = set()

    for line_index, m in enumerate(_SPEAKER_LINE.finditer(transcript)):
        speaker = m.group("name").strip()
        body = m.group("body") or ""
        ids_here = _ids_in_text(body)
        has_close_verb = bool(_CLOSE_VERB_RE.search(body))
        has_reopen_verb = bool(_REOPEN_VERB_RE.search(body))

        # Reopen wins over close within the same utterance.
        if has_reopen_verb:
            for eid in ids_here:
                closed_at.pop(eid, None)

        # Check for stale references — id present, id already closed, no
        # reopen on this utterance.
        for eid in ids_here:
            if has_reopen_verb:
                continue
            if eid in closed_at:
                closed_line, closed_by = closed_at[eid]
                if closed_line >= line_index:
                    continue
                key = (eid, line_index)
                if key in already_flagged:
                    continue
                already_flagged.add(key)
                agents = sorted({closed_by, speaker})
                findings.append(FailureRecord(
                    timestep=0,
                    failure_type=NAME,
                    agents_involved=agents,
                    evidence_action_ids=[],
                    summary=(
                        f"text scan: speaker {speaker!r} referenced id "
                        f"{eid!r} after speaker {closed_by!r} indicated it "
                        f"was closed"
                    ),
                    snapshot_timestep=0,
                ))

        # Record new closures.
        if has_close_verb and not has_reopen_verb:
            for eid in ids_here:
                closed_at.setdefault(eid, (line_index, speaker))

    return findings


__all__ = [
    "CLOSED_STATUS_TOKENS",
    "DEFAULT_ENTITY_ID_FIELDS",
    "DEFAULT_STATUS_FIELDS",
    "MAST_MODES",
    "NAME",
    "REOPEN_STATUS_TOKENS",
    "SOURCE",
    "SUMMARY",
    "detect",
    "detect_from_text",
]
