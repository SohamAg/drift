"""Detector: same entity received contradictory verdicts in one trace.

Failure pattern: entity_id E received verdict `approve` (from agent A or in
an earlier step) AND verdict `reject` (from agent B or in a later step),
both in the same trace. Merges the native-sim `contradictory_refund`,
`contradictory_review`, `contradictory_diagnosis` detectors into one
generic pattern.

Sources:
- MAST 3.2 — specification ambiguity / conflicting decisions
- Anthropic engineering blog — "agent reaches different conclusions when
  re-prompted" pattern in multi-agent research system
- Cognition "Don't Build Multi-Agents" — Principle 2 "actions carry
  implicit decisions, and conflicting decisions carry bad results"

Why it matters: contradictory decisions on the same entity are the canonical
coordination-failure signature. In a well-designed MAS the ownership is
clear (one agent decides, others advise); when two agents produce opposing
verdicts and both land in state, downstream logic acts on whichever it
sees first. Silent data corruption at coordination time.

Detection strategy (structured / adapter trace):

  1. Walk trace, extracting (entity_id, verdict_polarity, evidence) triples.
     Verdicts come from either:
       - structured fields whose name suggests a decision (`verdict`,
         `decision`, `outcome`, `status`, `approval`, `resolution`, ...)
       - rationale text containing strong polarity tokens
     Entity ids come from adjacent structured fields (`case_id`,
     `ticket_id`, `target_case_id`, ...) or a `context` id.

  2. Bucket by entity_id. If any entity has BOTH a positive and a negative
     verdict recorded, fire once per contradicted entity.

  3. Fire once per entity — subsequent contradictions on the same entity
     don't multiply findings.

Relationship to the LLM judge's `coordination_contradiction`:
  This is the DETERMINISTIC, cheap version. Fires on structural
  contradictions (verdict field values that flip polarity). The judge
  catches subtler semantic contradictions the token matcher misses.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from drift.failures.base import FailureRecord
from drift.failures.library.base import (
    CoordinationDetectorContext,
    TraceStep,
    collect_string_values,
)

NAME = "contradictory_decisions"
SUMMARY = (
    "Same entity id received both a positive and a negative verdict in one trace."
)
MAST_MODES = ["3.2", "3.3"]  # specification ambiguity / step-out-of-order
SOURCE = (
    "MAST 3.2 (specification ambiguity); Anthropic engineering blog on "
    "agent reaching different conclusions when re-prompted; Cognition "
    "principle 2 (conflicting decisions carry bad results)"
)


# ---------------------------------------------------------------------------
# Vocabulary — smart defaults so the detector works without user config.
# Overridable via detect() kwargs.
# ---------------------------------------------------------------------------

DEFAULT_VERDICT_FIELDS: frozenset[str] = frozenset({
    "verdict", "decision", "outcome", "status", "approval", "resolution",
    "verdict_type", "review_result", "approval_status", "final_decision",
    "action",  # native-sim used "action" as a verdict-carrying enum
})

DEFAULT_ENTITY_ID_FIELDS: frozenset[str] = frozenset({
    "case_id", "ticket_id", "pr_id", "task_id", "order_id", "issue_id",
    "entity_id", "target_case_id", "target_id", "target", "context_id",
    "id",
})

# Strong polarity tokens. Kept short + tight to reduce false positives.
POSITIVE_TOKENS: frozenset[str] = frozenset({
    "approve", "approved", "accept", "accepted", "pass", "passed",
    "ok", "okay", "resolved", "closed", "success", "successful",
    "ship", "shipit", "shipped", "lgtm", "yes", "confirm", "confirmed",
    "granted", "authorized", "authorize", "green", "greenlight",
    "refund", "issue_refund",   # native-sim refund-topology positive kinds
    "close", "close_ticket",
})

NEGATIVE_TOKENS: frozenset[str] = frozenset({
    "reject", "rejected", "deny", "denied", "fail", "failed",
    "block", "blocked", "unresolved", "escalate", "escalated",
    "no", "decline", "declined", "refuse", "refused",
    "revoked", "revoke", "denied_access", "red", "hold", "on_hold",
    "deny_refund", "escalate_ticket",  # native-sim negative-kind analogues
})

# Ambiguous / neutral values that should NOT be treated as either polarity.
NEUTRAL_TOKENS: frozenset[str] = frozenset({
    "pending", "review", "under_review", "in_progress", "in progress",
    "n/a", "na", "unknown", "todo", "wip", "queued", "assigned",
})


def _canonicalize_id(id_text: str) -> str:
    return id_text.strip().lower().replace("_", "-").replace(" ", "-")


def _classify_token(tok: str) -> str:
    """Return 'positive', 'negative', 'neutral', or '' for an unknown token."""
    t = str(tok).strip().lower()
    if not t:
        return ""
    # Compare against neutral first — some neutrals (like 'no' would not appear
    # in neutral, but 'no update needed' should not fire) prevent false polarity.
    if t in NEUTRAL_TOKENS:
        return "neutral"
    if t in POSITIVE_TOKENS:
        return "positive"
    if t in NEGATIVE_TOKENS:
        return "negative"
    return ""


_POLARITY_TOKEN_RE = re.compile(r"\b([a-z][a-z_]{1,30})\b", re.I)


def _polarity_of_text(text: str) -> str:
    """Best-effort polarity from free text: first strong polarity token wins.

    We short-circuit on neutrals only if they appear alone; otherwise the
    presence of a strong positive/negative token elsewhere overrides.
    """
    if not text:
        return ""
    saw_neutral = False
    for m in _POLARITY_TOKEN_RE.finditer(text.lower()):
        cls = _classify_token(m.group(1))
        if cls == "positive" or cls == "negative":
            return cls
        if cls == "neutral":
            saw_neutral = True
    return "neutral" if saw_neutral else ""


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Vote:
    step: int
    agent: str
    entity_id: str
    polarity: str  # 'positive' | 'negative'
    evidence: str


def _extract_entity_id(
    step: TraceStep,
    entity_fields: frozenset[str],
) -> str:
    """Pull an entity id from a step's structured fields.

    Walks the update dict; the first matching id-shaped field wins.
    Returns '' if none found — the step's verdict then can't be attributed
    to an entity, so we skip it.
    """
    for k, v in step.update.items():
        if k.lower() in entity_fields and isinstance(v, str) and v:
            return _canonicalize_id(v)
    # Also try state_after (some graphs update entity in same super-step).
    for k, v in (step.state_after or {}).items():
        if k.lower() in entity_fields and isinstance(v, str) and v:
            return _canonicalize_id(v)
    return ""


def _extract_verdicts_from_update(
    update: dict[str, Any],
    verdict_fields: frozenset[str],
) -> list[tuple[str, str]]:
    """Return (polarity, evidence) pairs for verdict-shaped values in the update.

    Prefers structured verdict fields; falls back to polarity tokens in
    rationale-shaped free text. Neutral values are dropped (returned as no
    entry). Returns [] if no polarity signal found.
    """
    out: list[tuple[str, str]] = []
    # Structured verdict fields first.
    for k, v in update.items():
        if k.lower() in verdict_fields and isinstance(v, str):
            polarity = _classify_token(v)
            if polarity in ("positive", "negative"):
                out.append((polarity, f"{k}={v!r}"))
    if out:
        return out
    # Fallback: scan rationale-shaped free text.
    for k, v in update.items():
        if k.lower() in {"rationale", "reasoning", "explanation", "thought", "message"}:
            if isinstance(v, str) and v.strip():
                polarity = _polarity_of_text(v)
                if polarity in ("positive", "negative"):
                    snippet = v.strip()
                    if len(snippet) > 140:
                        snippet = snippet[:137] + "..."
                    out.append((polarity, f"{k}: {snippet!r}"))
    return out


# ---------------------------------------------------------------------------
# Structured detector
# ---------------------------------------------------------------------------


def detect(
    ctx: CoordinationDetectorContext,
    *,
    verdict_fields: Iterable[str] | None = None,
    entity_id_fields: Iterable[str] | None = None,
) -> list[FailureRecord]:
    """Fire once per entity that received both a positive AND a negative verdict."""
    vfields = frozenset(f.lower() for f in (verdict_fields or DEFAULT_VERDICT_FIELDS))
    efields = frozenset(f.lower() for f in (entity_id_fields or DEFAULT_ENTITY_ID_FIELDS))

    votes_by_entity: dict[str, list[_Vote]] = {}

    for step in ctx.steps:
        eid = _extract_entity_id(step, efields)
        if not eid:
            continue
        for polarity, evidence in _extract_verdicts_from_update(step.update, vfields):
            votes_by_entity.setdefault(eid, []).append(_Vote(
                step=step.step,
                agent=step.agent,
                entity_id=eid,
                polarity=polarity,
                evidence=evidence,
            ))

    findings: list[FailureRecord] = []
    for eid, votes in votes_by_entity.items():
        polarities = {v.polarity for v in votes}
        if "positive" in polarities and "negative" in polarities:
            # Build a compact evidence line: last positive + last negative vote.
            pos = next(v for v in reversed(votes) if v.polarity == "positive")
            neg = next(v for v in reversed(votes) if v.polarity == "negative")
            agents_involved = sorted({v.agent for v in votes if v.agent})
            last_step = max(v.step for v in votes)
            findings.append(FailureRecord(
                timestep=last_step,
                failure_type=NAME,
                agents_involved=agents_involved,
                evidence_action_ids=[],
                summary=(
                    f"entity {eid!r} received contradictory verdicts: "
                    f"POSITIVE at step {pos.step} by {pos.agent!r} ({pos.evidence}); "
                    f"NEGATIVE at step {neg.step} by {neg.agent!r} ({neg.evidence})"
                ),
                snapshot_timestep=last_step,
            ))
    # Deterministic ordering — sort by last-step so the earliest-clearing
    # contradiction is first in output.
    findings.sort(key=lambda f: f.timestep)
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

# Recognize "case 42", "TICKET-9", "#123" mentions in text. Duplicated from
# hallucinated_reference because we don't want a hard dependency; keep this
# module standalone.
_ID_PATTERNS = [
    re.compile(r"\b([A-Z]{2,10}-\d{1,10})\b"),
    re.compile(r"(?<![\w#])(#\d{2,10})\b"),
    re.compile(
        r"\b(case|ticket|pr|issue|order|task|entity|session|job)"
        r"[\s_\-]*([A-Za-z0-9]*\d[A-Za-z0-9]*)\b",
        re.I,
    ),
]


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
    """Text-only variant: pair id mentions with adjacent polarity signals.

    Recall-conservative: without structured data we can't be sure a polarity
    token refers to a given entity id. We fire only when the same id appears
    in two different utterances that carry OPPOSING polarity signals.
    """
    if not transcript:
        return []
    findings: list[FailureRecord] = []
    votes_by_entity: dict[str, list[tuple[int, str, str, str]]] = {}

    for line_index, m in enumerate(_SPEAKER_LINE.finditer(transcript)):
        speaker = m.group("name").strip()
        body = m.group("body") or ""
        polarity = _polarity_of_text(body)
        if polarity not in ("positive", "negative"):
            continue
        for ident in _ids_in_text(body):
            votes_by_entity.setdefault(ident, []).append(
                (line_index, speaker, polarity, body.strip()[:140])
            )

    for eid, votes in votes_by_entity.items():
        polarities = {v[2] for v in votes}
        if "positive" in polarities and "negative" in polarities:
            pos = next(v for v in reversed(votes) if v[2] == "positive")
            neg = next(v for v in reversed(votes) if v[2] == "negative")
            findings.append(FailureRecord(
                timestep=0,
                failure_type=NAME,
                agents_involved=sorted({pos[1], neg[1]}),
                evidence_action_ids=[],
                summary=(
                    f"text scan: entity {eid!r} received contradictory "
                    f"verdicts — POSITIVE by {pos[1]!r} and NEGATIVE by "
                    f"{neg[1]!r} in the transcript"
                ),
                snapshot_timestep=0,
            ))
    return findings


__all__ = [
    "DEFAULT_ENTITY_ID_FIELDS",
    "DEFAULT_VERDICT_FIELDS",
    "MAST_MODES",
    "NAME",
    "NEGATIVE_TOKENS",
    "NEUTRAL_TOKENS",
    "POSITIVE_TOKENS",
    "SOURCE",
    "SUMMARY",
    "detect",
    "detect_from_text",
]
