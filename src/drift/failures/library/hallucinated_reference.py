"""Detector: agent references an entity_id that has never appeared in state.

Failure pattern: agent A's step output references entity id X (case-42,
TICKET-9, #123, task_id="foo-7") — but X has never appeared in any prior
state snapshot AND is not being defined by A's own update. The reference is
against thin air. Classic grounding failure — the agent's output is
plausible-sounding but is not tied to any real entity in the system's state.

Sources:
- MAST 2.x — grounding failures / disagreement with retrieved context
- Anthropic engineering blog on subagent hallucination (multi-agent research system)
- Cognition "Don't Build Multi-Agents" — subagents proceed on stale/absent context

Why it matters: in multi-agent systems, downstream agents inherit references
from upstream agents. If agent A hallucinates case-999, agent B may act on
"case-999" as if it's real and cascade damage. This is different from an LLM
hallucinating in a single-agent app: here the hallucination becomes shared
context and propagates.

Detection strategy (structured / adapter trace):

  1. Build a running `known_ids` set from `initial_state` and each prior
     `state_after`. IDs surface via (a) structured entity fields
     (`case_id`, `task_id`, `pr_id`, `order_id`, `issue_id`, `entity_id`,
     `id` inside list items) and (b) ID-pattern regexes in string values
     (rationales, tool outputs) — the two paths cover both machine and
     human-shaped state.

  2. For each step, scan `update` for ID-pattern mentions in string
     content. Any mention whose ID is NOT in `known_ids` AND is NOT being
     defined by this step's own `state_after` structured fields = a
     hallucinated reference.

  3. Fire once per (step, agent, id) tuple; downstream steps referencing
     the same hallucinated ID don't refire — they inherited the
     hallucination. We flag the ORIGIN of the hallucination, not its
     propagation.

Detection strategy (text-only variant):

  Precision-loose fallback: without state to anchor against, we cannot
  distinguish hallucinated ID from legitimate reference to a real one.
  So the text variant only fires on a much stricter signature: the same
  ID mentioned by one agent, then referenced by another, with no prior
  introduction pattern. Documented as recall-conservative in the MAST
  validation.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from drift.failures.base import FailureRecord
from drift.failures.library.base import (
    CoordinationDetectorContext,
    TraceStep,
    collect_string_values,
)

NAME = "hallucinated_reference"
SUMMARY = (
    "Agent referenced an entity id that has never appeared in state and "
    "isn't being defined by this step."
)
MAST_MODES = ["2.4", "2.6"]  # grounding failures / disagreement with retrieved context
SOURCE = (
    "MAST 2.x (grounding failures); Anthropic engineering blog on subagent "
    "hallucination in multi-agent research system"
)


# ---------------------------------------------------------------------------
# ID pattern surface — deliberately conservative to keep false positives low.
# ---------------------------------------------------------------------------

# Prefixed-numeric IDs: TICKET-42, PR-123, CASE-7, ISSUE-9. 2-10 uppercase.
_PREFIX_ID = re.compile(r"\b([A-Z]{2,10})-(\d{1,10})\b")

# Hash-prefixed numbers: #42, #100. Require at least 2 digits so we don't
# match "point #1" style bullets. Optional word-boundary preceding.
_HASH_ID = re.compile(r"(?<![\w#])#(\d{2,10})\b")

# Word-prefixed IDs: case-42, case_42, case 42, ticket-9, pr-3, issue-100,
# order-15, task-7, entity-A5. Lowercase-normalized on capture.
_ENTITY_WORDS = (
    "case", "ticket", "pr", "issue", "order", "task", "entity", "session",
    "job", "batch", "run", "req", "request",
)
_WORD_ID = re.compile(
    rf"\b(?P<word>{'|'.join(_ENTITY_WORDS)})[\s_\-]*"
    r"(?P<id>[A-Za-z0-9]{1,20})\b",
    re.IGNORECASE,
)

# Structured entity-id fields we recognize as canonical id-carrying keys.
# When a step's update sets one of these, it's DEFINING the id (or moving
# to a known one); not a hallucination candidate.
_ID_FIELD_NAMES = frozenset({
    "id", "case_id", "ticket_id", "pr_id", "task_id", "order_id",
    "issue_id", "entity_id", "session_id", "job_id", "run_id",
    "target_case_id", "target_id",
})


def _canonicalize(id_text: str) -> str:
    """Lowercase + strip so `Case-42`, `case-42`, `case_42` collapse together."""
    return id_text.strip().lower().replace("_", "-").replace(" ", "-")


def _extract_ids_from_text(text: str) -> list[str]:
    """Regex-scan a string for all ID-shaped tokens; return canonical forms.

    Order-preserving, deduped. We intentionally DO NOT match bare numbers —
    numbers alone are too ambiguous (counts, phone numbers, timestamps).
    """
    seen: dict[str, None] = {}
    if not text:
        return []
    for m in _PREFIX_ID.finditer(text):
        seen.setdefault(_canonicalize(f"{m.group(1)}-{m.group(2)}"), None)
    for m in _HASH_ID.finditer(text):
        seen.setdefault(_canonicalize(f"#{m.group(1)}"), None)
    for m in _WORD_ID.finditer(text):
        word = m.group("word").lower()
        ident = m.group("id")
        # Skip "case sensitive", "task list" type false positives — the id
        # part must contain at least one digit or be a short alnum token.
        if not any(ch.isdigit() for ch in ident):
            continue
        seen.setdefault(_canonicalize(f"{word}-{ident}"), None)
    return list(seen.keys())


def _extract_structured_ids(state: dict[str, Any]) -> set[str]:
    """Extract IDs that appear as VALUES of recognized ID fields.

    Walks nested dicts + lists. Values may be strings; we canonicalize.
    This is the "structured" surface — an id sitting in `case_id="c-42"`
    counts, but the same id mentioned inside a rationale string doesn't.
    """
    out: set[str] = set()

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
            if key_hint.lower() in _ID_FIELD_NAMES and v:
                out.add(_canonicalize(v))

    _walk(state)
    return out


def _extract_all_state_ids(state: dict[str, Any]) -> set[str]:
    """All IDs surfacing anywhere in state — structured OR text-embedded.

    This is the "known" set: an id counts as known if it's either a
    structured value OR mentioned in any string content (rationale,
    generated messages, etc). Broad on purpose so we don't over-flag.
    """
    out = _extract_structured_ids(state)
    strings: list[str] = []
    collect_string_values(state, strings)
    blob = " ".join(strings)
    out.update(_extract_ids_from_text(blob))
    return out


def _extract_referenced_ids_from_update(update: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract (id, evidence_text) pairs an agent's update REFERENCES.

    Scans string content of the update. Structured-id values are excluded
    from the reference list — they're either definitions (agent adds a
    new id) or moves to an existing id, both of which get validated
    against the known-set separately (through `state_after`).

    Returns list of (canonical_id, snippet_of_source_string) so callers
    can build informative evidence in the FailureRecord summary.
    """
    pairs: list[tuple[str, str]] = []
    seen_ids: set[str] = set()

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
            if key_hint.lower() in _ID_FIELD_NAMES:
                return  # structured id value, not a "reference in text"
            for ident in _extract_ids_from_text(v):
                if ident in seen_ids:
                    continue
                seen_ids.add(ident)
                snippet = v.strip()
                if len(snippet) > 140:
                    snippet = snippet[:137] + "..."
                pairs.append((ident, snippet))

    _walk(update)
    return pairs


# ---------------------------------------------------------------------------
# Structured detector
# ---------------------------------------------------------------------------


def detect(ctx: CoordinationDetectorContext) -> list[FailureRecord]:
    """Fire once at the origin of each hallucinated id reference.

    We flag the FIRST agent to mention an unknown id in a text context,
    not downstream agents that inherit the hallucination via propagated
    state or messages. That's why `_flagged_ids` accumulates across the
    whole trace and short-circuits repeat mentions.
    """
    findings: list[FailureRecord] = []
    known_ids: set[str] = set()
    if ctx.initial_state:
        known_ids |= _extract_all_state_ids(ctx.initial_state)

    flagged_ids: set[str] = set()

    for step in ctx.steps:
        # IDs this step's structured fields are DEFINING (or moving to) —
        # these are treated as legitimate even if previously unseen. If
        # they're new, they'll be added to known_ids at the end of the
        # step so subsequent references land as legitimate.
        defined_this_step = _extract_structured_ids(step.state_after) - known_ids

        for ident, snippet in _extract_referenced_ids_from_update(step.update):
            if ident in known_ids or ident in flagged_ids:
                continue
            if ident in defined_this_step:
                # Agent legitimately introduced the id this step; the text
                # mention is describing the entity it's just defined.
                continue
            findings.append(FailureRecord(
                timestep=step.step,
                failure_type=NAME,
                agents_involved=[step.agent] if step.agent else [],
                evidence_action_ids=[],
                summary=(
                    f"agent {step.agent!r} referenced id {ident!r} at step "
                    f"{step.step} but no prior state or definition mentions "
                    f"it (evidence: {snippet!r})"
                ),
                snapshot_timestep=step.step,
            ))
            flagged_ids.add(ident)

        # Update known-set from this step's full state_after (structured +
        # text-embedded), so subsequent steps benefit from anything the
        # step legitimately introduced.
        known_ids |= _extract_all_state_ids(step.state_after)

    return findings


# ---------------------------------------------------------------------------
# Raw-text variant (precision-loose)
# ---------------------------------------------------------------------------


# One-line utterance format: "Agent X: ...content..." — captures speaker + body.
_SPEAKER_LINE = re.compile(
    r"^\s*(?:\[|\<)?\s*"
    r"(?P<name>[A-Za-z][A-Za-z0-9_ ]{1,40}?)"
    r"\s*(?:\]|\>)?\s*:\s*"
    r"(?P<body>.*?)\s*$",
    re.M,
)


def detect_from_text(transcript: str) -> list[FailureRecord]:
    """Fire when a later-speaker references an id no earlier speaker introduced.

    Recall-conservative: without state, the only signature we can trust is
    "id appears fresh in speaker N's utterance and speaker N is not the
    first speaker". Fires at most once per novel id.

    Fires at most once per (speaker, id) — an agent can be flagged for
    multiple distinct hallucinated ids, but not repeatedly for the same one.
    """
    if not transcript:
        return []
    findings: list[FailureRecord] = []
    introduced: set[str] = set()  # ids named by any speaker so far
    flagged: set[tuple[str, str]] = set()

    lines = list(_SPEAKER_LINE.finditer(transcript))
    if len(lines) < 2:
        return []

    first_speaker_seen: set[str] = set()
    for m in lines:
        speaker_raw = m.group("name").strip()
        body = m.group("body") or ""
        speaker = speaker_raw.lower()

        ids = _extract_ids_from_text(body)
        # First mention by anyone -> introduced. We treat the first speaker
        # as authoritative (their ids are the initial context).
        is_first_speaker = speaker not in first_speaker_seen
        first_speaker_seen.add(speaker)

        for ident in ids:
            if ident in introduced:
                continue
            # If NOT the first speaker to appear and no one else introduced
            # this id, we call it a text-level hallucination.
            if len(first_speaker_seen) > 1 and (speaker, ident) not in flagged:
                findings.append(FailureRecord(
                    timestep=0,
                    failure_type=NAME,
                    agents_involved=[speaker_raw],
                    evidence_action_ids=[],
                    summary=(
                        f"text scan: speaker {speaker_raw!r} referenced id "
                        f"{ident!r} with no prior speaker introducing it"
                    ),
                    snapshot_timestep=0,
                ))
                flagged.add((speaker, ident))
            introduced.add(ident)
    return findings


__all__ = [
    "MAST_MODES",
    "NAME",
    "SOURCE",
    "SUMMARY",
    "detect",
    "detect_from_text",
]
