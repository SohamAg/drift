"""Detector: the receiving agent's output references none of the specific
content the previous, different agent just wrote.

Failure: agent N introduces specific, identifiable content — an id-shaped
token (TICKET-42, #101, case-7) or a short new discrete state value — and
agent N+1 (a DIFFERENT agent, the very next step) produces output that
mentions none of it. Repeated across a trace, this is the classic "lost in
the handoff" failure: downstream agents acting as if the upstream agent's
specific findings never happened. MAST 1.4 (Loss of Conversation History)
+ 2.5 (Ignoring Other Agents' Input).

Distinct from `hallucinated_reference` (agent invents a NEW id that was
never real) and `stale_state_reference` (agent cites a REMOVED id) — this
one is about an agent dropping something that's still live and correct,
simply not carrying it forward.

Deliberately requires >= 2 missed handoffs (not 1) before firing, and only
counts a handoff as a candidate when the sender actually introduced
something concrete enough to track (an id-shaped token, or a new short
string value on a new key) — most steps write nothing distinctive enough
to expect a citation, and flagging those would make the detector noise,
not signal.
"""
from __future__ import annotations

import re
from typing import Any

from drift.failures.base import FailureRecord
from drift.failures.library.base import (
    CoordinationDetectorContext,
    TraceStep,
    collect_string_values,
)

NAME = "context_loss_on_handoff"
SUMMARY = "Receiving agent referenced none of the sending agent's specific content."
MAST_MODES = ["1.4", "2.5"]  # Loss of Conversation History; Ignoring Other Agents' Input
SOURCE = "MAST 1.4 (Loss of Conversation History) + 2.5 (Ignoring Other Agents' Input)"

DEFAULT_MIN_MISSES = 2

# Same id-shape family as hallucinated_reference, kept intentionally
# narrow/conservative — see that detector's docstring for the rationale.
_ID_SHAPE = re.compile(r"\b([A-Za-z]{2,10}-\d+|#\d{2,}|[a-z]+[-_ ]\d+)\b", re.I)


def _text_blob(update: dict[str, Any], rationale: str) -> str:
    parts: list[str] = []
    collect_string_values(update, parts)
    if rationale:
        parts.append(rationale)
    return " ".join(parts)


def _extract_anchors(update: dict[str, Any], rationale: str, prior_keys: set[str]) -> set[str]:
    """Concrete, trackable content this step introduced: id-shaped tokens
    anywhere in its text, plus short new string values on keys that didn't
    exist before this step (a decision/classification/verdict-shaped
    write, not prose)."""
    text = _text_blob(update, rationale)
    anchors = {m.group(1).lower() for m in _ID_SHAPE.finditer(text)}
    for key, value in update.items():
        if key in prior_keys:
            continue
        if isinstance(value, str) and 2 <= len(value.strip()) <= 40:
            anchors.add(value.strip().lower())
    return anchors


def detect(ctx: CoordinationDetectorContext, *, min_misses: int = DEFAULT_MIN_MISSES) -> list[FailureRecord]:
    steps = ctx.steps
    if len(steps) < 2:
        return []

    seen_keys: set[str] = set((ctx.initial_state or {}).keys())
    misses: list[tuple[TraceStep, TraceStep, set[str]]] = []
    for i in range(len(steps) - 1):
        cur, nxt = steps[i], steps[i + 1]
        anchors = _extract_anchors(cur.update, cur.rationale, seen_keys)
        seen_keys |= set(cur.update.keys())
        if not anchors or nxt.agent == cur.agent:
            continue
        receiver_blob = _text_blob(nxt.update, nxt.rationale).lower()
        if not any(a in receiver_blob for a in anchors):
            misses.append((cur, nxt, anchors))

    if len(misses) < min_misses:
        return []

    cur, nxt, anchors = misses[0]
    example = ", ".join(sorted(anchors)[:3])
    return [FailureRecord(
        timestep=nxt.step,
        failure_type=NAME,
        agents_involved=sorted({cur.agent, nxt.agent}),
        evidence_action_ids=[],
        summary=(
            f"{len(misses)} handoff(s) where the receiving agent referenced none of "
            f"the sender's content; first at step {nxt.step}: {cur.agent!r} wrote "
            f"[{example}] but {nxt.agent!r}'s output mentions none of it"
        ),
        snapshot_timestep=nxt.step,
    )]


# ---------------------------------------------------------------------------
# Raw-text variant — for MAST validation.
# ---------------------------------------------------------------------------

_SPEAKER_LINE = re.compile(
    r"^[\[\<]?\s*(?P<name>[A-Za-z][A-Za-z0-9_ ]{1,40}?)\s*[:\]>]\s*(?P<body>.*)$",
    re.M,
)


def detect_from_text(transcript: str, *, min_misses: int = DEFAULT_MIN_MISSES) -> list[FailureRecord]:
    """Text-only variant: treat consecutive different-speaker lines as
    handoffs, extract id-shaped anchors from the sender's line, check for
    them in the receiver's line. No state, so only the id-shape anchor
    source applies (no "new key" signal available in raw text)."""
    if not transcript:
        return []
    lines = [
        (m.group("name").strip(), m.group("body"))
        for m in _SPEAKER_LINE.finditer(transcript)
        if m.group("body").strip()
    ]
    if len(lines) < 2:
        return []

    misses: list[tuple[str, str, set[str]]] = []
    for i in range(len(lines) - 1):
        sender, sender_body = lines[i]
        receiver, receiver_body = lines[i + 1]
        if receiver == sender:
            continue
        anchors = {m.group(1).lower() for m in _ID_SHAPE.finditer(sender_body)}
        if not anchors:
            continue
        if not any(a in receiver_body.lower() for a in anchors):
            misses.append((sender, receiver, anchors))

    if len(misses) < min_misses:
        return []

    sender, receiver, anchors = misses[0]
    example = ", ".join(sorted(anchors)[:3])
    return [FailureRecord(
        timestep=0,
        failure_type=NAME,
        agents_involved=sorted({sender, receiver}),
        evidence_action_ids=[],
        summary=(
            f"text scan: {len(misses)} handoff(s) with no cross-reference; "
            f"first: {sender!r} mentioned [{example}] but {receiver!r} mentions none of it"
        ),
        snapshot_timestep=0,
    )]


__all__ = [
    "DEFAULT_MIN_MISSES",
    "MAST_MODES",
    "NAME",
    "SOURCE",
    "SUMMARY",
    "detect",
    "detect_from_text",
]
