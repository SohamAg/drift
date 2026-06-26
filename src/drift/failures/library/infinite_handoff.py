"""Detector: two agents handing off back and forth without progress.

Failure: agents A and B alternate `A -> B -> A -> B -> ...` past a threshold,
and the state isn't materially advancing — no new top-level fields, no new
entity references introduced. The classic "you handle it" / "no you handle
it" loop. MAST 1.3 (Step Repetition); Cognition's named open problem #2
(cross-sibling discovery sharing — agents punting without context transfer).

Why it matters: looks like activity (lots of super-steps) but is wasted work.
Particularly toxic when each handoff burns LLM cost. Anthropic's "minor
failures compound catastrophically in stateful tool calls" lives here.

The detector fires when ALL hold inside a sliding window:
  - Same unordered pair {A, B} owns at least `min_alternations` adjacent
    super-steps (default 4, per Cognition's "more than a couple of bounces")
  - No new keys appeared in the state across those steps
  - No previously-empty string field became non-empty
  - No list/dict field grew (length stable)

We require ALL three "no progress" conditions because any one alone has
false positives — chitchat between agents is normal; growth without
correlated value-progression is normal. The conjunction is the signature.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from drift.failures.base import FailureRecord
from drift.failures.library.base import (
    CoordinationDetectorContext,
    TraceStep,
)

NAME = "infinite_handoff"
SUMMARY = (
    "Two agents alternated past threshold with no measurable state advancement."
)
MAST_MODES = ["1.3", "2.3"]  # Step Repetition; Task derailment (handoffs ≠ progress)
SOURCE = "MAST 1.3 (Step Repetition); Cognition open problem #2 (cross-sibling sharing)"

DEFAULT_MIN_ALTERNATIONS = 4  # A→B→A→B is 4 steps
DEFAULT_MAX_LOOK_BACK = 12     # bound the window; longer = more false positives


def _alternation_run(agents_seq: list[str]) -> tuple[str, str, int] | None:
    """Find the longest trailing run where exactly two distinct agents alternate.

    Returns (a, b, length) where the run is the LAST `length` steps and
    they bounce between agents `a` and `b`. length is the count of steps in
    the alternation, not the count of A->B transitions.

    Returns None if the trailing pattern doesn't involve exactly two agents
    or doesn't strictly alternate. We anchor at the tail because that's
    where the most recent (most diagnostic) handoff loop lives.
    """
    if len(agents_seq) < 2:
        return None
    # Walk backwards from the end, collecting the longest strict alternation.
    last = agents_seq[-1]
    prev = agents_seq[-2]
    if last == prev:
        return None
    run = 2
    a, b = last, prev
    for i in range(len(agents_seq) - 3, -1, -1):
        expected = a if (run % 2 == 0) else b
        if agents_seq[i] == expected:
            run += 1
        else:
            break
    if run < 2:
        return None
    return (a, b, run)


@dataclass
class _ProgressFingerprint:
    """Snapshot of what counts as 'state advancement' between two steps."""
    keys: frozenset[str]
    nonempty_string_keys: frozenset[str]
    container_sizes: dict[str, int]


def _fingerprint(state: dict[str, Any]) -> _ProgressFingerprint:
    keys = frozenset(state.keys())
    nonempty_str = frozenset(
        k for k, v in state.items() if isinstance(v, str) and v.strip()
    )
    sizes = {
        k: len(v)
        for k, v in state.items()
        if isinstance(v, (list, dict, set, tuple))
    }
    return _ProgressFingerprint(keys=keys, nonempty_string_keys=nonempty_str, container_sizes=sizes)


def _has_progress(fp_before: _ProgressFingerprint, fp_after: _ProgressFingerprint) -> bool:
    """True if `after` shows measurable advancement vs `before`.

    Specifically: new key(s), new non-empty-string fields, or any container
    that grew. We deliberately don't count value changes alone — that's
    natural-language wobble territory and would let the loop hide.
    """
    if fp_after.keys - fp_before.keys:
        return True
    if fp_after.nonempty_string_keys - fp_before.nonempty_string_keys:
        return True
    for k, sz_after in fp_after.container_sizes.items():
        if sz_after > fp_before.container_sizes.get(k, 0):
            return True
    return False


def detect(
    ctx: CoordinationDetectorContext,
    *,
    min_alternations: int = DEFAULT_MIN_ALTERNATIONS,
    max_look_back: int = DEFAULT_MAX_LOOK_BACK,
) -> list[FailureRecord]:
    """Detect a no-progress handoff loop. Fires at most once per pair."""
    if len(ctx.steps) < min_alternations:
        return []
    # Use only the trailing window — older parts of the trace may have
    # been progress-making and we don't want to penalize a graph that
    # was working before the loop started.
    window = ctx.steps[-max_look_back:]
    agents_seq = [s.agent for s in window]
    run = _alternation_run(agents_seq)
    if run is None:
        return []
    a, b, run_len = run
    if run_len < min_alternations:
        return []
    # Confirm no progress across the run.
    run_steps = window[-run_len:]
    fp_first = _fingerprint(run_steps[0].state_after)
    fp_last = _fingerprint(run_steps[-1].state_after)
    if _has_progress(fp_first, fp_last):
        return []
    last_step = run_steps[-1].step
    return [FailureRecord(
        timestep=last_step,
        failure_type=NAME,
        agents_involved=sorted({a, b}),
        evidence_action_ids=[],
        summary=(
            f"agents {a!r} and {b!r} alternated for {run_len} consecutive "
            f"steps (steps {run_steps[0].step}–{run_steps[-1].step}) "
            f"with no new keys, no new non-empty fields, and no container growth"
        ),
        snapshot_timestep=last_step,
    )]


# ---------------------------------------------------------------------------
# Raw-text variant — for MAST validation.
# ---------------------------------------------------------------------------


# Matches lines naming the agent at the start of an utterance:
# "Agent X:", "[Agent X]", "Response from Agent X", "<Agent X>:"
_AGENT_PREFIX = re.compile(
    r"^[\[\<]?\s*(?:response from|message to|to|from)?\s*"
    r"(?P<name>[A-Z][A-Za-z0-9_ ]{2,40}?\b(?:agent|bot|reviewer|verifier|planner|executor|critic|developer|tester|coder)?)\s*"
    r"[:\]>]",
    re.I | re.M,
)


def _extract_speaker_sequence(transcript: str) -> list[str]:
    """Pull an ordered list of speakers from a transcript using cheap heuristics.

    Designed for MAST traces which mix several frameworks' formats. Misses
    are OK — false negatives on this path are honest because raw text is
    weaker than structured trace.
    """
    seq: list[str] = []
    for m in _AGENT_PREFIX.finditer(transcript):
        name = m.group("name").strip()
        # Normalize whitespace; collapse "Agent  X" -> "agent x".
        name = re.sub(r"\s+", " ", name).lower()
        if seq and seq[-1] == name:
            continue  # collapse consecutive same-speaker lines
        seq.append(name)
    return seq


def detect_from_text(
    transcript: str,
    *,
    min_alternations: int = DEFAULT_MIN_ALTERNATIONS,
) -> list[FailureRecord]:
    """Text-only variant: find a long alternating-pair run in the speaker sequence.

    Cannot check state-progress (no state in raw text), so this variant
    fires more aggressively than the structured one. Documented in the
    MAST validation as a precision-loose, recall-conservative anchor.
    """
    if not transcript:
        return []
    seq = _extract_speaker_sequence(transcript)
    if len(seq) < min_alternations:
        return []
    run = _alternation_run(seq)
    if run is None:
        return []
    a, b, run_len = run
    if run_len < min_alternations:
        return []
    return [FailureRecord(
        timestep=0,
        failure_type=NAME,
        agents_involved=sorted({a, b}),
        evidence_action_ids=[],
        summary=(
            f"text scan: speakers {a!r} and {b!r} alternated for {run_len} "
            f"consecutive utterances at end of transcript"
        ),
        snapshot_timestep=0,
    )]


__all__ = [
    "DEFAULT_MAX_LOOK_BACK",
    "DEFAULT_MIN_ALTERNATIONS",
    "MAST_MODES",
    "NAME",
    "SOURCE",
    "SUMMARY",
    "detect",
    "detect_from_text",
]
