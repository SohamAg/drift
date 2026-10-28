"""Detector: a single agent repeated the exact same action several times
in a row with no change in output.

Failure: unlike `infinite_handoff` (two agents strictly alternating with
no progress), this catches the single-agent variant — one node firing
repeatedly, back to back, producing a byte-identical update each time.
The classic "stuck retry loop": an agent calls the same tool with the
same arguments over and over, or a ReAct-style self-loop re-emits the
same action because whatever triggered the retry never actually
resolved. MAST 1.3 (Step Repetition) — the same mode `infinite_handoff`
cites — covers this manifestation too; Anthropic's "minor failures
compound catastrophically in stateful tool calls" postmortem note is
exactly this pattern at the tool-call level.

Why it matters: each repeat is often a real (billed) tool call or LLM
turn. A stuck retry silently burns cost and time without the system
ever surfacing that it's stuck — from the outside, "the agent is doing
something" looks identical to "the agent is doing nothing useful."

The detector fires when the SAME agent appears in `min_repeats`
(default 3) CONSECUTIVE trace steps — no other agent interleaves — and
produces the exact same `update` dict every time. Deliberately strict
equality, not fuzzy similarity: a retry that changes even one field
(an incrementing attempt counter, a slightly reworded message) is
evidence the agent noticed something changed, which is a different
(and less clearly broken) situation than a byte-identical repeat. An
empty update (`{}`) never counts — a node that writes nothing isn't
repeating an "action," just a structural pass-through.
"""
from __future__ import annotations

import re

from drift.failures.base import FailureRecord
from drift.failures.library.base import CoordinationDetectorContext

NAME = "repeated_identical_action"
SUMMARY = "A single agent repeated the exact same action several times in a row."
MAST_MODES = ["1.3"]  # Step Repetition — single-agent manifestation
SOURCE = "MAST 1.3 (Step Repetition); Anthropic multi-agent research postmortem (stateful tool call failures compound)"

DEFAULT_MIN_REPEATS = 3


def detect(
    ctx: CoordinationDetectorContext, *, min_repeats: int = DEFAULT_MIN_REPEATS
) -> list[FailureRecord]:
    """Fire once per consecutive identical-action run of length >= min_repeats."""
    if min_repeats < 2:
        raise ValueError("min_repeats must be >= 2")
    steps = ctx.steps
    out: list[FailureRecord] = []
    i = 0
    n = len(steps)
    while i < n:
        if not steps[i].update:
            i += 1
            continue
        j = i
        while (
            j + 1 < n
            and steps[j + 1].agent == steps[i].agent
            and steps[j + 1].update == steps[i].update
        ):
            j += 1
        run_len = j - i + 1
        if run_len >= min_repeats:
            last = steps[j]
            out.append(FailureRecord(
                timestep=last.step,
                failure_type=NAME,
                agents_involved=[steps[i].agent],
                evidence_action_ids=[],
                summary=(
                    f"agent {steps[i].agent!r} repeated the exact same action "
                    f"{run_len} times in a row (steps {steps[i].step}-{last.step}) "
                    f"with no change to its output"
                ),
                snapshot_timestep=last.step,
            ))
        i = j + 1
    return out


# ---------------------------------------------------------------------------
# Raw-text variant — for MAST validation.
# ---------------------------------------------------------------------------


# Same speaker-prefix shape infinite_handoff.py matches, kept local rather
# than shared so each detector module stays independently readable — see
# that module's docstring for why the pattern looks like this.
_AGENT_PREFIX = re.compile(
    r"^[\[\<]?\s*(?:response from|message to|to|from)?\s*"
    r"(?P<name>[A-Z][A-Za-z0-9_ ]{2,40}?\b(?:agent|bot|reviewer|verifier|planner|executor|critic|developer|tester|coder)?)\s*"
    r"[:\]>]\s*(?P<rest>.*)$",
    re.I,
)


def _extract_speaker_lines(transcript: str) -> list[tuple[str, str]]:
    """Pull (speaker, normalized_line) pairs from a transcript. Misses are
    OK — honest false negatives on raw text are weaker than the structured
    path by design."""
    out: list[tuple[str, str]] = []
    for line in transcript.splitlines():
        m = _AGENT_PREFIX.match(line.strip())
        if not m:
            continue
        name = re.sub(r"\s+", " ", m.group("name").strip()).lower()
        rest = re.sub(r"\s+", " ", m.group("rest").strip()).lower()
        if name and rest:
            out.append((name, rest))
    return out


def detect_from_text(
    transcript: str, *, min_repeats: int = DEFAULT_MIN_REPEATS
) -> list[FailureRecord]:
    """Text-only variant: a long run of consecutive identical (speaker, line)
    pairs. Looser than the structured path — natural-language rewording of
    the same underlying action won't match, so this under-fires relative to
    `detect` on paraphrased retries, by design (recall-conservative)."""
    if not transcript:
        return []
    pairs = _extract_speaker_lines(transcript)
    i = 0
    n = len(pairs)
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1] == pairs[i]:
            j += 1
        run_len = j - i + 1
        if run_len >= min_repeats:
            speaker, line = pairs[i]
            return [FailureRecord(
                timestep=0,
                failure_type=NAME,
                agents_involved=[speaker],
                evidence_action_ids=[],
                summary=(
                    f"text scan: speaker {speaker!r} repeated the same line "
                    f"{run_len} times in a row: {line[:140]!r}"
                ),
                snapshot_timestep=0,
            )]
        i = j + 1
    return []


__all__ = [
    "DEFAULT_MIN_REPEATS",
    "MAST_MODES",
    "NAME",
    "SOURCE",
    "SUMMARY",
    "detect",
    "detect_from_text",
]
