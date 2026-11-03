"""Detector: an agent issued the same tool call, with the same arguments,
over and over — and kept getting the same answer back.

Failure: an AI turn calls tool T with arguments A; the result comes back;
the next AI turn calls T(A) again; same result; again. Nothing in the
conversation changed between calls, so nothing in the agent's situation
changed either — the loop is burning tool budget and LLM turns while
looking, from the outside, exactly like work. MAST 1.3 (Step Repetition)
at the tool layer; the "tool repetition rate" metric of Think Locally,
Explain Globally (arXiv:2601.17915) is this count expressed as a ratio.

Why it exists alongside `repeated_identical_action`: that detector needs
the whole node `update` to be byte-identical across consecutive steps,
which an LLM-written message never is (the ids and wording wobble). This
one keys on the structured tool call — name plus canonical JSON of the
arguments — which the model emits exactly, and on the tool result, which
a deterministic tool returns exactly.

Fires when, inside one trace, the same (tool, args) pair occurs at least
`min_repeats` times (default 3) AND every paired result for it is
identical (or no result for it ever arrived). A repeated call whose
results differ is polling against something that is changing — status
checks, pagination cursors in the result — and is NOT flagged. Tool calls
drift itself injected as chaos (`drift_call_*`) are never counted.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from drift.failures.base import FailureRecord
from drift.failures.library.base import (
    CoordinationDetectorContext,
    Msg,
    conversation,
    pair_tool_results,
)

NAME = "tool_call_loop"
SUMMARY = "The same tool call with the same arguments was repeated and kept returning the same result."
MAST_MODES = ["1.3"]  # Step Repetition — tool-layer manifestation
SOURCE = (
    "MAST 1.3 (Step Repetition); Think Locally, Explain Globally (arXiv:2601.17915) "
    "tool repetition rate; AgentChaos (arXiv:2608.06790) retry amplification under faults"
)

DEFAULT_MIN_REPEATS = 3

# Handoff / routing tools (langgraph-supervisor's `transfer_to_<agent>` /
# `transfer_back_to_supervisor`, langgraph-swarm's handoffs) take no arguments
# and return a canned acknowledgement by design, so "same call, same result"
# is their normal operation, not a stuck loop. Bouncing between agents is a
# node-sequence question — `infinite_handoff` owns it.
ROUTING_TOOL_PATTERN = re.compile(r"^(?:transfer_(?:to|back)|handoff|hand_off|route_to|delegate_to)", re.I)


def _canonical_args(args: Any) -> str:
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        return repr(args)


def detect(
    ctx: CoordinationDetectorContext, *, min_repeats: int = DEFAULT_MIN_REPEATS
) -> list[FailureRecord]:
    if min_repeats < 2:
        raise ValueError("min_repeats must be >= 2")
    msgs = conversation(ctx)
    if not msgs:
        return []
    pairs = pair_tool_results(msgs)
    by_idx = {m.index: m for m in msgs}

    # (tool, canonical args) -> list of (ai Msg, result Msg | None)
    occurrences: dict[tuple[str, str], list[tuple[Msg, Msg | None]]] = defaultdict(list)
    for m in msgs:
        if m.role != "ai" or not m.tool_calls or m.step == 0 or m.is_drift_injected:
            continue
        result = by_idx.get(pairs.get(m.index, -1))
        for tc in m.tool_calls:
            name = tc.get("name")
            if not isinstance(name, str) or not name or ROUTING_TOOL_PATTERN.match(name):
                continue
            occurrences[(name, _canonical_args(tc.get("args")))].append((m, result))

    findings: list[FailureRecord] = []
    for (tool, args_key), occ in occurrences.items():
        if len(occ) < min_repeats:
            continue
        results = [r.text for _, r in occ if r is not None]
        if results and len(set(results)) > 1:
            continue  # results changed between calls — polling, not a loop
        steps = [m.step for m, _ in occ]
        agents = sorted({m.agent for m, _ in occ if m.agent})
        sample = results[0] if results else ""
        sample = " ".join(sample.split())
        sample = (sample[:60] + "…") if len(sample) > 60 else sample
        findings.append(FailureRecord(
            timestep=steps[-1],
            failure_type=NAME,
            agents_involved=agents,
            evidence_action_ids=[],
            summary=(
                f"tool {tool!r} was called {len(occ)} times with identical arguments "
                f"{args_key} (steps {steps[0]}–{steps[-1]})"
                + (f"; every result was the same ({sample!r})" if results else "; no result ever came back")
            ),
            snapshot_timestep=steps[-1],
        ))
    return findings


# ---------------------------------------------------------------------------
# Raw-text variant — for MAST-style transcripts.
# ---------------------------------------------------------------------------

# "Calling tool search with {...}", "tool_call: search(...)", "Action: search", "search({"q": "x"})"
_TOOL_CALL_LINE = re.compile(
    r"(?:calling|call(?:ed|s)?|invok(?:e|ing|ed)|action|tool_call|tool)\s*[:=]?\s*"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_\.]{1,60})\s*(?P<args>\(.*?\)|\{.*?\})?",
    re.I,
)


def detect_from_text(transcript: str, *, min_repeats: int = DEFAULT_MIN_REPEATS) -> list[FailureRecord]:
    """Text-only variant: count identical `tool(args)` mentions. No results to
    compare against, so this fires on bare repetition — precision-loose."""
    if not transcript:
        return []
    counts: dict[str, int] = defaultdict(int)
    for line in transcript.splitlines():
        m = _TOOL_CALL_LINE.search(line)
        if not m:
            continue
        key = f"{m.group('name').lower()}{(m.group('args') or '').strip()}"
        counts[key] += 1
    out: list[FailureRecord] = []
    for key, n in counts.items():
        if n >= min_repeats:
            out.append(FailureRecord(
                timestep=0,
                failure_type=NAME,
                agents_involved=[],
                evidence_action_ids=[],
                summary=f"text scan: tool call {key!r} appears {n} times verbatim",
                snapshot_timestep=0,
            ))
    return out


__all__ = [
    "DEFAULT_MIN_REPEATS",
    "MAST_MODES",
    "ROUTING_TOOL_PATTERN",
    "NAME",
    "SOURCE",
    "SUMMARY",
    "detect",
    "detect_from_text",
]
