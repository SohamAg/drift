"""Detector: a handoff went to an agent that never said anything — and the
caller carried on as if it had.

Failure: the supervisor calls `transfer_to_research_expert`, the handoff
tool reports success, and the next AI turn is the supervisor again (or
another specialist) — the research expert produced no message at all. The
supervisor does not notice. It routes onward, and whatever the empty
expert was supposed to supply is now supplied by someone else's
imagination. Live (drift `skip_node[research_expert]` on the 2-agent
langgraph-supervisor system, gpt-4o-mini, 2/2 rollouts): math_expert
produced precise-looking populations for Tokyo and Paris from its own
knowledge and summed them; the final answer read as researched and was
not. `novel_tool_call` saw the math tools; nothing named the root cause.

Real-world causes: the specialist crashed and an error handler swallowed
it, a conditional edge sent control back early, a timeout upstream, a
subgraph whose output mode dropped its messages. The handoff contract —
"you asked X, X answered" — broke and the caller never checked.

Fires once per handoff when ALL hold:
  - an AI message calls a handoff tool whose name names a target agent
    (`transfer_to_<agent>`, `handoff_to_<agent>`, `delegate_to_<agent>`,
    `route_to_<agent>`) and the tool result is not an error,
  - a later AI message exists (the run went on),
  - the next AI message after the handoff result is NOT from the target:
    its `name`, or the root of the node that wrote it, differs from
    `<agent>`.
`transfer_back_to_*` (a specialist returning control) is not a handoff
to check. A handoff that is the last thing in the run is not flagged
(the run ended; premature_termination territory).
"""
from __future__ import annotations

import re

from drift.failures.base import FailureRecord
from drift.failures.library.base import CoordinationDetectorContext, Msg, conversation, pair_tool_results

NAME = "empty_handoff"
SUMMARY = (
    "A handoff to an agent succeeded but that agent never produced a message before control "
    "moved on — the caller continued as if it had answered."
)
MAST_MODES = ["2.4", "3.2"]  # information withholding; no or incomplete verification
SOURCE = (
    "MAST (arXiv:2503.13657) 2.4 Information withholding / 3.2 No or incomplete verification; "
    "LangGraph supervisor/swarm handoff tools (transfer_to_<agent>)"
)

_HANDOFF = re.compile(r"^(?:transfer_to_|handoff_to_|hand_off_to_|delegate_to_|route_to_)(?P<agent>.+)$", re.I)


def _target(call_name: str) -> str:
    m = _HANDOFF.match(call_name or "")
    return m.group("agent") if m else ""


def _speaker(m: Msg) -> set[str]:
    out = {m.name} if m.name else set()
    if m.agent:
        out.add(m.agent.split("/", 1)[0])
        out.add(m.agent.rsplit("/", 1)[-1])
    return {s.lower() for s in out if s}


def detect(ctx: CoordinationDetectorContext) -> list[FailureRecord]:
    msgs = conversation(ctx)
    if not msgs:
        return []
    pairs = pair_tool_results(msgs)
    by_idx = {m.index: m for m in msgs}
    findings: list[FailureRecord] = []
    for m in msgs:
        if m.role != "ai" or not m.tool_calls or m.is_drift_injected:
            continue
        for tc in m.tool_calls:
            target = _target(str(tc.get("name") or ""))
            if not target:
                continue
            result = next((x for x in msgs if x.index > m.index and x.role == "tool"
                           and x.tool_call_id is not None and x.tool_call_id == tc.get("id")), None)
            if result is None and len(m.tool_calls) == 1:
                res_idx = pairs.get(m.index)
                result = by_idx.get(res_idx) if res_idx is not None else None
            if result is None or result.is_error_result:
                continue
            nxt = next((x for x in msgs if x.index > result.index and x.role == "ai"), None)
            if nxt is None:
                continue
            if target.lower() in _speaker(nxt):
                continue
            who = nxt.name or nxt.agent or "another agent"
            what = "handed off again" if any(_target(str(c.get("name") or "")) for c in nxt.tool_calls) else (
                "answered" if nxt.text.strip() and not nxt.tool_calls else "carried on")
            caller = m.name or m.agent or "the caller"
            findings.append(FailureRecord(
                timestep=result.step or m.step,
                failure_type=NAME,
                agents_involved=sorted({a for a in (caller, target, who) if a}),
                evidence_action_ids=[],
                summary=(
                    f"{caller} handed off to {target!r}{f' at step {m.step}' if m.step > 0 else ''} and the handoff succeeded, "
                    f"but {target!r} produced no message: the next turn is {who!r}, which {what} — "
                    f"nothing checked that {target!r} answered"
                ),
                snapshot_timestep=result.step or m.step,
            ))
    return findings


def detect_from_text(transcript: str) -> list[FailureRecord]:
    """No text-only variant: needs structured handoff calls and speakers."""
    return []


__all__ = ["MAST_MODES", "NAME", "SOURCE", "SUMMARY", "detect", "detect_from_text"]
