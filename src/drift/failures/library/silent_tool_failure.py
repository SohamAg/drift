"""Detector: a tool returned an error and the system's final answer never
let on — no retry, no fallback, no mention of the failure.

Failure: a ToolMessage carries `status="error"` (or an error-shaped body —
"Error: ...", a traceback) and then the conversation simply continues to a
confident final answer. The agent neither re-issued the call, nor got a
later successful result from that tool, nor said anything a user could
read as "something went wrong". AgentChaos (arXiv:2608.06790) calls this
outcome class *Silent* — plausible output with no warning — and measured
it at 22–66% of faulted runs across six agent frameworks, precisely
because an error inside a tool result is an HTTP 200 to the agent loop and
bypasses every exception handler.

Relationship to `unacknowledged_error`: that detector reads structured
error FIELDS a node writes to state (`error`, `failed=True`). This one
reads the chat messages — the tool layer — where most real failures in a
LangGraph agent actually land. Together they cover both channels.

Fires once per failed tool (first unacknowledged occurrence) when ALL hold:
  - a tool result is an error (status, or error-shaped text),
  - a later AI message with text content and no tool calls exists (the
    system DID produce a final answer — a trace that stops right after the
    error isn't silent, it's truncated),
  - between the error and the end of the trace there is no acknowledgment:
    no further call to that tool, no later non-error result from it, and no
    AI text containing a failure/retry/fallback token.
The failed tool result drift itself injected via `inject_tool_error_result`
counts — that pattern exists exactly so this detector can answer "does your
graph surface a failed dependency, or paper over it?"
"""
from __future__ import annotations

import re

from drift.failures.base import FailureRecord
from drift.failures.library.base import (
    CoordinationDetectorContext,
    Msg,
    any_token,
    conversation,
    pair_tool_results,
)

NAME = "silent_tool_failure"
SUMMARY = "A tool call failed and the final answer never surfaced it — no retry, fallback, or mention."
MAST_MODES = ["3.3", "2.3"]  # Incorrect verification; task derailment — output not grounded in a succeeded step
SOURCE = (
    "AgentChaos (arXiv:2608.06790) 'Silent' outcome class; Anthropic multi-agent "
    "research postmortem (errors treated as success); MAST 3.3"
)

# AI text that counts as surfacing the failure to the user or to itself.
ACK_TOKENS = (
    "error", "errors", "failed", "failure", "fail", "unable", "couldn't", "could not",
    "can't", "cannot", "unavailable", "timed out", "timeout", "time out", "retry",
    "retrying", "try again", "fallback", "fell back", "falling back", "not able",
    "unsuccessful", "problem", "issue", "went wrong", "sorry", "apologize", "apologise",
    "rate limit", "rate-limit", "unreachable", "did not return", "didn't return",
    "no response", "not available", "exception",
)


def _ack_in_text(text: str) -> str | None:
    return any_token(text.lower(), ACK_TOKENS)


def detect(ctx: CoordinationDetectorContext) -> list[FailureRecord]:
    msgs = conversation(ctx)
    if not msgs:
        return []
    pairs = pair_tool_results(msgs)
    result_to_call: dict[int, Msg] = {}
    by_idx = {m.index: m for m in msgs}
    for ai_idx, res_idx in pairs.items():
        result_to_call[res_idx] = by_idx[ai_idx]

    findings: list[FailureRecord] = []
    reported_tools: set[str] = set()
    for m in msgs:
        if not m.is_error_result:
            continue
        caller = result_to_call.get(m.index)
        tool = ""
        if caller is not None:
            for tc in caller.tool_calls:
                if tc.get("id") == m.tool_call_id or len(caller.tool_calls) == 1:
                    tool = str(tc.get("name") or "")
                    break
        tool = tool or m.name or "<unknown tool>"
        if tool in reported_tools:
            continue

        later = [x for x in msgs if x.index > m.index]
        final_answer: Msg | None = None
        acknowledged = False
        for x in later:
            if x.role == "ai" and any(str(tc.get("name") or "") == tool for tc in x.tool_calls):
                acknowledged = True  # retried
                break
            if x.role == "tool" and not x.is_error_result:
                c = result_to_call.get(x.index)
                if c is not None and any(str(tc.get("name") or "") == tool for tc in c.tool_calls):
                    acknowledged = True  # a later success for the same tool
                    break
            if x.role == "ai" and x.text.strip():
                if _ack_in_text(x.text):
                    acknowledged = True
                    break
                if not x.tool_calls:
                    final_answer = x  # keep the LAST final-shaped answer
        if acknowledged or final_answer is None:
            continue

        reported_tools.add(tool)
        err_preview = " ".join(m.text.split())
        err_preview = (err_preview[:70] + "…") if len(err_preview) > 70 else err_preview
        ans_preview = " ".join(final_answer.text.split())
        ans_preview = (ans_preview[:70] + "…") if len(ans_preview) > 70 else ans_preview
        where = f"step {m.step}" if m.step > 0 else "the initial state (drift-injected)"
        agents = sorted({a for a in (m.agent, final_answer.agent) if a})
        findings.append(FailureRecord(
            timestep=final_answer.step,
            failure_type=NAME,
            agents_involved=agents,
            evidence_action_ids=[],
            summary=(
                f"tool {tool!r} returned an error at {where} ({err_preview!r}); the final "
                f"answer at step {final_answer.step} ({ans_preview!r}) mentions no failure, "
                f"and the tool was never retried"
            ),
            snapshot_timestep=final_answer.step,
        ))
    return findings


# ---------------------------------------------------------------------------
# Raw-text variant — for MAST-style transcripts.
# ---------------------------------------------------------------------------

_ERROR_LINE = re.compile(r"^\s*(?:\[?tool[^\]]*\]?\s*[:>]?\s*)?(?:error\b|traceback \(most recent call last\))", re.I | re.M)


def detect_from_text(transcript: str) -> list[FailureRecord]:
    """Text-only variant: an error-shaped line exists and no later line
    carries an acknowledgment token. Cannot tell tool results from agent
    prose, so this is a loose anchor, not a verdict."""
    if not transcript:
        return []
    m = _ERROR_LINE.search(transcript)
    if not m:
        return []
    tail = transcript[m.end():]
    if not tail.strip():
        return []
    if _ack_in_text(tail):
        return []
    return [FailureRecord(
        timestep=0,
        failure_type=NAME,
        agents_involved=[],
        evidence_action_ids=[],
        summary="text scan: an error-shaped line is followed by output that never mentions a failure, retry, or fallback",
        snapshot_timestep=0,
    )]


__all__ = [
    "ACK_TOKENS",
    "MAST_MODES",
    "NAME",
    "SOURCE",
    "SUMMARY",
    "detect",
    "detect_from_text",
]
