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
    AI text that surfaces a failure. "Surfaces" is deliberately two-tier:
      * STRONG tokens ("error", "failed", "timed out", "retry", …) count
        wherever they appear — they are failure vocabulary.
      * WEAK tokens ("can't", "cannot", "unable", "sorry", "couldn't", …) are
        ordinary prose; they count only when spoken by the agent that made
        the failed call, or in the final answer itself. On a live supervisor
        run the planted `lookup` failure went unmentioned while a DIFFERENT
        specialist said "I can't search the web" about its own tools — a
        capability statement, not an acknowledgment — and the supervisor's
        final answer surfaced nothing. That is a silent failure.
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

# Failure vocabulary — counts as surfacing the failure wherever it appears.
STRONG_ACK_TOKENS = (
    "error", "errors", "failed", "failure", "unavailable", "timed out", "timeout",
    "time out", "retry", "retrying", "try again", "fallback", "fell back",
    "falling back", "unsuccessful", "went wrong", "rate limit", "rate-limit",
    "unreachable", "did not return", "didn't return", "no response",
    "not available", "exception", "could not reach", "couldn't reach",
    "could not retrieve", "couldn't retrieve", "could not fetch", "couldn't fetch",
)
# Ordinary hedging prose. Counts only from the agent that made the failed call
# or in the final answer — anywhere else it is as likely to be a specialist
# describing its own limits ("I can't search the web") as an acknowledgment.
WEAK_ACK_TOKENS = (
    "fail", "unable", "couldn't", "could not", "can't", "cannot", "not able",
    "problem", "issue", "sorry", "apologize", "apologise",
)
# Union, for the text-only variant and for callers that just want "any".
ACK_TOKENS = STRONG_ACK_TOKENS + WEAK_ACK_TOKENS


def _ack_in_text(text: str) -> str | None:
    return any_token(text.lower(), ACK_TOKENS)


def _strong_ack(text: str) -> str | None:
    return any_token(text.lower(), STRONG_ACK_TOKENS)


def _weak_ack(text: str) -> str | None:
    return any_token(text.lower(), WEAK_ACK_TOKENS)


def _agent_root(label: str) -> str:
    """`math_expert/agent` and `math_expert` are the same agent (inner step
    vs the parent's commit); compare on the first path segment."""
    return (label or "").split("/", 1)[0]


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
        caller_agent = _agent_root(caller.agent) if caller is not None else ""

        later = [x for x in msgs if x.index > m.index]
        # The final answer is the LAST final-shaped AI message (text, no
        # tool calls) after the error — what the user actually receives.
        final_answer: Msg | None = None
        for x in later:
            if x.role == "ai" and x.text.strip() and not x.tool_calls:
                final_answer = x
        if final_answer is None:
            continue

        acknowledged = False
        bystander_hedge: Msg | None = None  # weak ack from someone else
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
                if _strong_ack(x.text):
                    acknowledged = True
                    break
                if _weak_ack(x.text):
                    same_agent = bool(caller_agent) and _agent_root(x.agent) == caller_agent
                    if same_agent or x is final_answer:
                        acknowledged = True
                        break
                    bystander_hedge = bystander_hedge or x
        if acknowledged:
            continue

        reported_tools.add(tool)
        err_preview = " ".join(m.text.split())
        err_preview = (err_preview[:70] + "…") if len(err_preview) > 70 else err_preview
        ans_preview = " ".join(final_answer.text.split())
        ans_preview = (ans_preview[:70] + "…") if len(ans_preview) > 70 else ans_preview
        where = f"step {m.step}" if m.step > 0 else "the initial state (drift-injected)"
        agents = sorted({a for a in (m.agent, final_answer.agent) if a})
        summary = (
            f"tool {tool!r} returned an error at {where} ({err_preview!r}); the final "
            f"answer at step {final_answer.step} ({ans_preview!r}) mentions no failure, "
            f"and the tool was never retried"
        )
        if bystander_hedge is not None:
            hedge = " ".join(bystander_hedge.text.split())
            hedge = (hedge[:60] + "…") if len(hedge) > 60 else hedge
            summary += (
                f"; a different agent ({bystander_hedge.agent or 'unknown'}) hedged "
                f"({hedge!r}) but that never reached the final answer"
            )
        findings.append(FailureRecord(
            timestep=final_answer.step,
            failure_type=NAME,
            agents_involved=agents,
            evidence_action_ids=[],
            summary=summary,
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
    "STRONG_ACK_TOKENS",
    "WEAK_ACK_TOKENS",
    "MAST_MODES",
    "NAME",
    "SOURCE",
    "SUMMARY",
    "detect",
    "detect_from_text",
]
