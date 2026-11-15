"""Detector: one model turn asked for the same tool call twice — and both ran.

Failure: an AI message carries two (or more) tool calls with the same tool
name and byte-identical arguments, and the tool layer executed each of
them. For a search that wastes a call. For `send_email`, `charge_card`,
`create_ticket` or `place_order` it is the double side effect every
non-idempotent API is built to prevent: two emails, two charges, two
tickets — and the conversation looks normal afterwards, because each call
got its own perfectly ordinary result.

LLMs emit parallel tool calls freely (OpenAI's parallel function calling
is on by default), and nothing in LangGraph's ToolNode de-duplicates them.
The model-layer chaos fault `model_duplicate_tool_call` plants exactly
this, so the graph's handling is testable before a model does it on its
own. Live, gpt-4o-mini react agent: the tool ran twice in 3/3 rollouts and
no other detector fired — the reason this one exists.

Fires once per (turn, tool, arguments) group when ALL hold:
  - one AI message has ≥ 2 tool calls with the same name and identical
    JSON-serialised arguments (calls with drift_call_* ids excluded),
  - at least two of those calls received a tool result that is not an
    error (they actually executed),
  - the tool is not a routing/handoff tool (`transfer_to_*`, …).
Calls repeated ACROSS turns are not this detector's business: a retry
after a result is a decision (`tool_call_loop` / `retry_amplification`
own the runaway case). Side-effect tools (same verb list as
`novel_tool_call`) are flagged first in the summary.
"""
from __future__ import annotations

import json

from drift.failures.base import FailureRecord
from drift.failures.library.base import CoordinationDetectorContext, conversation
from drift.failures.library.novel_tool_call import ROUTING_TOOL_PATTERN, is_side_effect_tool

NAME = "duplicate_tool_call"
SUMMARY = (
    "One model turn requested the same tool call (same name, same arguments) twice and both "
    "executed — a double side effect for any non-idempotent tool."
)
MAST_MODES = ["1.3"]  # Step Repetition — at the tool-call layer, inside one turn
SOURCE = (
    "OpenAI parallel function calling (on by default; no de-duplication in LangGraph's ToolNode); "
    "Stripe/AWS idempotency-key guidance for non-idempotent APIs; MAST 1.3 (Step Repetition)"
)


def _key(tc: dict) -> tuple[str, str]:
    try:
        args = json.dumps(tc.get("args"), sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        args = str(tc.get("args"))
    return (str(tc.get("name") or ""), args)


def detect(ctx: CoordinationDetectorContext) -> list[FailureRecord]:
    msgs = conversation(ctx)
    # tool_call_id -> did it get a non-error result?
    executed: dict[str, bool] = {}
    for m in msgs:
        if m.role == "tool" and m.tool_call_id is not None:
            executed[str(m.tool_call_id)] = executed.get(str(m.tool_call_id), False) or not m.is_error_result

    findings: list[FailureRecord] = []
    for m in msgs:
        # No step filter: under a checkpoint fault the duplicated request is
        # already in the resumed run's starting state (step 0), but both
        # executions happen inside the run — and execution is what fires.
        if m.role != "ai" or len(m.tool_calls) < 2:
            continue
        groups: dict[tuple[str, str], list[dict]] = {}
        for tc in m.tool_calls:
            cid = tc.get("id")
            if isinstance(cid, str) and cid.startswith("drift_call_"):
                continue
            name = str(tc.get("name") or "")
            if not name or ROUTING_TOOL_PATTERN.search(name):
                continue
            groups.setdefault(_key(tc), []).append(tc)
        for (name, args), calls in groups.items():
            if len(calls) < 2:
                continue
            ran = [c for c in calls if executed.get(str(c.get("id")))]
            if len(ran) < 2:
                continue
            side = is_side_effect_tool(name)
            head = "SIDE-EFFECT tool" if side else "tool"
            args_s = " ".join(args.split())
            args_s = (args_s[:80] + "…") if len(args_s) > 80 else args_s
            summary = (
                f"one turn{f" at step {m.step}" if m.step > 0 else ""} requested {head} {name!r} {len(calls)}× with identical "
                f"arguments {args_s} and {len(ran)} of those calls executed"
            )
            if side:
                summary += " — the action happened that many times"
            findings.append(FailureRecord(
                timestep=m.step,
                failure_type=NAME,
                agents_involved=[m.agent] if m.agent else [],
                evidence_action_ids=[],
                summary=summary,
                snapshot_timestep=m.step,
            ))
    findings.sort(key=lambda f: (0 if "SIDE-EFFECT" in f.summary else 1, f.timestep))
    return findings


def detect_from_text(transcript: str) -> list[FailureRecord]:
    """No text-only variant: needs structured tool calls and their results."""
    return []


__all__ = ["MAST_MODES", "NAME", "SOURCE", "SUMMARY", "detect", "detect_from_text"]
