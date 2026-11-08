"""Detector: under a perturbation the graph did far more work than the
baseline needed for the same task — a retry or re-planning storm.

Failure: the baseline answered with 2 tool calls and 3 LLM turns; after
one injected fault (a failed tool result, a garbled instruction, a missing
message) the same graph makes 7 tool calls and 9 LLM turns and may well
still land on the right answer. The outcome buckets — crashed / diverged /
unchanged — call that UNCHANGED. The bill does not. AgentChaos
(arXiv:2608.06790) measured LLM calls per task rising up to 4.24× baseline
under injected faults across six agent frameworks, because nothing in the
loop bounds "try again": the model sees an error-shaped tool message and
decides, unboundedly, to retry, re-plan, re-delegate.

Why it exists alongside `tool_call_loop`: that detector needs the SAME
call (name + args) with the SAME result three times — the degenerate loop.
This one measures volume regardless of shape: varied retries, a supervisor
re-delegating to a different specialist each time, a planner that keeps
adding steps. Different signal; different fix (a budget, not a cache).

Fires when the perturbed run's graph-made tool calls OR LLM turns are at
least `min_ratio` × the baseline's (default 2.0) AND at least `min_extra`
more in absolute terms (default 3 — a 1→2 bump is noise). "Graph-made"
is measured the same way on both sides. Preferred: from each run's
per-step messages via `conversation()` (which de-duplicates a parent
node re-emitting its subgraph's messages), because a supervisor compiled
with output_mode="last_message" never writes its sub-agents' tool calls
to the root state — the final state would under-report the work by
exactly the part that matters. That needs the baseline's trace, which
the adapter supplies on every perturbation. Without it (SDK callers who
only pass `baseline_state`) both sides fall back to final-state messages
minus initial-state messages, consistently. Messages already present in
the initial state — the user's history, drift's own injected call/result
pair — never count. Silent when there is no baseline to compare against
(the baseline run itself) or when the graph has no messages channel.
"""
from __future__ import annotations

from drift.failures.base import FailureRecord
from drift.failures.library.base import (
    CoordinationDetectorContext,
    Msg,
    TraceStep,
    conversation,
    messages_in_step,
)

NAME = "retry_amplification"
SUMMARY = (
    "Under a perturbation the graph made ≥2× the tool calls or LLM turns the baseline "
    "needed — a retry / re-planning storm, whatever the final answer."
)
MAST_MODES = ["1.3"]  # Step repetition — at the budget level rather than the identical-call level
SOURCE = (
    "AgentChaos (arXiv:2608.06790): LLM calls per task up to 4.24× baseline under "
    "injected faults; MAST 1.3; tool-repetition-rate metric (arXiv:2601.17915)"
)

DEFAULT_MIN_RATIO = 2.0
DEFAULT_MIN_EXTRA = 3


def _work(msgs: list[Msg]) -> tuple[int, int]:
    """(graph-made tool calls, graph-made LLM turns) in a message list,
    excluding anything drift injected as chaos."""
    calls = 0
    turns = 0
    for m in msgs:
        if m.role != "ai" or m.is_drift_injected:
            continue
        turns += 1
        calls += sum(
            1 for tc in m.tool_calls
            if not (isinstance(tc.get("id"), str) and tc["id"].startswith("drift_call_"))
        )
    return calls, turns


def _state_work(state: dict | None, initial: dict | None) -> tuple[int, int] | None:
    """Work a run did: its final messages minus its initial messages.
    None when the state carries no messages channel."""
    if not state:
        return None
    final = messages_in_step(TraceStep(step=1, agent="", kind="final", update=dict(state)))
    if not final:
        return None
    calls, turns = _work(final)
    if initial:
        init = messages_in_step(TraceStep(step=0, agent="", kind="initial", update=dict(initial)))
        i_calls, i_turns = _work(init)
        calls, turns = max(0, calls - i_calls), max(0, turns - i_turns)
    return calls, turns


def _trace_work(ctx: CoordinationDetectorContext) -> tuple[int, int] | None:
    """Work a run did, from its per-step messages (graph-made = step ≥ 1).
    None when no step carried a messages channel."""
    msgs = [m for m in conversation(ctx) if m.step > 0]
    if not msgs and not any(messages_in_step(s) for s in ctx.steps):
        return None
    return _work(msgs)


def _both_sides(ctx: CoordinationDetectorContext) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """(perturbed work, baseline work), measured the same way."""
    base_ctx = ctx.baseline_view()
    if base_ctx is not None and base_ctx.steps:
        p = _trace_work(ctx)
        b = _trace_work(base_ctx)
        if p is not None and b is not None:
            return p, b
    if not ctx.steps:
        return None
    # Final-state fallback. The baseline's initial state isn't known here,
    # but it differs from the perturbed one only by the perturbation (one
    # message dropped / garbled / injected), and injected messages are
    # excluded by id, so the perturbed initial state is the right subtrahend.
    p = _state_work(ctx.steps[-1].state_after, ctx.initial_state)
    b = _state_work(ctx.baseline_state, ctx.initial_state)
    if p is None or b is None:
        return None
    return p, b


def _amplified(perturbed: int, base: int, min_ratio: float, min_extra: int) -> bool:
    if perturbed - base < min_extra:
        return False
    if base == 0:
        return True
    return perturbed >= min_ratio * base


def _ratio(perturbed: int, base: int) -> str:
    return f"{perturbed / base:.1f}×" if base else "baseline made none"


def detect(
    ctx: CoordinationDetectorContext,
    *,
    min_ratio: float = DEFAULT_MIN_RATIO,
    min_extra: int = DEFAULT_MIN_EXTRA,
) -> list[FailureRecord]:
    if ctx.baseline_state is None and ctx.baseline_steps is None:
        return []
    sides = _both_sides(ctx)
    if sides is None:
        return []
    (p_calls, p_turns), (b_calls, b_turns) = sides
    if p_turns == 0 and p_calls == 0:
        return []

    calls_hit = _amplified(p_calls, b_calls, min_ratio, min_extra)
    turns_hit = _amplified(p_turns, b_turns, min_ratio, min_extra)
    if not (calls_hit or turns_hit):
        return []

    parts = []
    if calls_hit:
        parts.append(f"{p_calls} tool calls vs {b_calls} at baseline ({_ratio(p_calls, b_calls)})")
    if turns_hit:
        parts.append(f"{p_turns} LLM turns vs {b_turns} at baseline ({_ratio(p_turns, b_turns)})")
    # Attribute to the nodes that wrote AI turns (from the per-step updates —
    # attribution, not counting, so re-emits don't matter here).
    by_agent: dict[str, int] = {}
    last_step = 0
    for st in ctx.steps:
        ai_here = [m for m in messages_in_step(st) if m.role == "ai" and not m.is_drift_injected]
        if ai_here:
            root = st.agent.split("/", 1)[0]
            by_agent[root] = by_agent.get(root, 0) + len(ai_here)
            last_step = max(last_step, st.step)
    agents = [a for a, _ in sorted(by_agent.items(), key=lambda kv: -kv[1])[:3] if a]
    return [FailureRecord(
        timestep=last_step,
        failure_type=NAME,
        agents_involved=agents,
        evidence_action_ids=[],
        summary=(
            "under this perturbation the graph made " + " and ".join(parts)
            + " — a retry / re-planning storm; the final answer may still be right, the cost is not"
        ),
        snapshot_timestep=last_step,
    )]


def detect_from_text(transcript: str) -> list[FailureRecord]:
    """No text-only variant: amplification is a comparison against a
    baseline run, which a single transcript does not carry."""
    return []


__all__ = [
    "DEFAULT_MIN_EXTRA",
    "DEFAULT_MIN_RATIO",
    "MAST_MODES",
    "NAME",
    "SOURCE",
    "SUMMARY",
    "detect",
    "detect_from_text",
]
