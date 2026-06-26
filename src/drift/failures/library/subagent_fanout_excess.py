"""Detector: orchestrator spawned far more subagents than the task required.

Failure: a parent/orchestrator agent dispatches K subagents for what should
have been a much smaller fanout. Anthropic's engineering blog cites their
own named incident — "spawning 50 subagents for trivial queries" — and notes
the system used ~15x the tokens of single-agent chat as their viability
threshold. Fanout cost is multiplicative; one fanout decision can swallow
an entire budget.

Why it matters: cost regressions in MAS are *invisible* without instrumentation
because each subagent looks like normal work. The orchestrator's decision
to fanout is the leverage point, not any individual subagent's behavior.

This detector fires when, inside a single trace:
  - Number of distinct subagents OR concurrent subagents at peak exceeds
    `max_subagents` (default 8 — well below Anthropic's 50, well above
    typical map-reduce of 2-5)
  - OR: ratio of subagents to *measurable produced outputs* exceeds
    `max_subagents_per_output` (default 3.0 — i.e., if the orchestrator
    spawned 9 subagents and only 3 distinct outputs landed in the final
    state, the rest were wasted)

The detector intentionally errs on the side of false negatives: it requires
either a hard count OR an evident overspawn-to-output ratio. We don't fire
on every parallel fanout — only the suspicious ones.
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
    collect_string_values,
)

NAME = "subagent_fanout_excess"
SUMMARY = "Orchestrator spawned more subagents than the task warranted."
MAST_MODES = ["2.3"]  # Task derailment — MAST doesn't enumerate fanout directly
SOURCE = "Anthropic multi-agent research postmortem (50-subagent incident, ~15x token threshold)"

DEFAULT_MAX_SUBAGENTS = 8
DEFAULT_MAX_SUBAGENTS_PER_OUTPUT = 3.0
DEFAULT_MIN_SUBAGENTS_FOR_RATIO = 4  # don't apply ratio rule to tiny fanouts

# Common orchestrator-shaped role / agent names.
_ORCHESTRATOR_PATTERN = re.compile(
    r"orchestrat|coordinat|manager|supervisor|planner|router|dispatcher",
    re.I,
)


def _count_outputs(state: dict[str, Any] | None) -> int:
    """Approximate produced-output count from the final state.

    Heuristic: count non-empty string fields + total elements in top-level
    list/dict containers. A graph that returned `{"reports": [r1, r2, r3]}`
    counts as 3, not 1. A graph that returned `{"summary": "..."}` counts
    as 1. Designed to be cheap and roughly right rather than precise.
    """
    if not state:
        return 0
    total = 0
    for v in state.values():
        if isinstance(v, str) and v.strip():
            total += 1
        elif isinstance(v, (list, tuple, set)):
            total += sum(1 for x in v if x not in (None, "", {}, []))
        elif isinstance(v, dict):
            total += len(v)
        elif isinstance(v, (int, float, bool)) and v:
            total += 1
    return total


def _looks_like_orchestrator(agent: str, roles_by_agent: dict[str, str]) -> bool:
    declared = roles_by_agent.get(agent, "").lower()
    if declared in ("orchestrator", "supervisor", "manager", "planner", "coordinator", "router"):
        return True
    return bool(_ORCHESTRATOR_PATTERN.search(agent or ""))


@dataclass
class _FanoutObservation:
    parent: str | None
    subagents: list[str]
    super_step_count: int


def _observe_fanout(ctx: CoordinationDetectorContext) -> _FanoutObservation:
    """Identify a parent + the set of subagents.

    Strategy:
      1. If exactly one agent looks like an orchestrator and other agents'
         steps follow it, treat them as subagents.
      2. Otherwise: the parent is the first-step agent if it shows up before
         many other distinct agents do. Subagents = all other distinct agents.

    Returns observation even when no clear parent is identifiable — the
    detector falls back to "distinct non-system agent count" for the
    threshold check in that case.
    """
    agents_in_order = [s.agent for s in ctx.steps if s.agent and not s.agent.startswith("__")]
    distinct = []
    seen: set[str] = set()
    for a in agents_in_order:
        if a not in seen:
            seen.add(a)
            distinct.append(a)

    parent: str | None = None
    for a in distinct:
        if _looks_like_orchestrator(a, ctx.roles_by_agent):
            parent = a
            break
    if parent is None and distinct:
        # Fall back to first agent if many distinct subordinates came after it.
        first = distinct[0]
        if len(distinct) - 1 >= DEFAULT_MIN_SUBAGENTS_FOR_RATIO - 1:
            parent = first

    subagents = [a for a in distinct if a != parent]
    return _FanoutObservation(
        parent=parent,
        subagents=subagents,
        super_step_count=len(agents_in_order),
    )


def detect(
    ctx: CoordinationDetectorContext,
    *,
    max_subagents: int = DEFAULT_MAX_SUBAGENTS,
    max_subagents_per_output: float = DEFAULT_MAX_SUBAGENTS_PER_OUTPUT,
    min_subagents_for_ratio: int = DEFAULT_MIN_SUBAGENTS_FOR_RATIO,
) -> list[FailureRecord]:
    """Fire if either the hard subagent count or the subagents/output ratio
    exceeds threshold."""
    obs = _observe_fanout(ctx)
    n_sub = len(obs.subagents)

    findings: list[FailureRecord] = []
    last_step = ctx.steps[-1].step if ctx.steps else 0

    # Hard count rule.
    if n_sub >= max_subagents:
        findings.append(FailureRecord(
            timestep=last_step,
            failure_type=NAME,
            agents_involved=([obs.parent] if obs.parent else []) + obs.subagents[: max_subagents],
            evidence_action_ids=[],
            summary=(
                f"orchestrator {obs.parent!r} spawned {n_sub} distinct subagents "
                f"({', '.join(obs.subagents[:6])}{'...' if n_sub > 6 else ''}) — "
                f"threshold {max_subagents}"
            ),
            snapshot_timestep=last_step,
        ))
        return findings  # short-circuit; ratio rule would double-fire

    # Ratio rule — only when the fanout is at least moderately large to begin with.
    if n_sub >= min_subagents_for_ratio and ctx.steps:
        final_state = ctx.steps[-1].state_after or {}
        n_outputs = _count_outputs(final_state)
        if n_outputs == 0:
            ratio = float("inf")
        else:
            ratio = n_sub / n_outputs
        if ratio > max_subagents_per_output:
            findings.append(FailureRecord(
                timestep=last_step,
                failure_type=NAME,
                agents_involved=([obs.parent] if obs.parent else []) + obs.subagents,
                evidence_action_ids=[],
                summary=(
                    f"orchestrator {obs.parent!r} spawned {n_sub} subagents "
                    f"but only {n_outputs} measurable output(s) landed "
                    f"(ratio {ratio:.1f} > {max_subagents_per_output})"
                ),
                snapshot_timestep=last_step,
            ))
    return findings


# ---------------------------------------------------------------------------
# Raw-text variant — for MAST validation.
# ---------------------------------------------------------------------------


# "Calling Agent X", "Spawned subagent X", "Delegating to X", "send_message(app_name='X')"
_SPAWN_PATTERNS = [
    re.compile(r"\bcall(?:ing)?\s+(?:agent|subagent|tool)\s+['\"]?([A-Za-z0-9_]{2,40})", re.I),
    re.compile(r"\bspawn(?:ed|ing)?\s+(?:agent|subagent)\s+['\"]?([A-Za-z0-9_]{2,40})", re.I),
    re.compile(r"\bdelegat(?:e|ing)\s+to\s+['\"]?([A-Za-z0-9_]{2,40})", re.I),
    re.compile(r"\bsend_message\(\s*app_name\s*=\s*['\"]([A-Za-z0-9_]{2,40})['\"]", re.I),
    re.compile(r"\b(?:transfer|handoff|dispatch)\s+to\s+['\"]?([A-Za-z0-9_]{2,40})", re.I),
    re.compile(r"^Entering\s+([A-Za-z0-9_ ]{2,40})\s+Agent", re.I | re.M),
]


def detect_from_text(
    transcript: str,
    *,
    max_subagents: int = DEFAULT_MAX_SUBAGENTS,
) -> list[FailureRecord]:
    """Text variant: count distinct subagent invocations across the trace.

    Looser than the structured variant — no ratio check, only the hard
    count. Will miss frameworks whose spawn vocabulary doesn't match
    these patterns.
    """
    if not transcript:
        return []
    subs: set[str] = set()
    for pat in _SPAWN_PATTERNS:
        for m in pat.finditer(transcript):
            name = m.group(1).strip().lower()
            if name and not name.startswith("_"):
                subs.add(name)
    if len(subs) >= max_subagents:
        return [FailureRecord(
            timestep=0,
            failure_type=NAME,
            agents_involved=sorted(subs)[: max_subagents],
            evidence_action_ids=[],
            summary=(
                f"text scan: {len(subs)} distinct subagents invoked "
                f"({', '.join(sorted(subs)[:6])}{'...' if len(subs) > 6 else ''}) — "
                f"threshold {max_subagents}"
            ),
            snapshot_timestep=0,
        )]
    return []


__all__ = [
    "DEFAULT_MAX_SUBAGENTS",
    "DEFAULT_MAX_SUBAGENTS_PER_OUTPUT",
    "MAST_MODES",
    "NAME",
    "SOURCE",
    "SUMMARY",
    "detect",
    "detect_from_text",
]
