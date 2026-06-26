"""Coordination-failure detector library — base types.

Each detector in this library targets a named coordination failure documented
in the multi-agent-systems literature (MAST, Anthropic engineering blog,
Cognition, IBM hidden-cycles paper). Detectors are deterministic Python
functions over a `CoordinationDetectorContext`; they don't call an LLM, they
read the agent trace directly. They complement the LLM judge: cheaper,
faster, and produce structured `failure_type` strings the judge can't.

There are two paths a detector accepts input through:

  1. Adapter trace path — `from_adapter_trace(trace, baseline_state, ...)`.
     Used inside `drift_test_async` after each perturbation. The trace is a
     list of `{step, node, update, state_after}` dicts (langgraph-shaped).

  2. Native DetectorContext path — `from_native(ctx)`. Used by `drift.run`
     style topologies that produce `Action` objects directly.

  3. Raw-text path — each detector also exposes `detect_from_text(transcript)`
     for unstructured trace transcripts (MAST dataset, captured logs). Lower
     precision than the structured path; ships so we can compute apples-to-
     apples F1 against the MAST human-labelled subset.

Both structured paths normalize into one `CoordinationDetectorContext` so
detector code is path-agnostic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from drift.agents.base import Action
from drift.failures.base import DetectorContext, FailureRecord


@dataclass
class TraceStep:
    """One super-step (or one Action) normalized to a uniform shape.

    Fields the library detectors actually read:
      - step:       monotonic 1-based step index
      - agent:      who acted (node name in langgraph, agent_name in native)
      - kind:       optional structured action kind (native) or `"node:<n>"` (adapter)
      - update:     dict of fields written this step (adapter), or
                    {"kind": ..., "target": ..., "rationale": ...} (native)
      - state_after: snapshot of running state after this step (adapter only;
                    empty dict for native because native re-derives from history)
      - rationale:  free-form text explanation if available (action.rationale)
    """

    step: int
    agent: str
    kind: str
    update: dict[str, Any] = field(default_factory=dict)
    state_after: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class CoordinationDetectorContext:
    """Framework-agnostic view a library detector consumes.

    Built from either the adapter trace or a native `DetectorContext`.
    Detectors must not reach past these fields — that's how the same
    detector code runs in both worlds.

    roles_by_agent maps agent/node name to a role label (e.g. "verifier",
    "planner", "executor"). Empty when the user hasn't declared roles —
    detectors that need roles fall back to name-pattern matching.

    initial_state and baseline_state are both optional. initial_state is what
    was fed into the graph; baseline_state is the unperturbed final state
    (only present inside the adapter context, where it anchors delta-style
    detectors). When neither is present, detectors that need one degrade
    gracefully (emit nothing rather than crash).
    """

    steps: list[TraceStep]
    initial_state: dict[str, Any] | None = None
    baseline_state: dict[str, Any] | None = None
    roles_by_agent: dict[str, str] = field(default_factory=dict)

    # ---- helpers ---------------------------------------------------------

    def role_of(self, agent: str) -> str:
        """Explicit role if declared, else empty string (no name-pattern match here;
        detectors that want pattern matching apply their own regex)."""
        return self.roles_by_agent.get(agent, "")

    def steps_by_agent(self) -> dict[str, list[TraceStep]]:
        out: dict[str, list[TraceStep]] = {}
        for s in self.steps:
            out.setdefault(s.agent, []).append(s)
        return out


CoordinationDetector = Callable[[CoordinationDetectorContext], list[FailureRecord]]


# ---------------------------------------------------------------------------
# Builders — wrap either input shape into a CoordinationDetectorContext.
# ---------------------------------------------------------------------------


def from_adapter_trace(
    trace: list[dict],
    *,
    initial_state: dict | None = None,
    baseline_state: dict | None = None,
    roles_by_agent: dict[str, str] | None = None,
) -> CoordinationDetectorContext:
    """Build a coordination context from an adapter-shaped trace.

    `trace` is what `_stream_or_invoke` returns: a list of
    `{step, node, update, state_after}` records.
    """
    steps: list[TraceStep] = []
    for entry in trace:
        update = entry.get("update") or {}
        if not isinstance(update, dict):
            update = {"_value": update}
        rationale = ""
        # Common rationale-shaped keys nodes write.
        for k in ("rationale", "reasoning", "explanation", "thought"):
            v = update.get(k)
            if isinstance(v, str) and v:
                rationale = v
                break
        steps.append(TraceStep(
            step=int(entry.get("step", len(steps) + 1)),
            agent=str(entry.get("node", "")),
            kind=f"node:{entry.get('node', '')}",
            update=dict(update),
            state_after=dict(entry.get("state_after") or {}),
            rationale=rationale,
        ))
    return CoordinationDetectorContext(
        steps=steps,
        initial_state=initial_state,
        baseline_state=baseline_state,
        roles_by_agent=dict(roles_by_agent or {}),
    )


def from_native(
    ctx: DetectorContext,
    *,
    roles_by_agent: dict[str, str] | None = None,
) -> CoordinationDetectorContext:
    """Build a coordination context from drift's native DetectorContext.

    Native `Action`s become TraceSteps with `update` = action fields and
    `state_after` derived from the matching history snapshot (best-effort
    by timestep). Empty state_after is fine — most library detectors that
    care about state delta come from the adapter path anyway.
    """
    snapshots_by_t = {}
    for snap in ctx.history.all_snapshots():
        snapshots_by_t[getattr(snap, "timestep", -1)] = snap

    steps: list[TraceStep] = []
    for i, a in enumerate(ctx.actions, start=1):
        update = {
            "kind": a.kind,
            "target_case_id": a.target_case_id,
            "rationale": a.rationale,
            "referenced_policy_version": a.referenced_policy_version,
        }
        snap = snapshots_by_t.get(a.timestep)
        state_after = snap.model_dump() if snap is not None else {}
        steps.append(TraceStep(
            step=a.timestep if a.timestep > 0 else i,
            agent=a.agent_name,
            kind=a.kind,
            update=update,
            state_after=state_after,
            rationale=a.rationale,
        ))
    return CoordinationDetectorContext(
        steps=steps,
        initial_state=None,
        baseline_state=None,
        roles_by_agent=dict(roles_by_agent or {}),
    )


# ---------------------------------------------------------------------------
# Shared utility helpers detectors reuse.
# ---------------------------------------------------------------------------


def role_matches(
    agent: str,
    roles_by_agent: dict[str, str],
    role_name: str,
    name_pattern: re.Pattern[str] | str | None = None,
) -> bool:
    """Resolve an agent's role with explicit-then-pattern fallback.

    1. If `roles_by_agent[agent] == role_name`, True.
    2. Else if `name_pattern` is a regex and the agent name matches, True.
    3. Else False.

    Detectors should prefer this helper over ad-hoc role checks so the
    explicit/regex precedence is consistent across the library.
    """
    declared = roles_by_agent.get(agent, "").lower()
    if declared == role_name.lower():
        return True
    if name_pattern is None:
        return False
    pat = name_pattern if isinstance(name_pattern, re.Pattern) else re.compile(name_pattern, re.I)
    return bool(pat.search(agent or ""))


def collect_string_values(value: Any, out: list[str], _depth: int = 0) -> None:
    """Recursively collect string leaves from a nested value into `out`.

    Detectors use this to scan an update dict for keyword matches without
    caring about the dict's shape. Bounded depth so a self-referential
    structure can't hang us.
    """
    if _depth > 8:
        return
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            collect_string_values(v, out, _depth + 1)
    elif isinstance(value, (list, tuple, set)):
        for v in value:
            collect_string_values(v, out, _depth + 1)


def lowercased_text_blob(update: dict[str, Any]) -> str:
    """Flatten an update dict's string content into one lowercase blob.

    Cheap textual matching surface. Order-preserving so regex anchors that
    care about adjacency still work loosely.
    """
    parts: list[str] = []
    collect_string_values(update, parts)
    return " ".join(parts).lower()


def any_token(blob: str, tokens: Iterable[str]) -> str | None:
    """Return the first matching token from `tokens` found in `blob` (lowercased
    word-boundary match), else None."""
    for tok in tokens:
        tok = tok.strip().lower()
        if not tok:
            continue
        # word boundary on each side; \b on punctuation tokens (e.g. "n/a") would
        # under-match, so allow space-separated or string-boundary too.
        if re.search(rf"(?<![a-z0-9_]){re.escape(tok)}(?![a-z0-9_])", blob):
            return tok
    return None


__all__ = [
    "CoordinationDetector",
    "CoordinationDetectorContext",
    "TraceStep",
    "any_token",
    "collect_string_values",
    "from_adapter_trace",
    "from_native",
    "lowercased_text_blob",
    "role_matches",
]
