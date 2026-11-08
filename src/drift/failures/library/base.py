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
      - subagent:   for a subgraph step started from inside a tool call, the
                    sub-agent's name recovered from the dispatching call
                    ("critic"); "" otherwise. `agent` keeps the node path.
    """

    step: int
    agent: str
    kind: str
    update: dict[str, Any] = field(default_factory=dict)
    state_after: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    subagent: str = ""

    @property
    def actor(self) -> str:
        """Human-facing label: `tools/model` → `critic (tools/model)` when the
        sub-agent is known. Detectors that name agents in summaries use this."""
        return f"{self.subagent} ({self.agent})" if self.subagent else self.agent


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

    required_followups and policy_field are provisional, single-hint config
    for the two detectors that can't run without SOME user-declared
    structure (premature_termination, policy_inconsistency_across_steps).
    Both are plain dict[str, str] / str — not a general rule DSL — because
    building the real extensible DSL is separate, larger, future work; this
    is the minimal shape that makes those two detectors usable today. Empty
    by default, in which case both detectors correctly emit nothing rather
    than guessing at structure nobody declared.
    """

    steps: list[TraceStep]
    initial_state: dict[str, Any] | None = None
    baseline_state: dict[str, Any] | None = None
    roles_by_agent: dict[str, str] = field(default_factory=dict)
    # trigger agent/node name -> required follow-up agent/node name.
    required_followups: dict[str, str] = field(default_factory=dict)
    # state field name holding a policy/config version, e.g. "policy_version".
    policy_field: str = ""

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
    required_followups: dict[str, str] | None = None,
    policy_field: str | None = None,
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
            subagent=str(entry.get("subagent") or ""),
        ))
    return CoordinationDetectorContext(
        steps=steps,
        initial_state=initial_state,
        baseline_state=baseline_state,
        roles_by_agent=dict(roles_by_agent or {}),
        required_followups=dict(required_followups or {}),
        policy_field=policy_field or "",
    )


def from_native(
    ctx: DetectorContext,
    *,
    roles_by_agent: dict[str, str] | None = None,
    required_followups: dict[str, str] | None = None,
    policy_field: str | None = None,
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
        required_followups=dict(required_followups or {}),
        policy_field=policy_field or "",
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


# ---------------------------------------------------------------------------
# Chat-message view — what the message-aware detectors read.
# ---------------------------------------------------------------------------
#
# The adapter normalises langchain messages into dicts (see
# drift.adapters.langgraph._message_to_dict); user-supplied initial states
# may carry OpenAI-style role dicts instead. `Msg` is the one shape the
# message-aware detectors consume, built from either.


@dataclass
class Msg:
    role: str                      # "human" | "ai" | "tool" | "system" | ""
    text: str                      # flattened text content
    step: int                      # trace step whose update introduced it
    agent: str                     # node that wrote it
    name: str = ""                 # message .name (agent name on multi-agent graphs)
    tool_calls: list[dict] = field(default_factory=list)   # [{"name","args","id"}]
    tool_call_id: str | None = None
    status: str = ""               # "" | "error" (tool results)
    index: int = 0                 # position in the overall conversation

    @property
    def is_error_result(self) -> bool:
        if self.role != "tool":
            return False
        if self.status == "error":
            return True
        head = self.text.lstrip()[:40].lower()
        return head.startswith("error") or "traceback (most recent call last)" in self.text.lower()

    @property
    def is_drift_injected(self) -> bool:
        """True for the tool call / result pair a drift chaos pattern seeded
        into the initial state (ids `drift_call_*`)."""
        if self.tool_call_id and self.tool_call_id.startswith("drift_call_"):
            return True
        return any(
            isinstance(tc.get("id"), str) and tc["id"].startswith("drift_call_")
            for tc in self.tool_calls
        )


def _message_dicts_in(value: Any) -> list[Any]:
    """Every list-of-messages found among a dict's top-level values, concatenated."""
    from drift.chaos import messages as _m  # local import: keep failures importable alone

    if not isinstance(value, dict):
        return []
    out: list[Any] = []
    for v in value.values():
        if _m.is_messages_list(v):
            out.extend(v)
    return out


def messages_in_step(step: TraceStep) -> list[Msg]:
    """Messages this step's `update` introduced, as `Msg` records."""
    from drift.chaos import messages as _m

    out: list[Msg] = []
    for raw in _message_dicts_in(step.update):
        calls = [
            {"name": c.get("name"), "args": c.get("args"), "id": c.get("id")}
            for c in _m.message_tool_calls(raw)
        ]
        tcid = raw.get("tool_call_id") if isinstance(raw, dict) else getattr(raw, "tool_call_id", None)
        status = raw.get("status", "") if isinstance(raw, dict) else getattr(raw, "status", "")
        name = raw.get("name", "") if isinstance(raw, dict) else getattr(raw, "name", "")
        out.append(Msg(
            role=_m.message_role(raw),
            text=_m.message_text(raw),
            step=step.step,
            agent=step.agent,
            name=str(name or ""),
            tool_calls=calls,
            tool_call_id=str(tcid) if tcid is not None else None,
            status=str(status or "") if status != "success" else "",
        ))
    return out


def _msg_fingerprint(m: Msg) -> tuple:
    return (
        m.role, m.text, m.tool_call_id,
        tuple((str(c.get("name")), repr(c.get("args"))) for c in m.tool_calls),
    )


def conversation(ctx: CoordinationDetectorContext) -> list[Msg]:
    """The conversation as the nodes wrote it: every step's new messages in
    order, plus — first — the messages already present in `initial_state`
    (step 0, agent "" ) so injected context the graph was handed is visible
    too. Indices are assigned in this order.

    Re-emit de-duplication. When a sub-agent runs as a subgraph (a
    `create_react_agent` under a supervisor, say) its inner steps
    (`math_expert/agent`, `math_expert/tools`) emit messages, and then the
    parent node (`math_expert`) emits its whole output — which can be the
    ENTIRE conversation so far, old messages included — as its own update;
    `add_messages` de-duplicates by id on the real state, but a naive read
    of the updates would count every message two or three times and turn
    one handoff into a "tool call loop". So, for a root-level step, only
    the messages the root state actually GREW by (`len(state_after
    messages) - len(previous)`, taken from the tail of the update) count as
    new, minus any already attributed to this node's inner steps. Inner
    (namespaced) steps keep every message in their update: the root state
    doesn't move until the parent commits, so growth can't be read there,
    and their updates are genuinely new. A real repeat (a loop) is a new
    message that grows the state, so it is kept. Traces whose `state_after`
    doesn't carry a messages channel (stubs, legacy accumulators) fall back
    to counting every update message.
    """
    out: list[Msg] = []
    prev_count: int | None = None
    if ctx.initial_state:
        seed = TraceStep(step=0, agent="", kind="initial", update=dict(ctx.initial_state))
        out.extend(messages_in_step(seed))
        prev_count = len(_message_dicts_in(ctx.initial_state))
    emitted_by_agent: dict[str, set[tuple]] = {}
    for s in ctx.steps:
        msgs = messages_in_step(s)
        is_inner = "/" in s.agent
        after_count = len(_message_dicts_in(s.state_after)) if s.state_after else None
        if not is_inner and after_count is not None and prev_count is not None:
            growth = after_count - prev_count
            if 0 <= growth < len(msgs):
                msgs = msgs[len(msgs) - growth:] if growth else []
            prefix = f"{s.agent}/"
            inner_fps: set[tuple] = set()
            for agent, fps in emitted_by_agent.items():
                if agent.startswith(prefix):
                    inner_fps |= fps
            msgs = [m for m in msgs if _msg_fingerprint(m) not in inner_fps]
        if not is_inner and after_count is not None:
            prev_count = after_count
        for m in msgs:
            out.append(m)
            emitted_by_agent.setdefault(s.agent, set()).add(_msg_fingerprint(m))
    for i, m in enumerate(out):
        m.index = i
    return out


def pair_tool_results(msgs: list[Msg]) -> dict[int, int]:
    """Map conversation index of each AI tool call's owning message -> index of
    its tool result. Pairs by tool_call_id when the result names one that is
    outstanding, else positionally (oldest outstanding call first)."""
    outstanding: list[tuple[str | None, int]] = []   # (call id, ai msg index)
    pairs: dict[int, int] = {}
    for m in msgs:
        if m.role == "ai" and m.tool_calls:
            for tc in m.tool_calls:
                outstanding.append((tc.get("id"), m.index))
        elif m.role == "tool":
            hit = None
            if m.tool_call_id is not None:
                for k, (cid, _) in enumerate(outstanding):
                    if cid == m.tool_call_id:
                        hit = k
                        break
            if hit is None and outstanding:
                hit = 0
            if hit is not None:
                _, ai_idx = outstanding.pop(hit)
                pairs.setdefault(ai_idx, m.index)
    return pairs


__all__ = [
    "CoordinationDetector",
    "CoordinationDetectorContext",
    "Msg",
    "TraceStep",
    "any_token",
    "collect_string_values",
    "conversation",
    "from_adapter_trace",
    "from_native",
    "lowercased_text_blob",
    "messages_in_step",
    "pair_tool_results",
    "role_matches",
]
