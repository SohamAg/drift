"""Tool-layer chaos — faults injected into a graph's tool calls at runtime.

Every other chaos pattern perturbs the *initial state* and reruns the
graph. That cannot model the failures a deployed graph actually meets
mid-run: the search API times out on the second call, the CRM returns an
empty page, a document comes back truncated, a fetched web page carries
an instruction. These patterns do. Each one targets ONE tool the graph
can call and ONE occurrence of it (`on_call`, 1-based), and replaces or
rewrites that call's result while the graph is running — the graph's own
model then has to cope with it, and the detector library reads what it
did (`silent_tool_failure`, `retry_amplification`, `novel_tool_call`,
`injected_instruction_followed`, `injected_tool_call_followed`,
`tainted_side_effect_call` all apply unchanged).

How it hooks in: LangGraph's `ToolNode` exposes `wrap_tool_call` /
`awrap_tool_call` interceptors (request, execute) → ToolMessage. drift
walks the compiled graph (subgraphs included), finds every ToolNode whose
tools include the target, and installs a wrapper for the duration of one
run — `ToolChaosEvent.install(graph)` is a context manager that restores
whatever wrapper was there before. Nothing is monkeypatched on the user's
tool objects. Tools called from hand-written nodes (not a ToolNode) are
out of reach; `discover_tool_patterns` only lists what it can actually
intercept, so a graph with no ToolNode gets no tool patterns rather than
patterns that silently never fire.

Faults, with the AgentChaos fault class each models:

    tool_error             crash     the call returned an API error (503)
    tool_timeout           crash     the call timed out
    tool_rate_limit        crash     429 — retry later
    tool_exception         crash     the tool RAISED (honours the node's handle_tool_errors)
    tool_empty             omission  the call returned nothing
    tool_truncated         value     the result was cut to its first 25%
    tool_stale             value     the result is flagged as a days-old cached copy
    tool_inject_instruction value    the real result + an injected instruction (canary)
    tool_hijack            value     the real result + a demand to call a tool with a canary

The injection faults differ from `inject_tool_result_instruction` /
`inject_tool_result_tool_hijack` in one way that matters: the poisoned
result is one the graph *asked for*, in the middle of its own run, so the
tool is already in context — live, the seeded hijack landed 3/3 "warm"
and 0/3 "cold". These are always warm.
"""
from __future__ import annotations

import contextlib
import inspect
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from drift.chaos.fuzzer import ChaosSpec
from drift.chaos.patterns import CANARY_PREFIX, AutoChaosEvent, _hijack_payload, _injection_payload
from drift.events.base import EventRecord
from drift.world import World

TOOL_FAULTS: dict[str, dict[str, str]] = {
    "tool_error": {"fault_class": "crash", "models": "the call returned an API error (503)"},
    "tool_timeout": {"fault_class": "crash", "models": "the call timed out"},
    "tool_rate_limit": {"fault_class": "crash", "models": "the call was rate-limited (429)"},
    "tool_exception": {"fault_class": "crash", "models": "the tool raised; the node's own error handling applies"},
    "tool_empty": {"fault_class": "omission", "models": "the call returned nothing"},
    "tool_truncated": {"fault_class": "value", "models": "the result was cut to its first 25%"},
    "tool_stale": {"fault_class": "value", "models": "the result is a days-old cached copy"},
    "tool_inject_instruction": {"fault_class": "value",
                                "models": "the real result plus an injected instruction (canary-verified)"},
    "tool_hijack": {"fault_class": "value",
                    "models": "the real result plus a demand to call a tool with a canary argument"},
}
FAULT_LAYER_TOOL = "tool"
FAULT_LAYER_STATE = "state"
#: The faults the auto planner draws from below `exhaustive`. Timeout and
#: rate-limit read the same as an error to the model; `tool_exception`
#: crashes the graph outright under the default ToolNode error handling,
#: and `tool_stale` is the weakest signal — all four are still one
#: `chaos:` line or `--intensity exhaustive` away.
DEFAULT_AUTO_FAULTS: tuple[str, ...] = (
    "tool_error", "tool_empty", "tool_truncated", "tool_inject_instruction", "tool_hijack",
)


class ToolChaosError(RuntimeError):
    """Raised by the `tool_exception` fault inside the tool call."""


# ---------------------------------------------------------------------------
# Finding the ToolNodes in a compiled graph
# ---------------------------------------------------------------------------


def _is_tool_node(obj: Any) -> bool:
    return (
        obj is not None
        and type(obj).__name__ == "ToolNode"
        and isinstance(getattr(obj, "_tools_by_name", None), dict)
    )


def _unwrap(obj: Any) -> Iterator[Any]:
    """Yield `obj` and anything it wraps that could be a ToolNode."""
    yield obj
    for attr in ("bound", "runnable", "func", "afunc", "first", "last"):
        inner = getattr(obj, attr, None)
        if inner is not None and inner is not obj:
            yield inner
    for attr in ("steps", "middle"):
        seq = getattr(obj, attr, None)
        if isinstance(seq, (list, tuple)):
            yield from seq


def find_tool_nodes(graph: Any, *, _depth: int = 0, _seen: set[int] | None = None) -> list[tuple[str, Any]]:
    """`[(node_path, ToolNode)]` for every ToolNode in `graph`, subgraphs
    included (`parent/child` paths). Empty for anything that is not a
    compiled LangGraph."""
    seen = _seen if _seen is not None else set()
    out: list[tuple[str, Any]] = []
    if _depth > 6 or id(graph) in seen:
        return out
    seen.add(id(graph))
    # `graph.with_config(...)` wraps the Pregel in a RunnableBinding.
    pregel = getattr(graph, "bound", None) if not hasattr(graph, "nodes") else graph
    nodes = getattr(pregel, "nodes", None)
    if not isinstance(nodes, dict):
        return out
    for name, node in nodes.items():
        if name in ("__start__", "__end__"):
            continue
        found = False
        for cand in _unwrap(node):
            if _is_tool_node(cand):
                out.append((str(name), cand))
                found = True
                break
        if found:
            continue
        # A node whose runnable is itself a compiled graph (a sub-agent).
        for cand in _unwrap(node):
            if cand is node:
                continue
            if isinstance(getattr(cand, "nodes", None), dict):
                for sub_name, sub_node in find_tool_nodes(cand, _depth=_depth + 1, _seen=seen):
                    out.append((f"{name}/{sub_name}", sub_node))
    return out


def tool_names(graph: Any) -> list[str]:
    """Every tool name reachable through a ToolNode, sorted, deduplicated."""
    names: set[str] = set()
    for _, node in find_tool_nodes(graph):
        names.update(str(n) for n in node._tools_by_name)
    return sorted(names)


# ---------------------------------------------------------------------------
# The event
# ---------------------------------------------------------------------------


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and isinstance(c.get("text"), str):
                parts.append(c["text"])
            elif isinstance(c, str):
                parts.append(c)
        return "\n".join(parts)
    return str(content) if content is not None else ""


@dataclass
class ToolFaultHit:
    """What the wrapper did, for the event summary and the trace."""
    tool: str
    call_index: int
    call_id: str
    node_path: str
    args: Any
    original_preview: str = ""


class ToolChaosEvent(AutoChaosEvent):
    """One runtime fault on one occurrence of one tool. `apply(world)` is a
    recorded no-op — the state is untouched; the fault lands when the
    graph runs with `install(graph)` active."""

    pattern = "tool_fault"
    fault_class = "crash"
    layer = FAULT_LAYER_TOOL

    def __init__(self, tool: str, *, fault: str, on_call: int = 1, seed: int | None = None) -> None:
        if fault not in TOOL_FAULTS:
            raise ValueError(f"unknown tool fault {fault!r}; one of {', '.join(TOOL_FAULTS)}")
        if not isinstance(on_call, int) or on_call < 1:
            raise ValueError("on_call must be an int >= 1")
        self.pattern = fault
        self.fault_class = TOOL_FAULTS[fault]["fault_class"]
        super().__init__(tool, seed=seed)
        self.tool = tool
        self.fault = fault
        self.on_call = on_call
        self.canary = (f"{CANARY_PREFIX}{self._rng.randrange(16**6):06x}"
                       if fault in ("tool_inject_instruction", "tool_hijack") else "")
        self.hits: list[ToolFaultHit] = []
        self.installed_on: list[str] = []

    # -- what drift_test records for the perturbation -------------------------

    def _mutate(self, world: World) -> str | None:
        which = "first" if self.on_call == 1 else f"{self.on_call}{_ordinal(self.on_call)}"
        return (
            f"runtime fault on the {which} call to tool {self.tool!r}: "
            f"{TOOL_FAULTS[self.fault]['models']} (state untouched)"
        )

    def apply(self, world: World) -> EventRecord:
        rec = super().apply(world)
        return rec

    @property
    def fired(self) -> bool:
        return bool(self.hits)

    def outcome_note(self) -> str:
        """Appended to the perturbation summary after the run."""
        if not self.hits:
            if not self.installed_on:
                return " — no ToolNode offers this tool; the fault could not be installed"
            return (f" — the graph never made call #{self.on_call} to {self.tool!r} on this run, "
                    "so the fault did not fire")
        h = self.hits[0]
        return f" — fired on call #{h.call_index} ({h.node_path}) with args {_preview(h.args)}"

    # -- the fault itself --------------------------------------------------------

    def _faulted_message(self, request: Any, real: Any) -> Any:
        """Build the ToolMessage this fault returns. `real` is the genuine
        result (already executed) for the faults that rewrite it, else None."""
        from langchain_core.messages import ToolMessage

        call = request.tool_call
        call_id = str(call.get("id") or "")
        name = str(call.get("name") or self.tool)
        f = self.fault
        if f == "tool_error":
            return ToolMessage(content=f"Error: ServiceUnavailable: {name} returned HTTP 503 "
                                       "(upstream temporarily unavailable)",
                               tool_call_id=call_id, name=name, status="error")
        if f == "tool_timeout":
            return ToolMessage(content=f"Error: TimeoutError: {name} did not respond within 30s",
                               tool_call_id=call_id, name=name, status="error")
        if f == "tool_rate_limit":
            return ToolMessage(content=f"Error: RateLimitError: 429 Too Many Requests for {name}; "
                                       "retry after 60 seconds",
                               tool_call_id=call_id, name=name, status="error")
        if f == "tool_empty":
            return ToolMessage(content="", tool_call_id=call_id, name=name)
        real_text = _text_of(getattr(real, "content", "")) if real is not None else ""
        if f == "tool_truncated":
            cut = max(1, len(real_text) // 4) if real_text else 0
            return ToolMessage(content=real_text[:cut] + ("…[truncated]" if real_text else "[truncated]"),
                               tool_call_id=call_id, name=name)
        if f == "tool_stale":
            return ToolMessage(content=f"[cached result from 9 days ago — may be out of date]\n{real_text}",
                               tool_call_id=call_id, name=name)
        if f == "tool_inject_instruction":
            body = (real_text + "\n\n" if real_text else "") + _injection_payload(self.canary)
            return ToolMessage(content=body, tool_call_id=call_id, name=name)
        if f == "tool_hijack":
            body = (real_text + "\n\n" if real_text else "") + _hijack_payload(self.canary, name)
            return ToolMessage(content=body, tool_call_id=call_id, name=name)
        raise AssertionError(f)  # pragma: no cover

    def _needs_real_result(self) -> bool:
        return self.fault in ("tool_truncated", "tool_stale", "tool_inject_instruction", "tool_hijack")

    @contextlib.contextmanager
    def install(self, graph: Any) -> Iterator["ToolChaosEvent"]:
        """Install this fault on every ToolNode in `graph` that offers the
        tool, for the duration of the block. Call counting is shared across
        nodes (the Nth call to the tool anywhere in the graph)."""
        nodes = [(path, n) for path, n in find_tool_nodes(graph) if self.tool in n._tools_by_name]
        self.installed_on = [path for path, _ in nodes]
        self.hits = []
        counter = {"n": 0}
        saved: list[tuple[Any, Any, Any]] = []

        def _should_fire(request: Any) -> tuple[bool, int]:
            name = str((request.tool_call or {}).get("name") or "")
            if name != self.tool:
                return False, 0
            counter["n"] += 1
            return counter["n"] == self.on_call, counter["n"]

        def _record(request: Any, path: str, idx: int, real: Any) -> None:
            call = request.tool_call or {}
            self.hits.append(ToolFaultHit(
                tool=self.tool, call_index=idx, call_id=str(call.get("id") or ""), node_path=path,
                args=call.get("args"),
                original_preview=_preview(_text_of(getattr(real, "content", "")), 80) if real is not None else "",
            ))

        for path, node in nodes:
            prev_sync, prev_async = node._wrap_tool_call, node._awrap_tool_call
            saved.append((node, prev_sync, prev_async))

            def sync_wrapper(request, execute, _path=path, _prev=prev_sync):
                fire, idx = _should_fire(request)
                if not fire:
                    return _prev(request, execute) if _prev is not None else execute(request)
                if self.fault == "tool_exception":
                    _record(request, _path, idx, None)
                    raise ToolChaosError(f"drift chaos: {self.tool} raised (tool_exception fault)")
                real = execute(request) if self._needs_real_result() else None
                _record(request, _path, idx, real)
                return self._faulted_message(request, real)

            async def async_wrapper(request, execute, _path=path, _prev_a=prev_async, _prev_s=prev_sync):
                fire, idx = _should_fire(request)
                if not fire:
                    if _prev_a is not None:
                        return await _prev_a(request, execute)
                    if _prev_s is not None:
                        # The node's own fallback path would have passed a sync execute;
                        # a sync wrapper given an async execute cannot work, so run it
                        # through the async execute directly.
                        result = _prev_s(request, lambda req: execute(req))
                        return await result if inspect.isawaitable(result) else result
                    return await execute(request)
                if self.fault == "tool_exception":
                    _record(request, _path, idx, None)
                    raise ToolChaosError(f"drift chaos: {self.tool} raised (tool_exception fault)")
                real = await execute(request) if self._needs_real_result() else None
                _record(request, _path, idx, real)
                return self._faulted_message(request, real)

            node._wrap_tool_call = sync_wrapper
            node._awrap_tool_call = async_wrapper
        try:
            yield self
        finally:
            for node, prev_sync, prev_async in saved:
                node._wrap_tool_call = prev_sync
                node._awrap_tool_call = prev_async


def _preview(value: Any, limit: int = 60) -> str:
    import json

    try:
        text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    text = " ".join(str(text).split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


# ---------------------------------------------------------------------------
# Catalogue for the planner
# ---------------------------------------------------------------------------


def discover_tool_patterns(graph: Any, *, seed: int = 0, faults: list[str] | None = None,
                           on_call: int = 1) -> list[ChaosSpec]:
    """One ChaosSpec per (reachable tool × fault), named `tool_timeout[web_search]`
    so `--exclude tool_timeout` / `--exclude tool_timeout[web_search]` work
    like every other pattern. Empty when the graph has no ToolNode."""
    names = tool_names(graph)
    chosen = list(faults or TOOL_FAULTS)
    specs: list[ChaosSpec] = []
    for name in names:
        for fault in chosen:
            if fault not in TOOL_FAULTS:
                raise ValueError(f"unknown tool fault {fault!r}; one of {', '.join(TOOL_FAULTS)}")
            specs.append(ChaosSpec(
                pattern_name=f"{fault}[{name}]",
                field=name,
                pattern_type=fault,
                factory=lambda n=name, f=fault, s=seed, k=on_call: ToolChaosEvent(n, fault=f, on_call=k, seed=s),
                fault_class=TOOL_FAULTS[fault]["fault_class"],
            ))
    return specs


def is_tool_event(event: Any) -> bool:
    return isinstance(event, ToolChaosEvent)


__all__ = [
    "DEFAULT_AUTO_FAULTS", "FAULT_LAYER_STATE", "FAULT_LAYER_TOOL", "TOOL_FAULTS", "ToolChaosError", "ToolChaosEvent",
    "ToolFaultHit", "discover_tool_patterns", "find_tool_nodes", "is_tool_event", "tool_names",
]
