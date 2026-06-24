"""LangGraph adapter — run drift's auto-chaos against a compiled graph.

The natural drift onboarding story for a LangGraph user is "test what happens
when your initial state has the kind of subtle wrongness you'd get in
production." This module makes that a one-liner:

    from drift.adapters.langgraph import drift_test

    result = drift_test(
        graph=my_compiled_graph,
        initial_state={"messages": [...], "decisions": []},
        intensity="moderate",
    )

    for p in result.perturbations:
        if p.crashed:
            print(f"CRASH under {p.event_name}: {p.error}")
        elif p.diverged:
            print(f"DIVERGED under {p.event_name}: {p.divergence_summary}")
        for f in p.judge_findings:
            print(f"  JUDGE [{f['failure_type']}]: {f['summary']}")

What it actually does:
  1. Walks the user's initial_state dict and enumerates schema-driven
     perturbations (flip_bool[is_admin], remove_dict_key[open_cases], etc.)
     via drift.chaos. Pattern selection dispatches on runtime types so the
     user doesn't have to declare anything.
  2. Streams the graph once with the unperturbed state (baseline), capturing
     per-super-step records (node name + state delta).
  3. For each perturbation, copies the initial state, applies the chaos
     mutation, streams the graph, captures (final_state, exception, trace).
  4. Compares each perturbed final state to the baseline; runs the LLM judge
     (if supplied) over the per-perturbation trace to surface coordination
     failures the deterministic crash/diverge buckets can't catch.

It does NOT import langgraph. Anything with `.invoke(dict) -> dict` (or
`.ainvoke` / `.stream` / `.astream`) works. Plain callables work too —
they just don't produce a trace and so can't be judged. That lets us ship +
test without pulling langgraph as a hard dep and covers homemade runners.

For chaos that fires *between* nodes (not just at initial state), the
user needs to switch their graph to a langgraph checkpointer and use a
stepwise driver — out of scope for the MVP adapter, where the perturbation
target is the initial-state schema.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from drift.agents.base import Action
from drift.chaos.engine import _normalize_intensity, plan_auto_chaos
from drift.chaos.fuzzer import discover_field_patterns
from drift.failures.base import DetectorContext
from drift.failures.judge import JudgeLLM, LLMJudgeDetector
from drift.world import World, WorldHistory, WorldState

# How many perturbations we'll run by default before clamping. Each
# perturbation = one full graph invocation, so users who set
# intensity="aggressive" against an LLM-heavy graph can rack up cost fast.
# This cap is configurable per-call via `max_perturbations`.
DEFAULT_MAX_PERTURBATIONS = 25

# Step horizon passed to plan_auto_chaos. The chaos planner uses this to
# spread events across a simulated timeline; for the adapter we don't care
# about timing (each perturbation is its own one-shot run) so we pick a
# horizon long enough to give the planner room to pick varied events.
_PLANNER_HORIZON = 30


@runtime_checkable
class _GraphLike(Protocol):
    """Minimum surface the adapter needs from a graph object.

    `.invoke()` is the sync entry point on a compiled langgraph StateGraph.
    `.ainvoke()` is the async one. Either is fine; the adapter detects
    which exists at call time. We use a Protocol (not isinstance) so any
    object — including a plain function or a test stub — works.
    """

    def invoke(self, state: dict, *args: Any, **kwargs: Any) -> dict: ...


@dataclass
class PerturbationResult:
    """Outcome of one chaos-perturbed graph invocation.

    Exactly one of `crashed` or `final_state` is meaningful per result:
      - crashed=True  -> error/error_type set, final_state is None
      - crashed=False -> final_state set, error fields are empty strings

    `trace` is the captured per-super-step sequence — populated only when the
    graph supports `.stream()` / `.astream()`. Plain callables run via
    `.invoke()` and produce no trace, so `judge_findings` stays empty for
    those even when a judge is supplied.
    """

    event_name: str
    event_summary: str
    perturbed_field: str
    pattern_type: str
    perturbed_initial_state: dict
    final_state: dict | None = None
    crashed: bool = False
    error: str = ""
    error_type: str = ""
    diverged: bool = False
    divergence_summary: str = ""
    duration_s: float = 0.0
    trace: list[dict] = field(default_factory=list)
    judge_findings: list[dict] = field(default_factory=list)


@dataclass
class BaselineResult:
    """The unperturbed run. Used as the comparison anchor."""

    initial_state: dict
    final_state: dict | None = None
    crashed: bool = False
    error: str = ""
    error_type: str = ""
    duration_s: float = 0.0
    trace: list[dict] = field(default_factory=list)
    judge_findings: list[dict] = field(default_factory=list)


@dataclass
class AdapterResult:
    """Top-level result of one drift_test call.

    Fields:
      baseline:       the unperturbed run.
      perturbations:  one entry per chaos perturbation attempted.
      intensity:      normalized intensity used ("off" | "light" | ...).
      patterns_total: how many unique chaos patterns the schema produced
                      (some may be filtered out before scheduling).
    """

    baseline: BaselineResult
    perturbations: list[PerturbationResult] = field(default_factory=list)
    intensity: str = "moderate"
    patterns_total: int = 0

    @property
    def n_crashed(self) -> int:
        return sum(1 for p in self.perturbations if p.crashed)

    @property
    def n_diverged(self) -> int:
        return sum(1 for p in self.perturbations if p.diverged and not p.crashed)

    @property
    def n_unchanged(self) -> int:
        return sum(
            1
            for p in self.perturbations
            if not p.crashed and not p.diverged
        )

    @property
    def n_judge_findings(self) -> int:
        """Total judge-flagged findings across baseline + all perturbations."""
        return len(self.baseline.judge_findings) + sum(
            len(p.judge_findings) for p in self.perturbations
        )

    def summary_lines(self) -> list[str]:
        """Human-readable one-line-per-bucket summary. Used by the example
        and any caller who wants a quick stdout report."""
        out = [
            f"drift × graph: {len(self.perturbations)} perturbation(s) "
            f"(intensity={self.intensity}, schema yielded {self.patterns_total} pattern(s))",
            f"  crashed         : {self.n_crashed}",
            f"  diverged        : {self.n_diverged}",
            f"  unchanged       : {self.n_unchanged}",
        ]
        if self.n_judge_findings:
            out.append(f"  judge findings  : {self.n_judge_findings}")
        if self.baseline.crashed:
            out.append(
                f"  ! baseline itself crashed: {self.baseline.error_type}: {self.baseline.error}"
            )
        return out


def _build_state_model(initial_state: dict) -> type[BaseModel]:
    """Wrap an arbitrary state dict in a Pydantic model so chaos can walk it.

    We don't try to infer per-field types — the chaos fuzzer dispatches on
    runtime values via discover_field_patterns. ConfigDict(extra="allow")
    means any field name is accepted; we set defaults to the user's
    initial values so the model can be re-instantiated cheaply.

    Returns a brand-new class so two adapter calls don't share state schema.
    """

    class AdapterState(BaseModel):
        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
        timestep: int = 0

    # Round-trip through model_validate so the user's dict becomes the
    # default fields (via extra) on every fresh instance.
    return AdapterState


def _materialize_world(state_cls: type[BaseModel], initial_state: dict) -> World:
    """Build a fresh World instance from the user's initial state dict.

    chaos events mutate via getattr/setattr on world.state, and they call
    world.record_change(...) for the audit trail. So we need a real World
    wrapper, not just the BaseModel.
    """
    payload = {k: deepcopy(v) for k, v in initial_state.items() if k != "timestep"}
    instance = state_cls.model_validate(payload)
    # The World ctor accepts WorldState; we pass our BaseModel subclass which
    # the runtime treats identically (it just calls getattr/setattr/model_dump).
    return World(initial=instance)  # type: ignore[arg-type]


def _state_to_dict(world: World) -> dict:
    """Extract the post-perturbation state as a plain dict for the graph."""
    out = world.state.model_dump()
    # Strip drift's own timestep field — the user's graph never asked for it.
    out.pop("timestep", None)
    return out


def _validate_graph(graph: Any) -> None:
    """Fail fast if `graph` doesn't expose any entry point we support.

    Recognized: `.astream` / `.stream` / `.ainvoke` / `.invoke` / `__call__`.
    Raised eagerly (outside the per-run try/except) so misuse surfaces as
    a TypeError to the caller rather than getting recorded as a baseline
    crash — a real user-code crash is interesting telemetry, a misconfigured
    graph reference is not.
    """
    for attr in ("astream", "stream", "ainvoke", "invoke"):
        if callable(getattr(graph, attr, None)):
            return
    if callable(graph):
        return
    raise TypeError(
        f"graph object {type(graph).__name__!r} has no .invoke/.ainvoke/.stream/.astream "
        "and is not callable; pass a compiled langgraph StateGraph, an async "
        "function, or any object with .invoke(state) -> dict"
    )


def _normalize_chunk(chunk: Any) -> dict[str, dict]:
    """Coerce a langgraph stream chunk into a {node_name: update} dict.

    LangGraph's `stream_mode="updates"` (default) yields chunks shaped like
    `{"node_name": {update_dict}}`. Some chunks carry framework-internal keys
    like "__start__" / "__end__" — we keep them so the trace has full fidelity,
    and downstream callers can filter if they care.
    """
    if isinstance(chunk, dict):
        return {str(k): (v if isinstance(v, dict) else {"_value": v}) for k, v in chunk.items()}
    return {"_chunk": {"_value": chunk}}


async def _stream_or_invoke(graph: Any, state: dict) -> tuple[dict | None, list[dict]]:
    """Run the graph once. Capture a per-super-step trace if possible.

    Returns (final_state, trace). The trace is a list of records
    `{step, node, update, state_after}` — empty if the graph supports
    invoke only (plain function / `.invoke()`-only stubs).

    Prefer `.astream()` -> `.stream()` -> `.ainvoke()` -> `.invoke()` ->
    `__call__`. Streaming uses default mode (`"updates"`) and accumulates
    each super-step's delta onto a running state dict. For most graphs
    this matches the canonical final state; graphs that rely on langgraph
    reducer channels (e.g. messages with `add_messages`) may see slight
    drift in the merged final_state vs what `.invoke()` would return. The
    trace is unaffected — it just records each node's emitted update.
    """
    running: dict = dict(state)
    trace: list[dict] = []
    step = 0

    astream = getattr(graph, "astream", None)
    if callable(astream):
        async for chunk in astream(state):
            for node, update in _normalize_chunk(chunk).items():
                if isinstance(update, dict):
                    running.update(update)
                step += 1
                trace.append({
                    "step": step,
                    "node": node,
                    "update": deepcopy(update),
                    "state_after": deepcopy(running),
                })
        return (running, trace)

    stream = getattr(graph, "stream", None)
    if callable(stream):
        for chunk in stream(state):
            for node, update in _normalize_chunk(chunk).items():
                if isinstance(update, dict):
                    running.update(update)
                step += 1
                trace.append({
                    "step": step,
                    "node": node,
                    "update": deepcopy(update),
                    "state_after": deepcopy(running),
                })
        return (running, trace)

    # Plain invoke fallback: no per-step trace possible.
    ainvoke = getattr(graph, "ainvoke", None)
    if callable(ainvoke):
        result = ainvoke(state)
        if inspect.isawaitable(result):
            result = await result
        return (result if isinstance(result, dict) else dict(result), [])

    invoke = getattr(graph, "invoke", None)
    if callable(invoke):
        result = invoke(state)
        if inspect.isawaitable(result):
            result = await result
        return (result if isinstance(result, dict) else dict(result), [])

    # _validate_graph guaranteed the graph is at minimum callable.
    result = graph(state)
    if inspect.isawaitable(result):
        result = await result
    return (result if isinstance(result, dict) else dict(result), [])


def _diff_states(baseline: dict | None, perturbed: dict | None) -> tuple[bool, str]:
    """Return (diverged, summary). Compares top-level keys only — that's
    where coordination-relevant output lives for almost every graph shape.

    For dict/list values we compare by serialized JSON to keep this honest
    about nested changes without trying to produce a deep diff (would bloat
    the result and most users will just glance at the summary).
    """
    if baseline is None or perturbed is None:
        # If either side crashed we don't claim divergence here; caller
        # uses the crashed flag separately.
        return (False, "")

    diffs: list[str] = []
    bkeys = set(baseline.keys())
    pkeys = set(perturbed.keys())

    for k in sorted(bkeys - pkeys):
        diffs.append(f"-{k}")
    for k in sorted(pkeys - bkeys):
        diffs.append(f"+{k}")

    import json

    for k in sorted(bkeys & pkeys):
        bv, pv = baseline[k], perturbed[k]
        if bv == pv:
            continue
        try:
            bs, ps = json.dumps(bv, default=str, sort_keys=True), json.dumps(
                pv, default=str, sort_keys=True
            )
        except Exception:
            bs, ps = repr(bv), repr(pv)
        # Truncate long values so the summary stays human-readable.
        bs = bs if len(bs) <= 60 else bs[:57] + "..."
        ps = ps if len(ps) <= 60 else ps[:57] + "..."
        diffs.append(f"{k}: {bs} -> {ps}")

    if not diffs:
        return (False, "")
    return (True, "; ".join(diffs))


async def _run_one(
    graph: Any, initial_state: dict
) -> tuple[dict | None, list[dict], str, str, float]:
    """Invoke the graph once, capturing exceptions, trace, and timing.

    Returns (final_state_or_None, trace, error_type, error_msg, duration).
    Trace is [] if the graph doesn't support streaming or if it crashed
    before any super-step yielded.
    """
    t0 = time.perf_counter()
    try:
        final, trace = await _stream_or_invoke(graph, initial_state)
        return (final, trace, "", "", time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001 — user code, anything possible
        return (None, [], type(exc).__name__, str(exc), time.perf_counter() - t0)


async def _run_judge_on_trace(
    judge_llm: JudgeLLM,
    user_guidelines: list[str] | None,
    trace: list[dict],
    state_cls: type[BaseModel],
) -> list[dict]:
    """Run the LLM judge over one perturbation's captured trace.

    Synthesizes a drift `DetectorContext` from the trace: each super-step
    becomes one `Action` (`kind="node:<name>"`, rationale = the node's
    state delta) plus one snapshot in a fresh `WorldHistory`. Then fires
    a one-shot `LLMJudgeDetector` with `every=1` so it runs immediately
    over the full window.

    Returns judge findings as a list of JSON-serializable dicts. Empty
    list if the trace had no super-steps (e.g. graph crashed at step 0,
    or the graph doesn't support streaming).
    """
    if not trace:
        return []

    actions: list[Action] = []
    history = WorldHistory(maxlen=max(len(trace), 16))
    for entry in trace:
        step = int(entry["step"])
        node = str(entry["node"])
        update_repr = json.dumps(entry.get("update") or {}, default=str)[:400]
        actions.append(Action(
            timestep=step,
            agent_name=node,
            kind=f"node:{node}",
            rationale=update_repr,
        ))
        # Snapshot the state AFTER this super-step. Wrap into the adapter's
        # Pydantic shell so the judge's window renderer can model_dump it.
        snap_payload = {
            k: v for k, v in (entry.get("state_after") or {}).items() if k != "timestep"
        }
        snap = state_cls.model_validate(snap_payload)
        snap.timestep = step  # type: ignore[attr-defined]
        history.record(snap, [])  # type: ignore[arg-type]

    detector = LLMJudgeDetector(
        judge=judge_llm,
        every=1,                      # fire on the synthetic "final" tick
        window=len(trace),            # show the judge every super-step
        user_guidelines=list(user_guidelines) if user_guidelines else None,
    )
    ctx = DetectorContext(
        timestep=len(trace),
        history=history,
        actions=actions,
        events=[],
        already_reported=set(),
    )
    findings = await detector(ctx)
    return [f.model_dump(mode="json") for f in findings]


async def drift_test_async(
    *,
    graph: Any,
    initial_state: dict,
    intensity: str | bool | None = "moderate",
    seed: int = 42,
    auto_chaos_exclude: Iterable[str] | None = None,
    max_perturbations: int = DEFAULT_MAX_PERTURBATIONS,
    state_factory: Callable[[], dict] | None = None,
    judge_llm: JudgeLLM | None = None,
    user_guidelines: Iterable[str] | None = None,
) -> AdapterResult:
    """Async variant of drift_test. Use from inside an existing event loop.

    See drift_test for argument docs. The async version exists because
    langgraph.ainvoke is the native call inside async webserver handlers
    (FastAPI, etc.) where wrapping in asyncio.run would crash.
    """
    if not isinstance(initial_state, dict):
        raise TypeError(
            f"initial_state must be a dict; got {type(initial_state).__name__}. "
            "If you're using a TypedDict, pass it as dict(my_state)."
        )
    _validate_graph(graph)

    state_cls = _build_state_model(initial_state)
    level = _normalize_intensity(intensity)
    guidelines = [g for g in (user_guidelines or []) if g and str(g).strip()]

    # Enumerate every applicable chaos pattern for telemetry, then schedule.
    # We schedule across _PLANNER_HORIZON synthetic steps; the timesteps are
    # only used to order events, not actually simulated here.
    seed_state = state_cls.model_validate(
        {k: v for k, v in initial_state.items() if k != "timestep"}
    )
    all_specs = discover_field_patterns(
        seed_state,  # type: ignore[arg-type]
        exclude_fields=None,
        seed=seed,
    )
    scheduled = plan_auto_chaos(
        state=seed_state,  # type: ignore[arg-type]
        steps=_PLANNER_HORIZON,
        intensity=level,
        seed=seed,
        exclude=auto_chaos_exclude,
    )

    # Clamp to the per-call ceiling so an aggressive intensity on a noisy
    # schema doesn't silently rack up LLM cost (plus optional judge cost).
    if len(scheduled) > max_perturbations:
        scheduled = scheduled[:max_perturbations]

    # Baseline run uses a fresh copy of the user's state so the graph can't
    # mutate the input dict and leak across perturbation runs.
    baseline_input = (
        deepcopy(state_factory()) if state_factory else deepcopy(initial_state)
    )
    bfinal, btrace, betype, berr, btime = await _run_one(graph, baseline_input)
    baseline_findings: list[dict] = []
    if judge_llm is not None and btrace:
        baseline_findings = await _run_judge_on_trace(
            judge_llm, guidelines, btrace, state_cls,
        )
    baseline = BaselineResult(
        initial_state=baseline_input,
        final_state=bfinal,
        crashed=bool(betype),
        error=berr,
        error_type=betype,
        duration_s=btime,
        trace=btrace,
        judge_findings=baseline_findings,
    )

    perturbations: list[PerturbationResult] = []
    for _t, event in scheduled:
        # Each perturbation gets its own World built from a fresh copy of
        # the user's state, then has one chaos event applied.
        pert_input = (
            deepcopy(state_factory()) if state_factory else deepcopy(initial_state)
        )
        world = _materialize_world(state_cls, pert_input)
        record = event.apply(world)
        post_state = _state_to_dict(world)

        final, ptrace, etype, err, took = await _run_one(graph, post_state)
        diverged, divsum = _diff_states(baseline.final_state, final)

        # Judge runs even on crashes — partial traces (steps before the crash)
        # are often the most diagnostic. Skipped only when there's literally
        # no trace data to feed it (graph doesn't stream, or crashed at step 0).
        pert_findings: list[dict] = []
        if judge_llm is not None and ptrace:
            pert_findings = await _run_judge_on_trace(
                judge_llm, guidelines, ptrace, state_cls,
            )

        perturbations.append(
            PerturbationResult(
                event_name=event.name,
                event_summary=record.summary,
                perturbed_field=getattr(event, "field", ""),
                pattern_type=getattr(event, "pattern", "auto_chaos"),
                perturbed_initial_state=post_state,
                final_state=final,
                crashed=bool(etype),
                error=err,
                error_type=etype,
                diverged=diverged,
                divergence_summary=divsum,
                duration_s=took,
                trace=ptrace,
                judge_findings=pert_findings,
            )
        )

    return AdapterResult(
        baseline=baseline,
        perturbations=perturbations,
        intensity=level,
        patterns_total=len(all_specs),
    )


def drift_test(
    *,
    graph: Any,
    initial_state: dict,
    intensity: str | bool | None = "moderate",
    seed: int = 42,
    auto_chaos_exclude: Iterable[str] | None = None,
    max_perturbations: int = DEFAULT_MAX_PERTURBATIONS,
    state_factory: Callable[[], dict] | None = None,
    judge_llm: JudgeLLM | None = None,
    user_guidelines: Iterable[str] | None = None,
) -> AdapterResult:
    """Run drift's auto-chaos against a compiled graph and report results.

    Takes a user's compiled LangGraph (or anything with .invoke / .ainvoke /
    .stream / .astream / __call__) plus an initial state dict, perturbs the
    initial state via drift's schema-driven chaos, and reports which
    perturbations crashed, silently diverged, or were absorbed. With a judge
    supplied, also runs drift's 6-family LLM judge over each perturbation's
    per-super-step trace.

    Args:
        graph: a compiled LangGraph StateGraph, OR any object with
            `.invoke(state) -> dict`, `.ainvoke(state) -> dict`, or that
            is plainly callable. We don't import langgraph.
        initial_state: the dict you'd normally pass to graph.invoke(). Must
            be a dict — TypedDict instances are dicts at runtime, so they
            work. The dict's runtime field types determine which chaos
            patterns are applicable (bool -> flip, dict -> clear/remove/
            inject, list -> clear/duplicate/reverse, str -> corrupt,
            numeric -> boundary).
        intensity: "off" | "light" (~8%) | "moderate" (~18%, default) |
            "aggressive" (~35%). Same scale as drift.run's auto_chaos.
            True is an alias for "moderate".
        seed: RNG seed for reproducible perturbation selection.
        auto_chaos_exclude: substrings to skip when scheduling chaos.
            E.g. ["flip_bool"] disables all bool flips; ["messages"]
            disables every pattern targeting the `messages` field.
        max_perturbations: hard cap on perturbation runs per call.
            Each perturbation is a full graph invocation; if your graph
            calls an LLM, this directly bounds cost. Default 25.
        state_factory: optional callable returning a fresh initial_state
            dict per invocation. Use when initial_state contains non-
            picklable / non-deep-copyable objects (e.g. opened connections).
            If supplied, takes precedence over deepcopy(initial_state).
        judge_llm: optional LLM judge (built via
            `drift.failures.judge.build_judge(...)`). When supplied, drift
            runs the 6-family coordination-failure judge over each
            perturbation's captured per-super-step trace and attaches
            findings to `PerturbationResult.judge_findings`. Requires the
            graph to support `.stream()` / `.astream()`; plain `.invoke()`-only
            callables produce no trace and the judge silently skips them.
        user_guidelines: optional plain-English patterns appended to the
            judge's prompt. Matches show up under
            `failure_type = "llm:user_guideline:<n>"`. Use to express
            coordination rules specific to your domain.

    Returns:
        AdapterResult with .baseline and .perturbations; see those
        classes for fields. .summary_lines() gives a quick stdout report.

    Notes:
        This calls asyncio.run() internally — don't call from inside an
        already-running event loop. Use drift_test_async in that case.
        Cost: 1 + len(scheduled_perturbations) graph invocations per call,
        plus one judge call per perturbation if judge_llm is supplied.
    """
    return asyncio.run(
        drift_test_async(
            graph=graph,
            initial_state=initial_state,
            intensity=intensity,
            seed=seed,
            auto_chaos_exclude=auto_chaos_exclude,
            max_perturbations=max_perturbations,
            state_factory=state_factory,
            judge_llm=judge_llm,
            user_guidelines=user_guidelines,
        )
    )


__all__ = [
    "AdapterResult",
    "BaselineResult",
    "PerturbationResult",
    "drift_test",
    "drift_test_async",
]
