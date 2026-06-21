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

What it actually does:
  1. Walks the user's initial_state dict and enumerates schema-driven
     perturbations (flip_bool[is_admin], remove_dict_key[open_cases], etc.)
     via drift.chaos. Pattern selection dispatches on runtime types so the
     user doesn't have to declare anything.
  2. Invokes the graph once with the unperturbed state (baseline).
  3. For each perturbation, copies the initial state, applies the chaos
     mutation, invokes the graph, captures (final_state, exception, time).
  4. Compares each perturbed final state to the baseline final state and
     reports crashed / diverged / unchanged.

It does NOT import langgraph. Anything with `.invoke(dict) -> dict` or
`.ainvoke(dict) -> dict` works. That lets us ship + test without pulling
langgraph as a hard dep and incidentally covers homemade graph runners too.

For chaos that fires *between* nodes (not just at initial state), the
user needs to switch their graph to a langgraph checkpointer and use a
stepwise driver — out of scope for the MVP adapter, where the perturbation
target is the initial-state schema.
"""
from __future__ import annotations

import asyncio
import inspect
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from drift.chaos.engine import _normalize_intensity, plan_auto_chaos
from drift.chaos.fuzzer import discover_field_patterns
from drift.world import World, WorldState

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


@dataclass
class BaselineResult:
    """The unperturbed run. Used as the comparison anchor."""

    initial_state: dict
    final_state: dict | None = None
    crashed: bool = False
    error: str = ""
    error_type: str = ""
    duration_s: float = 0.0


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

    def summary_lines(self) -> list[str]:
        """Human-readable one-line-per-bucket summary. Used by the example
        and any caller who wants a quick stdout report."""
        out = [
            f"drift × graph: {len(self.perturbations)} perturbation(s) "
            f"(intensity={self.intensity}, schema yielded {self.patterns_total} pattern(s))",
            f"  crashed   : {self.n_crashed}",
            f"  diverged  : {self.n_diverged}",
            f"  unchanged : {self.n_unchanged}",
        ]
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
    """Fail fast if `graph` doesn't have one of the entry points we support.

    Raised eagerly (outside the per-run try/except) so misuse surfaces as
    a TypeError to the caller rather than getting recorded as a baseline
    crash — a real user-code crash is interesting telemetry, a misconfigured
    graph reference is not.
    """
    if (
        callable(getattr(graph, "ainvoke", None))
        or callable(getattr(graph, "invoke", None))
        or callable(graph)
    ):
        return
    raise TypeError(
        f"graph object {type(graph).__name__!r} has no .invoke/.ainvoke and "
        "is not callable; pass a compiled langgraph StateGraph, an async "
        "function, or any object with .invoke(state) -> dict"
    )


async def _invoke_graph(graph: Any, state: dict) -> dict:
    """Call graph.ainvoke or graph.invoke, returning the resulting state.

    Tries ainvoke first (native for compiled langgraph), falls back to
    sync invoke, then to plain __call__. Assumes _validate_graph has
    already ruled out the no-entry-point case.
    """
    ainvoke = getattr(graph, "ainvoke", None)
    if callable(ainvoke):
        result = ainvoke(state)
        if inspect.isawaitable(result):
            result = await result
        return result if isinstance(result, dict) else dict(result)

    invoke = getattr(graph, "invoke", None)
    if callable(invoke):
        result = invoke(state)
        if inspect.isawaitable(result):
            result = await result
        return result if isinstance(result, dict) else dict(result)

    # Plain callable fallback — _validate_graph already confirmed it's callable.
    result = graph(state)
    if inspect.isawaitable(result):
        result = await result
    return result if isinstance(result, dict) else dict(result)


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
) -> tuple[dict | None, str, str, float]:
    """Invoke the graph once, capturing exceptions and timing.

    Returns (final_state_or_None, error_type, error_msg, duration_seconds).
    """
    t0 = time.perf_counter()
    try:
        final = await _invoke_graph(graph, initial_state)
        return (final, "", "", time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001 — user code, anything possible
        return (None, type(exc).__name__, str(exc), time.perf_counter() - t0)


async def drift_test_async(
    *,
    graph: Any,
    initial_state: dict,
    intensity: str | bool | None = "moderate",
    seed: int = 42,
    auto_chaos_exclude: Iterable[str] | None = None,
    max_perturbations: int = DEFAULT_MAX_PERTURBATIONS,
    state_factory: Callable[[], dict] | None = None,
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
    # schema doesn't silently rack up LLM cost.
    if len(scheduled) > max_perturbations:
        scheduled = scheduled[:max_perturbations]

    # Baseline run uses a fresh copy of the user's state so the graph can't
    # mutate the input dict and leak across perturbation runs.
    baseline_input = (
        deepcopy(state_factory()) if state_factory else deepcopy(initial_state)
    )
    bfinal, betype, berr, btime = await _run_one(graph, baseline_input)
    baseline = BaselineResult(
        initial_state=baseline_input,
        final_state=bfinal,
        crashed=bool(betype),
        error=berr,
        error_type=betype,
        duration_s=btime,
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

        final, etype, err, took = await _run_one(graph, post_state)
        diverged, divsum = _diff_states(baseline.final_state, final)

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
) -> AdapterResult:
    """Run drift's auto-chaos against a compiled graph and report results.

    The minimum viable adapter: takes a user's compiled LangGraph (or
    anything with .invoke / .ainvoke / __call__) plus an initial state
    dict, perturbs the initial state via drift's schema-driven chaos, and
    reports which perturbations crashed vs diverged from the baseline.

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

    Returns:
        AdapterResult with .baseline and .perturbations; see those
        classes for fields. .summary_lines() gives a quick stdout report.

    Notes:
        This calls asyncio.run() internally — don't call from inside an
        already-running event loop. Use drift_test_async in that case.
        Cost: 1 + len(scheduled_perturbations) graph invocations per call.
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
        )
    )


__all__ = [
    "AdapterResult",
    "BaselineResult",
    "PerturbationResult",
    "drift_test",
    "drift_test_async",
]
