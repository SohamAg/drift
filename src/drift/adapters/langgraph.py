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

When the graph IS a real compiled LangGraph (its `.stream` accepts
`stream_mode`), the adapter streams with `stream_mode=["updates", "values"]`
and `subgraphs=True`: `state_after` is the graph's true reducer-applied
state, subgraph super-steps (including sub-agents started inside a tool
call) appear as `parent/inner` steps, langchain messages are normalised to
small stable dicts, and a crash's partial trace is kept. See
`_stream_or_invoke` / `_NativeTraceBuilder`.

For chaos that fires *between* nodes (not just at initial state), the
user needs to switch their graph to a langgraph checkpointer and use a
stepwise driver — out of scope for the MVP adapter, where the perturbation
target is the initial-state schema.
"""
from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
import inspect
import json
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from collections import Counter
from difflib import SequenceMatcher

from drift.agents.base import Action
from drift.chaos.engine import _normalize_intensity, plan_auto_chaos
from drift.chaos.fuzzer import discover_field_patterns
from drift.chaos.tools import (
    DEFAULT_AUTO_FAULTS,
    FAULT_LAYER_STATE,
    FAULT_LAYER_TOOL,
    discover_tool_patterns,
    is_tool_event,
)
from drift.failures.base import DetectorContext
from drift.failures.judge import JudgeLLM, LLMJudgeDetector
from drift.failures.library import run_all_on_trace as _run_library_on_trace
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

# Phase 2 divergence-cascade defaults.
#
# divergence_mode controls how baseline vs perturbed final state is compared:
#   "exact"      — equality on the whole dict (today's behavior; fast, free,
#                  but noisy on any LLM-driven output)
#   "tiered"     — cascade through tiers 0 (structural) → 1 (canonical equal)
#                  → 2 (cheap similarity gated by baseline noise) → 3 (judge,
#                  budget-capped). Only fields surviving the cheap tiers ever
#                  reach the judge, keeping LLM cost O(handful) per run.
#   "off"        — skip divergence detection entirely; report crash only.
DEFAULT_DIVERGENCE_MODE = "exact"

# When a state/messages chaos pattern lands.
#   "initial" — perturb the initial state and rerun the graph from the top
#               (every pattern since v0.1).
#   "mid_run" — run the baseline on a checkpointer, fork it at the step the
#               planner scheduled the event for, apply the pattern to that
#               checkpoint's state through `update_state`, resume. The graph
#               meets the fault in the middle of its own run, with its own
#               context in place. Needs a compiled LangGraph; drift attaches
#               a process-wide MemorySaver when the graph has no checkpointer.
#               Anything that cannot be forked (stub graph, a baseline with
#               fewer than two root steps, an update_state failure) falls
#               back to "initial" for that perturbation and says so in
#               `timing_note`. Tool-layer faults are mid-run by nature and
#               are unaffected.
CHAOS_TIMING_INITIAL = "initial"
CHAOS_TIMING_MID_RUN = "mid_run"
CHAOS_TIMINGS = (CHAOS_TIMING_INITIAL, CHAOS_TIMING_MID_RUN)
DEFAULT_BASELINE_ROLLOUTS = 1            # >1 enables noise-floor measurement
# How many times each perturbed state is run. 1 = one shot (the historical
# behaviour). For an LLM-driven graph one perturbed run is a sample, not a
# verdict: on a live 2-agent supervisor the planted prompt injection was
# followed in 1 of 4 runs. N > 1 reports k/N per perturbation and marks the
# perturbation intermittent when its rollouts disagree. Cost scales linearly.
DEFAULT_PERTURBATION_ROLLOUTS = 1
_OUTCOME_RANK = {"crashed": 0, "diverged": 1, "unchanged": 2}
DEFAULT_MAX_JUDGE_CALLS = 10             # tier-3 hard ceiling across one drift_test run
DEFAULT_SIMILARITY_THRESHOLD = 0.85      # tier-2 lexical cutoff (0..1, SequenceMatcher ratio)


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
class FieldNoiseBand:
    """Per-field natural variance measured across N baseline rollouts.

    The point of the noise band: LLM-driven graphs produce slightly different
    output for the same input. If a field naturally takes one of {A, B, C}
    across baseline rollouts, then seeing C in a perturbed run isn't
    meaningful divergence — it's within noise. Comparators consult this band
    to filter signal from natural model wobble.

    For enumerated / decision-shaped fields (small distinct_values set),
    `is_within_noise` for a perturbed value is simply membership.
    For text fields, we record pairwise lexical similarity stats so tier-2
    can ask: "is this perturbed-vs-baseline similarity lower than the typical
    baseline-vs-baseline similarity I saw?"
    For numerics, we keep min/max so tier-2 can range-check.
    """

    name: str
    sample_count: int
    distinct_values: list[Any] = field(default_factory=list)
    value_frequencies: dict[str, int] = field(default_factory=dict)   # str-keyed for JSON
    # Text-field stats — only populated when all samples are strings.
    text_min_similarity: float | None = None  # lowest pairwise sim across baselines
    text_mean_similarity: float | None = None
    # Numeric stats — only when all samples are int/float (and not bool).
    numeric_min: float | None = None
    numeric_max: float | None = None


@dataclass
class FieldDivergence:
    """One field where the perturbed final-state differs from baseline.

    `tier` records which level of the cost cascade actually fired:
      0 — structural (key added/removed/type changed): always meaningful
      1 — exact: values differ after canonical equality, no noise consulted
      2 — similarity: differ enough to exceed the baseline noise band
      3 — judge: LLM judge confirmed semantically different
    """

    name: str
    tier: int
    baseline_value: Any
    perturbed_value: Any
    summary: str
    similarity_score: float | None = None     # tier 2 (and tier 3 if computed)
    within_noise_band: bool | None = None     # tier 2/3: were we inside variance?
    judge_equivalent: bool | None = None      # tier 3 only
    judge_reasoning: str = ""                 # tier 3 only


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
    # "crash" | "omission" | "value" — the pattern's dependability fault class
    # (see drift.chaos.patterns). Lets results group "the input went missing"
    # apart from "the input was wrong" apart from "a dependency failed".
    fault_class: str = ""
    # "state" — the initial state was perturbed and the graph rerun;
    # "tool" — a runtime fault on one tool call (drift.chaos.tools), state
    # untouched. `fault_fired` is None for state patterns; for tool patterns
    # it says whether the graph actually made the faulted call on this run
    # (False = the fault never landed, so the outcome says nothing).
    fault_layer: str = "state"
    fault_fired: bool | None = None
    # "initial" | "mid_run" — where this perturbation landed (see
    # CHAOS_TIMING_*). `fork_step` is the baseline root super-step the run
    # was forked after when mid-run; `timing_note` explains a fallback.
    chaos_timing: str = "initial"
    fork_step: int | None = None
    timing_note: str = ""
    final_state: dict | None = None
    crashed: bool = False
    error: str = ""
    error_type: str = ""
    # "" when not crashed; else "recursion_limit" | "exception" — see _classify_crash.
    crash_class: str = ""
    diverged: bool = False
    divergence_summary: str = ""
    duration_s: float = 0.0
    trace: list[dict] = field(default_factory=list)
    judge_findings: list[dict] = field(default_factory=list)
    # Phase 2: per-field divergences after the tiered cost cascade.
    # Empty when divergence_mode="off" or when the perturbation crashed before
    # producing a final_state. With divergence_mode="exact" (default), each
    # entry is a tier-0/1 diff. With divergence_mode="tiered", entries also
    # include tier-2 (similarity) and tier-3 (judge) outcomes.
    divergence_details: list[FieldDivergence] = field(default_factory=list)
    # UNCHANGED-audit: per-field divergence candidates the cascade FILTERED
    # OUT — tier-2 within-noise-band matches and tier-3 judge-equivalent
    # matches. Preserved so the user can audit "was this really noise or
    # did drift quietly drop a real change?" Each entry carries the
    # similarity score / judge reasoning that justified the filter.
    filtered_divergences: list[FieldDivergence] = field(default_factory=list)
    # Phase 3: deterministic coordination-failure detector findings.
    # Populated whenever the perturbation produced a trace; empty otherwise.
    # Each entry is a FailureRecord.model_dump(mode="json") dict.
    coordination_findings: list[dict] = field(default_factory=list)
    # LangGraph thread the run executed on when the graph has a checkpointer
    # (each run gets its own, so runs never share state). "" otherwise.
    thread_id: str = ""
    # Repeated rollouts of the SAME perturbed state (perturbation_rollouts
    # > 1). This object is the representative rollout — the most severe one
    # (crashed > diverged > unchanged; ties → lowest index) — so the
    # top-level `crashed` / `diverged` read "in at least one rollout".
    # `rollouts` holds the other N-1 rollouts in full; empty when N == 1.
    rollout_index: int = 0
    rollouts: list["PerturbationResult"] = field(default_factory=list)

    # -- rollout aggregates -------------------------------------------------
    @property
    def n_rollouts(self) -> int:
        return 1 + len(self.rollouts)

    def all_rollouts(self) -> list["PerturbationResult"]:
        """Every rollout of this perturbation, representative included, in
        execution order."""
        return sorted([self, *self.rollouts], key=lambda r: r.rollout_index)

    @property
    def outcome(self) -> str:
        """This rollout's bucket: "crashed" | "diverged" | "unchanged"."""
        return "crashed" if self.crashed else "diverged" if self.diverged else "unchanged"

    @property
    def outcome_counts(self) -> dict[str, int]:
        """How many rollouts landed in each bucket, e.g. {"crashed": 1,
        "diverged": 2, "unchanged": 2}. Keys are always present."""
        counts = {"crashed": 0, "diverged": 0, "unchanged": 0}
        for r in self.all_rollouts():
            counts[r.outcome] += 1
        return counts

    @property
    def is_intermittent(self) -> bool:
        """True when the rollouts of this perturbation did not all land in
        the same bucket — the graph's response to this input is a coin
        flip, which is itself a finding (and the reason one run proves
        nothing either way)."""
        return self.n_rollouts > 1 and sum(1 for v in self.outcome_counts.values() if v) > 1

    @property
    def finding_rollout_counts(self) -> dict[str, int]:
        """failure_type -> number of ROLLOUTS in which that coordination
        detector fired (not the number of findings). `tool_call_loop: 2`
        with n_rollouts 5 reads "loops 2 times out of 5"."""
        out: dict[str, int] = {}
        for r in self.all_rollouts():
            seen = {f.get("failure_type", "") for f in r.coordination_findings}
            for ft in seen:
                if ft:
                    out[ft] = out.get(ft, 0) + 1
        return dict(sorted(out.items()))

    def outcome_label(self) -> str:
        """One-line k/N summary: "crashed 1/5 · diverged 2/5 · unchanged 2/5".
        For a single rollout just the bucket name."""
        if self.n_rollouts == 1:
            return self.outcome
        n = self.n_rollouts
        return " · ".join(f"{k} {v}/{n}" for k, v in self.outcome_counts.items() if v)


@dataclass
class BaselineResult:
    """The unperturbed run. Used as the comparison anchor."""

    initial_state: dict
    final_state: dict | None = None
    crashed: bool = False
    error: str = ""
    error_type: str = ""
    # "" when not crashed; else "recursion_limit" | "exception" — see _classify_crash.
    crash_class: str = ""
    duration_s: float = 0.0
    trace: list[dict] = field(default_factory=list)
    judge_findings: list[dict] = field(default_factory=list)
    coordination_findings: list[dict] = field(default_factory=list)
    # LangGraph thread the baseline ran on when the graph has a checkpointer;
    # drift_test_fork uses it to fork from a real checkpoint. "" otherwise.
    thread_id: str = ""


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
    # Phase 2: divergence cascade telemetry.
    divergence_mode: str = "exact"   # "exact" | "tiered" | "off"
    baseline_rollouts: int = 1       # how many baseline samples informed the noise band
    perturbation_rollouts: int = 1   # how many times each perturbed state was run
    noise_band: dict[str, FieldNoiseBand] = field(default_factory=dict)
    judge_calls_used: int = 0        # tier-3 calls actually fired (cost telemetry)
    judge_calls_budget: int = DEFAULT_MAX_JUDGE_CALLS
    chaos_timing: str = CHAOS_TIMING_INITIAL   # what the caller asked for

    @property
    def n_crashed(self) -> int:
        return sum(1 for p in self.perturbations if p.crashed)

    @property
    def n_diverged(self) -> int:
        return sum(1 for p in self.perturbations if p.diverged and not p.crashed)

    @property
    def n_recursion_limit(self) -> int:
        """Crashes that were LangGraph's recursion limit firing — i.e. the graph
        looped until the framework stopped it. A subset of `n_crashed`."""
        return sum(
            1 for p in self.perturbations
            if p.crashed and p.crash_class == CRASH_CLASS_RECURSION_LIMIT
        )

    @property
    def n_unchanged(self) -> int:
        return sum(
            1
            for p in self.perturbations
            if not p.crashed and not p.diverged
        )

    @property
    def n_intermittent(self) -> int:
        """Perturbations whose repeated rollouts disagreed on the outcome
        bucket. Always 0 when perturbation_rollouts == 1 — one sample can't
        disagree with itself, which is the point of running more."""
        return sum(1 for p in self.perturbations if p.is_intermittent)

    @property
    def n_graph_runs(self) -> int:
        """Total graph invocations this call made (baseline incl. noise
        rollouts + every perturbation rollout). Cost telemetry."""
        base = self.baseline_rollouts if self.baseline_rollouts > 1 else 0
        return base + 1 + sum(p.n_rollouts for p in self.perturbations)

    @property
    def n_judge_findings(self) -> int:
        """Total judge-flagged findings across baseline + all perturbations."""
        return len(self.baseline.judge_findings) + sum(
            len(p.judge_findings) for p in self.perturbations
        )

    @property
    def n_coordination_findings(self) -> int:
        """Total library-detector findings across baseline + all perturbations."""
        return len(self.baseline.coordination_findings) + sum(
            len(p.coordination_findings) for p in self.perturbations
        )

    @property
    def n_filtered_divergences(self) -> int:
        """Total tier-2/3 candidates the cascade dropped across all perturbations.

        These are diffs that *occurred* but the noise band or tier-3 judge
        cleared. A non-zero count under UNCHANGED-bucket perturbations is a
        cue to audit — open the row to see what was filtered and why.
        """
        return sum(len(p.filtered_divergences) for p in self.perturbations)

    def summary_lines(self) -> list[str]:
        """Human-readable one-line-per-bucket summary. Used by the example
        and any caller who wants a quick stdout report."""
        out = [
            f"drift × graph: {len(self.perturbations)} perturbation(s) "
            f"(intensity={self.intensity}, schema yielded {self.patterns_total} pattern(s))",
            f"  crashed         : {self.n_crashed}",
        ]
        if self.n_recursion_limit:
            out.append(f"    of which looped to recursion limit: {self.n_recursion_limit}")
        out += [
            f"  diverged        : {self.n_diverged}",
            f"  unchanged       : {self.n_unchanged}",
        ]
        if self.perturbation_rollouts > 1:
            out.append(
                f"  rollouts        : {self.perturbation_rollouts} per perturbation "
                f"({self.n_graph_runs} graph runs total)"
            )
            out.append(f"  intermittent    : {self.n_intermittent} (rollouts disagreed)")
        if self.n_judge_findings:
            out.append(f"  judge findings  : {self.n_judge_findings}")
        if self.n_coordination_findings:
            out.append(f"  coordination    : {self.n_coordination_findings}")
        if self.n_filtered_divergences:
            out.append(f"  filtered (audit): {self.n_filtered_divergences}")
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


# ---- message normalization -------------------------------------------------
#
# LangGraph graphs built on `add_messages` carry langchain `BaseMessage`
# objects in state. Those objects are hostile to everything drift does
# downstream: they aren't JSON, their `==` includes a per-run random `id`,
# and `response_metadata` / `usage_metadata` differ on every LLM call. Left
# raw, the `messages` field diverges on EVERY perturbation, which is pure
# noise. We normalize each message to a small, stable dict at capture time:
#
#     {"type": "ai", "content": "...", "name": "...",
#      "tool_calls": [{"name": ..., "args": ...}], "status": "error"}
#
# Per-run message `id`s are dropped. Tool-call ids are KEPT but renumbered in
# order of first appearance (`call_1`, `call_2`, ...) with the matching
# ToolMessage.tool_call_id remapped — stable across runs, and the normalised
# state still round-trips into `add_messages` (which rejects an AI tool_call
# without an id). Fork-edit-replay depends on that round trip. The provider-id
# -> call_k map is shared across one whole run (`_ToolCallIds`), so a node's
# `update` and the later full `state_after` agree on every id — otherwise a
# tools node's update would still carry the provider id while the agent's
# update said `call_1`, and a detector pairing the two would miss.


def _is_message_object(value: Any) -> bool:
    if isinstance(value, (dict, list, tuple, set, str, bytes, int, float, bool)) or value is None:
        return False
    return (
        hasattr(value, "type")
        and hasattr(value, "content")
        and callable(getattr(value, "model_dump", None))
    )


class _ToolCallIds:
    """Provider tool-call id -> stable `call_<k>` id, in order of first sight.
    One instance per graph run."""

    def __init__(self) -> None:
        self.map: dict[Any, str] = {}
        self._anon = 0

    def assign(self, provider_id: Any) -> str:
        if provider_id is None:
            self._anon += 1
            provider_id = ("__anon__", self._anon)
        new = self.map.get(provider_id)
        if new is None:
            new = f"call_{len(self.map) + 1}"
            self.map[provider_id] = new
        return new

    def lookup(self, provider_id: Any) -> Any:
        return self.map.get(provider_id, provider_id)


def _message_to_dict(msg: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": str(getattr(msg, "type", "") or ""),
        "content": _normalize_messages(getattr(msg, "content", "")),
    }
    name = getattr(msg, "name", None)
    if name:
        out["name"] = name
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = [
            {"name": tc.get("name"), "args": tc.get("args"), "id": tc.get("id")}
            for tc in tool_calls
            if isinstance(tc, dict)
        ]
    if out["type"] == "tool":
        out["tool_call_id"] = getattr(msg, "tool_call_id", None)
    status = getattr(msg, "status", None)
    if status and status != "success":
        out["status"] = status
    return out


def _renumber_tool_call_ids(msgs: list[Any], ids: _ToolCallIds) -> None:
    """In-place: replace provider-issued tool-call ids (random per run) with
    stable `call_<k>` ids via the run-wide map, remapping each
    ToolMessage.tool_call_id to match. Only touches dicts produced by
    `_message_to_dict`."""
    for m in msgs:
        if not isinstance(m, dict):
            continue
        for tc in m.get("tool_calls") or []:
            if isinstance(tc, dict):
                # Ids drift itself injected (chaos patterns use `drift_call_*`)
                # are already stable and mark the call as drift's, not the
                # graph's — detectors rely on that; leave them alone.
                if isinstance(tc.get("id"), str) and tc["id"].startswith("drift_call_"):
                    continue
                tc["id"] = ids.assign(tc.get("id"))
    for m in msgs:
        if isinstance(m, dict) and m.get("type") == "tool" and "tool_call_id" in m:
            m["tool_call_id"] = ids.lookup(m["tool_call_id"])


def _normalize_messages(value: Any, _depth: int = 0, ids: _ToolCallIds | None = None) -> Any:
    """Return a JSON-friendly deep copy of `value` with BaseMessage objects
    replaced by trimmed dicts. Non-message leaves are deep-copied. `ids` is
    the run-wide tool-call id map; a fresh one is used when not given."""
    if _depth > 24:
        return deepcopy(value)
    if ids is None:
        ids = _ToolCallIds()
    if _is_message_object(value):
        return _message_to_dict(value)
    if isinstance(value, dict):
        return {k: _normalize_messages(v, _depth + 1, ids) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        had_message = any(_is_message_object(v) for v in value)
        out = [_normalize_messages(v, _depth + 1, ids) for v in value]
        if had_message:
            _renumber_tool_call_ids(out, ids)
        return out
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return deepcopy(value)


def _accepts_stream_mode(fn: Any) -> bool:
    """True when `fn` (a graph's .stream/.astream) takes a `stream_mode` kwarg
    — i.e. it is a real LangGraph `Pregel`, not a hand-rolled stub. We check
    for the explicit parameter name rather than `**kwargs` so test doubles
    that merely accept anything don't get sent langgraph-specific kwargs."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if "stream_mode" in sig.parameters:
        return True
    # `graph.with_config(...)` wraps the Pregel in a langchain RunnableBinding
    # whose .stream is `(input, config=None, **kwargs)` — stream_mode is
    # forwarded through **kwargs. Accept that only when the owner really is a
    # langchain Runnable (has with_config + get_graph), so a hand-rolled stub
    # that merely takes **kwargs stays on the legacy path.
    owner = getattr(fn, "__self__", None)
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    return bool(
        has_var_kw
        and owner is not None
        and callable(getattr(owner, "with_config", None))
        and callable(getattr(owner, "get_graph", None))
    )


def _namespace_labels(ns: Any) -> list[str]:
    """`("tools:3f2a...", "inner:9b1c...")` -> `["tools", "inner"]`.

    LangGraph namespaces are `node:task_id` strings; the task id is a random
    uuid per run, so we keep only the node part for stable labels."""
    out: list[str] = []
    for part in (ns or ()):
        s = str(part)
        out.append(s.split(":", 1)[0] if ":" in s else s)
    return out


# Tool-call argument keys that name the sub-agent being dispatched (deepagents
# `task(subagent_type=...)`, hand-rolled `delegate(agent=...)`, …) and keys
# that carry the task text the sub-agent is started with.
_SUBAGENT_NAME_KEYS = (
    "subagent_type", "subagent", "agent", "agent_name", "agent_type",
    "worker", "specialist", "assistant", "to", "recipient", "target",
)
_SUBAGENT_TASK_KEYS = (
    "description", "task", "input", "prompt", "query", "instructions",
    "message", "request", "question", "goal",
)
# Tool names that are generic dispatchers — the tool name itself says
# nothing about WHICH sub-agent ran.
_GENERIC_DISPATCH_TOOLS = frozenset({
    "task", "delegate", "delegate_task", "call_agent", "run_agent",
    "invoke_agent", "subagent", "sub_agent", "spawn", "dispatch", "handoff",
    "transfer", "ask_agent", "call_subagent", "run_subagent",
})


def _pending_tool_calls(root_state: Any) -> list[dict]:
    """AI tool calls in the root conversation that have no tool result yet,
    oldest first. Works on drift's normalised message dicts."""
    msgs = root_state.get("messages") if isinstance(root_state, dict) else None
    if not isinstance(msgs, list):
        return []
    calls: dict[str, dict] = {}
    order: list[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("type") in ("ai", "assistant"):
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict):
                    cid = str(tc.get("id") or len(order))
                    calls[cid] = tc
                    order.append(cid)
        elif m.get("type") == "tool":
            calls.pop(str(m.get("tool_call_id") or ""), None)
    return [calls[c] for c in order if c in calls]


def _first_human_text(sub_state: Any) -> str:
    msgs = sub_state.get("messages") if isinstance(sub_state, dict) else None
    if not isinstance(msgs, list):
        return ""
    for m in msgs:
        if isinstance(m, dict) and m.get("type") in ("human", "user"):
            c = m.get("content")
            if isinstance(c, str):
                return c.strip()
            if isinstance(c, list):
                return " ".join(
                    str(b.get("text", "")) for b in c if isinstance(b, dict)
                ).strip()
    return ""


def _name_from_call(tc: dict) -> str:
    args = tc.get("args")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    if isinstance(args, dict):
        for k in _SUBAGENT_NAME_KEYS:
            v = args.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    name = str(tc.get("name") or "").strip()
    if name and name.lower() not in _GENERIC_DISPATCH_TOOLS:
        return name
    return ""


def _task_from_call(tc: dict) -> str:
    args = tc.get("args")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return ""
    if not isinstance(args, dict):
        return ""
    for k in _SUBAGENT_TASK_KEYS:
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _infer_subagent(root_state: Any, sub_state: Any) -> str:
    """Name the sub-agent a tool-dispatched subgraph belongs to, or "".

    A subgraph started from inside a tool call (deepagents' `task` tool, a
    hand-rolled `delegate` tool) streams as `tools/model`, `tools/tools` —
    the PARENT node's label. The name is recoverable from the root
    conversation: the tool call that is still unanswered when the subgraph
    runs names the sub-agent in its arguments (`subagent_type="critic"`) or
    in the tool's own name (`research_agent(...)`).

    Several sub-agents can be in flight at once (deepagents fans out two
    `task` calls in one AI message and both run under the same `tools`
    namespace), so the pending call is matched to THIS subgraph by its task
    text: the first human message of the subgraph's state is the
    `description` / `task` / `input` the parent passed. Only when there is
    exactly one pending call with a name does the match fall back to it.
    Returns "" rather than guess when neither rule decides.
    """
    pending = _pending_tool_calls(root_state)
    if not pending:
        return ""
    want = _first_human_text(sub_state)
    if want:
        matches = [tc for tc in pending if _task_from_call(tc) and _task_from_call(tc) == want]
        if not matches:
            # Tolerate wrappers: "Task: <text>" or the text embedded in a template.
            matches = [tc for tc in pending if _task_from_call(tc) and _task_from_call(tc) in want]
        if len(matches) == 1:
            return _name_from_call(matches[0])
        if len(matches) > 1:
            names = {_name_from_call(tc) for tc in matches}
            if len(names) == 1:
                return names.pop()
            return ""
    named = [tc for tc in pending if _name_from_call(tc)]
    if len(named) == 1:
        return _name_from_call(named[0])
    names = {_name_from_call(tc) for tc in named}
    if len(names) == 1 and names:
        return names.pop()
    return ""


class _NativeTraceBuilder:
    """Turns LangGraph `(namespace, mode, data)` chunks into drift trace records.

    With `stream_mode=["updates", "values"]` LangGraph emits, per super-step,
    one `updates` chunk (what the node wrote) followed by one `values` chunk
    (the full reducer-applied state). We pair them so `state_after` is the
    graph's real state — not drift's own `dict.update` approximation, which
    silently replaced reducer channels (`messages` with `add_messages`) with
    just the latest delta.

    Subgraph steps (`subgraphs=True`) arrive under a non-empty namespace,
    including subgraphs invoked from inside a tool call — the deepagents
    pattern that used to be invisible. They are recorded as their own steps,
    labelled `parent/inner`, with `state_after` = the ROOT state at that
    moment (unchanged until the parent node commits) and the subgraph's own
    state under `sub_state_after`. Keeping `state_after` root-shaped means
    every detector that diffs state across steps keeps a single schema.

    A subgraph step that belongs to a tool-dispatched sub-agent also carries
    `subagent` — the name recovered from the pending tool call that started
    it (see `_infer_subagent`), so `tools/model` reads as the critic's model
    step rather than an anonymous one.
    """

    def __init__(self, initial_state: dict, trace: list[dict], max_steps: int | None):
        self.ids = _ToolCallIds()
        self.root_state: dict = _normalize_messages(initial_state, ids=self.ids)
        self.trace = trace
        self.max_steps = max_steps
        self.step = 0
        self._pending: dict[tuple, list[dict]] = {}
        # last (sub_state, subagent) seen per namespace — a subgraph's final
        # step (deepagents' `after_model` middleware) often gets no values
        # chunk of its own, so it inherits the name from the previous one.
        self._last_sub: dict[tuple, tuple[dict | None, str]] = {}
        self._stop_requested = False

    def feed(self, item: Any) -> bool:
        """Ingest one chunk. Returns True when the caller should stop consuming."""
        if not (isinstance(item, tuple) and len(item) == 3):
            # Defensive: a shape we don't recognise — record it rather than drop it.
            self._record((), "_chunk", {"_value": item})
            return self._flush(())
        ns, mode, data = item
        ns_key = tuple(str(p) for p in (ns or ()))
        if mode == "updates":
            for node, update in _normalize_chunk(data).items():
                self._record(ns_key, node, update)
            return False
        if mode == "values":
            if not ns_key and isinstance(data, dict):
                self.root_state = _normalize_messages(data, ids=self.ids)
                # The parent committed: any subgraph step still waiting for a
                # values chunk is finished — close it against the state we have.
                for other in [k for k in self._pending if k]:
                    self._flush_orphans(other)
            return self._flush(ns_key, data)
        return False

    def _flush_orphans(self, ns_key: tuple) -> None:
        last_state, last_name = self._last_sub.get(ns_key, (None, ""))
        for rec in self._pending.pop(ns_key, []):
            rec["state_after"] = deepcopy(self.root_state)
            if last_name:
                rec["subagent"] = last_name

    def _record(self, ns_key: tuple, node: str, update: Any) -> None:
        self.step += 1
        labels = _namespace_labels(ns_key)
        rec = {
            "step": self.step,
            "node": "/".join(labels + [str(node)]) if labels else str(node),
            "update": _normalize_messages(update, ids=self.ids),
            "state_after": None,  # filled by the matching values chunk
        }
        if labels:
            rec["namespace"] = labels
        self.trace.append(rec)
        self._pending.setdefault(ns_key, []).append(rec)
        if self.max_steps is not None and self.step >= self.max_steps:
            self._stop_requested = True

    def _flush(self, ns_key: tuple, data: Any = None) -> bool:
        recs = self._pending.pop(ns_key, [])
        sub_state = None
        subagent = ""
        if ns_key and isinstance(data, dict) and recs:
            sub_state = _normalize_messages(data, ids=self.ids)
            subagent = _infer_subagent(self.root_state, sub_state)
            self._last_sub[ns_key] = (sub_state, subagent)
        for rec in recs:
            rec["state_after"] = deepcopy(self.root_state)
            if sub_state is not None:
                rec["sub_state_after"] = sub_state
                if subagent:
                    rec["subagent"] = subagent
        return self._stop_requested and not self._pending

    def finish(self) -> None:
        """Fill any record whose values chunk never arrived (crash mid-step)."""
        for ns_key in list(self._pending):
            if ns_key:
                self._flush_orphans(ns_key)
        for recs in self._pending.values():
            for rec in recs:
                if rec["state_after"] is None:
                    rec["state_after"] = deepcopy(self.root_state)
        self._pending.clear()


_NATIVE_STREAM_KWARGS: dict[str, Any] = {
    "stream_mode": ["updates", "values"],
    "subgraphs": True,
}


def _checkpointer_of(graph: Any) -> Any:
    """The graph's checkpointer, if compiled with one. Looks through a
    `.with_config(...)` RunnableBinding (`.bound`). None otherwise."""
    cp = getattr(graph, "checkpointer", None)
    if cp is None:
        cp = getattr(getattr(graph, "bound", None), "checkpointer", None)
    return cp or None


def _thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _accepts_config(fn: Any) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return "config" in sig.parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


async def _stream_or_invoke(
    graph: Any, state: dict | None, *, max_steps: int | None = None,
    trace_sink: list[dict] | None = None, config: dict | None = None,
    resume_state: dict | None = None,
) -> tuple[dict | None, list[dict], bool]:
    """Run the graph once. Capture a per-super-step trace if possible.

    Returns (final_state, trace, truncated). The trace is a list of
    records `{step, node, update, state_after[, namespace, sub_state_after, subagent]}`
    — empty if the graph supports invoke only (plain function /
    `.invoke()`-only stubs). `truncated` is True only when `max_steps` cut
    the run short (so `final_state` is the accumulated state AT that step,
    not the graph's true completion state — callers that surface
    `final_state` to a user must make that distinction visible, not call
    it "final" unqualified).

    `trace_sink`, when given, is the list records are appended to as they
    arrive. If the graph raises mid-run the caller still holds every step
    captured before the exception — the partial trace of a crash (a
    recursion-limit loop, say) is usually the most diagnostic artefact of
    the whole run, and discarding it was a real loss.

    Two streaming paths:

    * **Native LangGraph** (`.stream`/`.astream` accepts `stream_mode`):
      stream with `stream_mode=["updates", "values"], subgraphs=True`.
      `state_after` is the graph's real reducer-applied state and subgraph
      super-steps (including ones started inside a tool call) are visible
      as `parent/inner` steps. BaseMessage objects are normalised to small
      dicts — see `_message_to_dict`.
    * **Legacy / stub** (`.stream`/`.astream` without `stream_mode`): the
      default-mode chunks are `{node: update}`; we accumulate updates onto
      a running dict. Correct for plain-replace channels, approximate for
      reducer channels (there is no way to know a stub's reducer).

    Prefer `.astream()` -> `.stream()` -> `.ainvoke()` -> `.invoke()` ->
    `__call__`. `max_steps` only applies to the streaming paths — it stops
    consuming the generator once that many super-steps have been captured,
    which LangGraph supports (an abandoned async/sync generator is not an
    error). It has no effect on the plain invoke/call fallback paths.

    `config` is a LangGraph RunnableConfig (e.g. `{"configurable":
    {"thread_id": ...}}`) forwarded on the native path and to invoke-style
    entry points that accept it; stubs never see it. `state=None` with a
    checkpoint config RESUMES that checkpoint (LangGraph semantics); pass
    `resume_state` so the trace builder knows the state it is resuming from.
    """
    trace: list[dict] = trace_sink if trace_sink is not None else []
    cfg_args = (config,) if config is not None else ()

    astream = getattr(graph, "astream", None)
    stream = getattr(graph, "stream", None)
    native = (callable(astream) and _accepts_stream_mode(astream)) or (
        callable(stream) and _accepts_stream_mode(stream)
    )

    if native:
        builder = _NativeTraceBuilder(
            state if state is not None else (resume_state or {}), trace, max_steps,
        )
        truncated = False
        try:
            if callable(astream):
                async for item in astream(state, *cfg_args, **_NATIVE_STREAM_KWARGS):
                    if builder.feed(item):
                        truncated = True
                        break
            else:
                for item in stream(state, *cfg_args, **_NATIVE_STREAM_KWARGS):
                    if builder.feed(item):
                        truncated = True
                        break
        finally:
            builder.finish()
        return (builder.root_state, trace, truncated)

    if state is None:
        raise ValueError("resuming from a checkpoint (state=None) requires a native LangGraph graph")
    ids = _ToolCallIds()
    running: dict = _normalize_messages(state, ids=ids)
    step = 0

    def _legacy_record(node: str, update: Any) -> None:
        nonlocal step
        upd = _normalize_messages(update, ids=ids)
        if isinstance(upd, dict):
            # Mirror `add_messages` for conversation channels: a node's update
            # is a DELTA of new messages, so the accumulated state appends.
            # Plain replacement (the pre-11-04 bug for native graphs) left
            # `state_after.messages` holding only the latest message and
            # every message-aware detector blind on hand-rolled stubs.
            from drift.chaos.messages import is_messages_list as _is_msgs

            for k, v in upd.items():
                cur = running.get(k)
                if isinstance(v, list) and isinstance(cur, list) and v and cur and _is_msgs(v) and _is_msgs(cur):
                    running[k] = list(cur) + list(v)
                else:
                    running[k] = v
        step += 1
        trace.append({
            "step": step,
            "node": node,
            "update": upd,
            "state_after": deepcopy(running),
        })

    if callable(astream):
        async for chunk in astream(state):
            for node, update in _normalize_chunk(chunk).items():
                _legacy_record(node, update)
                if max_steps is not None and step >= max_steps:
                    return (running, trace, True)
        return (running, trace, False)

    if callable(stream):
        for chunk in stream(state):
            for node, update in _normalize_chunk(chunk).items():
                _legacy_record(node, update)
                if max_steps is not None and step >= max_steps:
                    return (running, trace, True)
        return (running, trace, False)

    # Plain invoke fallback: no per-step trace possible, no truncation possible.
    def _as_dict(result: Any) -> dict:
        return _normalize_messages(result if isinstance(result, dict) else dict(result))

    ainvoke = getattr(graph, "ainvoke", None)
    if callable(ainvoke):
        result = ainvoke(state, *(cfg_args if _accepts_config(ainvoke) else ()))
        if inspect.isawaitable(result):
            result = await result
        return (_as_dict(result), trace, False)

    invoke = getattr(graph, "invoke", None)
    if callable(invoke):
        result = invoke(state, *(cfg_args if _accepts_config(invoke) else ()))
        if inspect.isawaitable(result):
            result = await result
        return (_as_dict(result), trace, False)

    # _validate_graph guaranteed the graph is at minimum callable.
    result = graph(state)
    if inspect.isawaitable(result):
        result = await result
    return (_as_dict(result), trace, False)


def _output_view(initial: dict | None, final: dict | None) -> dict | None:
    """What the graph PRODUCED, for divergence purposes.

    A messages channel (`add_messages`) carries the run's input at the front
    of its final value. Comparing finals directly then reports a perturbed
    INPUT as a diverged OUTPUT — every message perturbation shows DIVERGE
    whether or not the graph's behaviour changed. So for each messages-shaped
    list field whose final value starts with that run's own initial value,
    keep only the appended tail. Other fields — including plain lists, where
    "echoed unchanged" and "replaced with an identical list" can't be told
    apart and the old whole-value compare stays right — pass through as-is.
    """
    from drift.chaos import messages as _m

    if final is None:
        return None
    if not initial:
        return final
    out = dict(final)
    for k, init_v in initial.items():
        fin_v = out.get(k)
        if not isinstance(fin_v, list) or not _m.is_messages_list(init_v):
            continue
        n = len(init_v)
        # Dialect-insensitive prefix check: the user's input is usually role
        # dicts / tuples while the final value is the normalised type-dict
        # form, so compare (role, text, tool_calls) rather than raw dicts.
        def _key(m: Any) -> tuple:
            return (
                _m.message_role(m),
                _m.message_text(m),
                _canonical([(c.get("name"), c.get("args")) for c in _m.message_tool_calls(m)]),
            )
        if len(fin_v) >= n and all(_key(a) == _key(b) for a, b in zip(fin_v[:n], init_v)):
            out[k] = fin_v[n:]
    return out


def _canonical(value: Any) -> str:
    """Deterministic JSON-y serialization for equality comparison.

    Used by tier 1 to canonicalize dict/list values before comparing — so
    {"a": 1, "b": 2} equals {"b": 2, "a": 1}. Falls back to repr() for
    types JSON can't serialize (e.g. langgraph BaseMessage instances).
    """
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except Exception:  # noqa: BLE001
        return repr(value)


def _shorten(s: str, limit: int = 60) -> str:
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _tier0_structural(
    baseline: dict, perturbed: dict
) -> list[FieldDivergence]:
    """Tier 0: structural changes — keys added/removed and type changes.

    These are always meaningful (they can't be model wobble), so they
    bypass the noise floor and all higher tiers.
    """
    out: list[FieldDivergence] = []
    bkeys, pkeys = set(baseline.keys()), set(perturbed.keys())

    for k in sorted(pkeys - bkeys):
        out.append(FieldDivergence(
            name=k, tier=0,
            baseline_value=None, perturbed_value=perturbed[k],
            summary=f"+{k} (key added)",
        ))
    for k in sorted(bkeys - pkeys):
        out.append(FieldDivergence(
            name=k, tier=0,
            baseline_value=baseline[k], perturbed_value=None,
            summary=f"-{k} (key removed)",
        ))
    for k in sorted(bkeys & pkeys):
        bt, pt = type(baseline[k]).__name__, type(perturbed[k]).__name__
        # bool vs int — Python treats them as same type for ==, but they're
        # semantically distinct decisions for a graph.
        if bt != pt:
            out.append(FieldDivergence(
                name=k, tier=0,
                baseline_value=baseline[k], perturbed_value=perturbed[k],
                summary=f"{k}: type {bt} -> {pt}",
            ))
    return out


def _tier1_exact(
    baseline: dict,
    perturbed: dict,
    skip_fields: set[str],
) -> list[FieldDivergence]:
    """Tier 1: canonical equality on remaining same-typed fields.

    skip_fields = names already reported by tier 0 (type-change etc.). Fields
    that survive tier 1 (i.e. still differ) are candidates for tier 2+.
    """
    out: list[FieldDivergence] = []
    for k in sorted(set(baseline.keys()) & set(perturbed.keys())):
        if k in skip_fields:
            continue
        bv, pv = baseline[k], perturbed[k]
        if bv == pv:
            continue
        bs, ps = _canonical(bv), _canonical(pv)
        if bs == ps:
            continue
        out.append(FieldDivergence(
            name=k, tier=1,
            baseline_value=bv, perturbed_value=pv,
            summary=f"{k}: {_shorten(bs)} -> {_shorten(ps)}",
        ))
    return out


def _diff_states(
    baseline: dict | None, perturbed: dict | None
) -> tuple[bool, str, list[FieldDivergence]]:
    """Backwards-compatible exact-mode comparator.

    Returns (diverged, summary_string, divergence_details). The first two
    fields preserve the legacy shape; the list lets phase-2 callers see
    per-field detail without re-parsing the summary string.

    For tiered comparison (similarity + judge), callers should use
    `_diff_states_tiered` directly.
    """
    if baseline is None or perturbed is None:
        return (False, "", [])
    tier0 = _tier0_structural(baseline, perturbed)
    seen = {d.name for d in tier0}
    tier1 = _tier1_exact(baseline, perturbed, skip_fields=seen)
    details = tier0 + tier1
    if not details:
        return (False, "", [])
    return (True, "; ".join(d.summary for d in details), details)


# --- Phase 2: noise floor + tier 2 (similarity) + tier 3 (judge) ----------


def _text_similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio over the raw text. Cheap, no embeddings.

    Range [0, 1]: 1 = identical, ~0.85 = same intent worded slightly differently,
    < 0.5 = substantively different. The exact cutoff is configurable per-call.
    """
    return SequenceMatcher(None, a, b).ratio()


def _analyze_field_variance(name: str, values: list[Any]) -> FieldNoiseBand:
    """Build a per-field noise band from N baseline samples.

    Strategy:
      - For enum-shaped fields (small distinct value set), record value
        frequencies. Tier 2 just checks membership.
      - For numeric fields (int / float, excluding bool), record min/max.
        Tier 2 checks the perturbed value's distance from the observed range.
      - For text fields (all samples are str), compute pairwise similarity
        stats. Tier 2 compares perturbed-vs-baseline similarity against the
        baseline-vs-baseline floor.
    """
    band = FieldNoiseBand(name=name, sample_count=len(values))
    # Distinct values (use canonical JSON as the dedup key so dicts/lists
    # with the same content collapse).
    seen: dict[str, Any] = {}
    freqs: Counter[str] = Counter()
    for v in values:
        key = _canonical(v)
        freqs[key] += 1
        seen.setdefault(key, v)
    band.distinct_values = list(seen.values())
    band.value_frequencies = dict(freqs)

    # Numeric stats (skip bool since bool is a subclass of int).
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values) and values:
        band.numeric_min = float(min(values))
        band.numeric_max = float(max(values))
        return band

    # Text stats. Strings compare as themselves; structured values (a list of
    # message dicts, a nested dict) compare on their canonical JSON, so an
    # LLM-written conversation gets the same wobble floor a plain text field
    # does instead of falling straight through to "not in the band".
    if len(values) >= 2 and all(isinstance(v, (str, list, dict)) for v in values):
        texts = [v if isinstance(v, str) else _canonical(v) for v in values]
        sims: list[float] = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sims.append(_text_similarity(texts[i], texts[j]))
        if sims:
            band.text_min_similarity = min(sims)
            band.text_mean_similarity = sum(sims) / len(sims)

    return band


async def _measure_noise_band(
    graph: Any,
    initial_state: dict,
    state_factory: Callable[[], dict] | None,
    n_rollouts: int,
    config_factory: Callable[[str], dict | None] | None = None,
) -> tuple[dict[str, FieldNoiseBand], list[dict | None]]:
    """Run baseline N times to amortize the noise floor.

    Returns (noise_band_by_field, list_of_final_states). The list of finals
    is exposed so the caller can pick one as THE baseline (we use the first)
    while still benefiting from the variance information from the rest.

    Note: drift can't re-seed an LLM. Re-running the same graph just gets
    whatever the model returns each time; for temp>0 models this is the
    natural wobble we want to measure. For deterministic graphs (mocks,
    temp=0 + cached LLMs), all N rollouts return the same final state and
    every band has distinct_values=[single value] — phase-2 still works,
    just with a zero-tolerance noise floor.
    """
    n_rollouts = max(1, int(n_rollouts))
    finals: list[dict | None] = []
    for i in range(n_rollouts):
        fresh = (
            deepcopy(state_factory()) if state_factory else deepcopy(initial_state)
        )
        cfg = config_factory(f"rollout-{i}") if config_factory else None
        final, _trace, etype, _err, _dur = await _run_one(graph, fresh, cfg)
        # If any rollout crashes we skip it for noise analysis but keep its
        # slot so callers can see counts didn't add up. Variance is measured
        # on what the graph produced, not on the echoed input (see _output_view).
        finals.append(None if etype else _output_view(_normalize_messages(fresh), final))

    successful = [f for f in finals if isinstance(f, dict)]
    if not successful:
        return ({}, finals)

    # Collect all keys that appeared in any successful rollout.
    all_keys: set[str] = set()
    for f in successful:
        all_keys.update(f.keys())

    bands: dict[str, FieldNoiseBand] = {}
    for k in all_keys:
        present_values = [f[k] for f in successful if k in f]
        if not present_values:
            continue
        bands[k] = _analyze_field_variance(k, present_values)
    return (bands, finals)


def _is_within_noise(
    perturbed_value: Any,
    band: FieldNoiseBand | None,
    similarity_threshold: float,
) -> tuple[bool, float | None]:
    """Tier 2 verdict: is this perturbed value plausibly within baseline noise?

    Returns (within_noise, similarity_score). similarity_score is None for
    non-text fields where the check is membership or range rather than ratio.

    Decision tree:
      - No noise band (band is None or sample_count < 2): we have no variance
        info, so we can't say "this is normal." Returns (False, None) and
        forces tier 3 or surfaces the divergence directly.
      - Perturbed value is exactly one of the observed distinct values: True.
      - Numeric band: within [min, max]: True.
      - Text band: similarity vs any baseline >= max(threshold, observed
        baseline-vs-baseline floor): True. We compare against EACH observed
        baseline value and take the best; this matches "is the perturbed
        output one of the things the baseline could plausibly have said?"
      - Otherwise: False.
    """
    if band is None or band.sample_count < 2:
        return (False, None)

    # Exact match against any observed distinct value (works for any type).
    pkey = _canonical(perturbed_value)
    if pkey in band.value_frequencies:
        return (True, None)

    # Numeric range check.
    if (
        band.numeric_min is not None
        and band.numeric_max is not None
        and isinstance(perturbed_value, (int, float))
        and not isinstance(perturbed_value, bool)
    ):
        return (band.numeric_min <= perturbed_value <= band.numeric_max, None)

    # Text similarity: take the best match against any observed baseline,
    # threshold against max(user_threshold, observed_baseline_floor).
    # Structured values compare on canonical JSON (see _analyze_field_variance).
    if band.text_min_similarity is not None and isinstance(perturbed_value, (str, list, dict)):
        p_text = perturbed_value if isinstance(perturbed_value, str) else _canonical(perturbed_value)
        baselines = [v if isinstance(v, str) else _canonical(v) for v in band.distinct_values]
        if not baselines:
            return (False, None)
        best_sim = max(_text_similarity(p_text, b) for b in baselines)
        floor = max(similarity_threshold, band.text_min_similarity)
        return (best_sim >= floor, best_sim)

    return (False, None)


_TIER3_JUDGE_SYSTEM = (
    "You are comparing two outputs from a multi-agent system to decide if "
    "they are semantically equivalent — i.e., do they convey the same "
    "decision, intent, or answer despite possibly differing in wording, "
    "formatting, or incidental detail.\n\n"
    "Reply strictly as JSON: {\"equivalent\": true|false, \"reasoning\": \"<one short sentence>\"}.\n"
    "Treat as equivalent: same decision in different words, reordered list "
    "with same items, same numeric answer formatted differently.\n"
    "Treat as NOT equivalent: different decision/action chosen, different "
    "answer, missing required information, added incorrect information."
)


async def _tier3_judge_equivalent(
    judge_llm: JudgeLLM,
    field: str,
    baseline_value: Any,
    perturbed_value: Any,
) -> tuple[bool, str]:
    """Tier 3 LLM equivalence check. Caller is responsible for budget gating.

    Returns (is_equivalent, reasoning). On any parse/network error, defaults
    to (False, "judge error: ...") so the divergence gets surfaced rather
    than silently dropped.
    """
    user = (
        f"Field: {field}\n\n"
        f"BASELINE:\n{_shorten(_canonical(baseline_value), 800)}\n\n"
        f"PERTURBED:\n{_shorten(_canonical(perturbed_value), 800)}"
    )
    try:
        raw = await judge_llm.judge(system=_TIER3_JUDGE_SYSTEM, user=user)
    except Exception as exc:  # noqa: BLE001 — judge is user-supplied, anything possible
        return (False, f"judge error: {type(exc).__name__}: {exc}")
    if not raw:
        return (False, "judge returned empty response")
    text = raw.strip()
    if not text.startswith("{"):
        i = text.find("{")
        text = text[i:] if i >= 0 else text
    if not text.endswith("}"):
        j = text.rfind("}")
        text = text[: j + 1] if j >= 0 else text
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return (False, f"judge non-JSON: {raw[:120]!r}")
    eq = bool(payload.get("equivalent"))
    reasoning = str(payload.get("reasoning") or "").strip() or ("equivalent" if eq else "different")
    return (eq, reasoning)


async def _diff_states_tiered(
    baseline: dict | None,
    perturbed: dict | None,
    noise_band: dict[str, FieldNoiseBand],
    judge_llm: JudgeLLM | None,
    similarity_threshold: float,
    judge_calls_remaining: int,
) -> tuple[bool, str, list[FieldDivergence], list[FieldDivergence], int]:
    """Full cost cascade. Returns (diverged, summary, confirmed, filtered, judge_used).

    Pipeline per field:
      tier 0 — structural change (always reported, never noise-filtered)
      tier 1 — exact canonical equality (drop if equal)
      tier 2 — within noise band? (drop if within)
      tier 3 — judge equivalence (drop if judged equivalent), budget-gated.

    Fields surviving all four tiers are returned as `confirmed` divergences.
    Fields dropped at tier 2 (noise band) or tier 3 (judge equivalent) are
    returned separately as `filtered` — these carry the similarity score /
    judge reasoning that justified the filter, so the user can audit
    UNCHANGED verdicts rather than trusting them blindly.
    """
    if baseline is None or perturbed is None:
        return (False, "", [], [], 0)

    tier0 = _tier0_structural(baseline, perturbed)
    seen0 = {d.name for d in tier0}
    tier1_candidates = _tier1_exact(baseline, perturbed, skip_fields=seen0)

    confirmed: list[FieldDivergence] = list(tier0)
    filtered: list[FieldDivergence] = []
    judge_used = 0

    for cand in tier1_candidates:
        band = noise_band.get(cand.name)
        within, sim = _is_within_noise(cand.perturbed_value, band, similarity_threshold)
        cand.similarity_score = sim
        cand.within_noise_band = within
        if within:
            # Filtered by noise floor — natural variance, not divergence.
            # Preserve the candidate so the user can audit "is this really
            # noise?" by inspecting similarity score vs the noise band.
            cand.tier = 2
            filtered.append(cand)
            continue
        if judge_llm is None or judge_calls_remaining - judge_used <= 0:
            # No judge available or budget exhausted: surface the
            # divergence; the user sees a tier-2 (or unmarked tier-1)
            # candidate with whatever similarity info we have.
            confirmed.append(cand)
            continue
        # Tier 3: ask the judge if these are semantically equivalent.
        judge_used += 1
        eq, reasoning = await _tier3_judge_equivalent(
            judge_llm, cand.name, cand.baseline_value, cand.perturbed_value,
        )
        cand.tier = 3
        cand.judge_equivalent = eq
        cand.judge_reasoning = reasoning
        if eq:
            # Judge cleared the field — keep the candidate around with its
            # reasoning string so the user can audit the verdict.
            filtered.append(cand)
            continue
        cand.summary = f"{cand.summary}  · judge: {reasoning}"
        confirmed.append(cand)

    if not confirmed:
        return (False, "", [], filtered, judge_used)
    return (
        True,
        "; ".join(d.summary for d in confirmed),
        confirmed,
        filtered,
        judge_used,
    )


# Crash classes. `error_type` is whatever the exception class was called;
# `crash_class` is drift's coarse reading of it so callers (and the UI) can
# tell "the graph looped until LangGraph's recursion limit stopped it" apart
# from "a node raised". A recursion-limit crash is a coordination failure in
# its own right — the loop is the finding — and the partial trace we now keep
# for crashes shows exactly which nodes were cycling.
CRASH_CLASS_RECURSION_LIMIT = "recursion_limit"
CRASH_CLASS_EXCEPTION = "exception"


def _classify_crash(error_type: str, error: str = "") -> str:
    if not error_type:
        return ""
    blob = f"{error_type} {error}".lower()
    if "graphrecursionerror" in blob or ("recursion" in blob and "limit" in blob):
        return CRASH_CLASS_RECURSION_LIMIT
    return CRASH_CLASS_EXCEPTION


async def _run_one(
    graph: Any, initial_state: dict, config: dict | None = None,
) -> tuple[dict | None, list[dict], str, str, float]:
    """Invoke the graph once, capturing exceptions, trace, and timing.

    Returns (final_state_or_None, trace, error_type, error_msg, duration).
    Trace is [] if the graph doesn't support streaming or if it crashed
    before any super-step yielded. On a crash AFTER some super-steps
    yielded, the partial trace is returned — not discarded.
    """
    t0 = time.perf_counter()
    sink: list[dict] = []
    try:
        final, trace, _truncated = await _stream_or_invoke(
            graph, initial_state, trace_sink=sink, config=config,
        )
        return (final, trace, "", "", time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001 — user code, anything possible
        return (None, sink, type(exc).__name__, str(exc), time.perf_counter() - t0)


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


def _run_coordination_library(
    trace: list[dict],
    initial_state: dict | None,
    baseline_state: dict | None,
    *,
    enabled: bool = True,
    baseline_trace: list[dict] | None = None,
    baseline_initial_state: dict | None = None,
    roles_by_agent: dict[str, str] | None = None,
    required_followups: dict[str, str] | None = None,
    policy_field: str | None = None,
    guideline_rules: list | None = None,
) -> list[dict]:
    """Run the curated coordination-failure detector library over one trace.

    Free + deterministic — no LLM cost. Returns FailureRecord dicts. Empty
    when disabled, when the trace is empty (graph doesn't stream or crashed
    at step 0), or when no detector matched. `baseline_trace` /
    `baseline_initial_state` let comparison detectors (retry_amplification)
    measure the unperturbed run the same way as this one.
    """
    if not enabled or not trace:
        return []
    findings = _run_library_on_trace(
        trace,
        initial_state=initial_state,
        baseline_state=baseline_state,
        baseline_trace=baseline_trace,
        baseline_initial_state=baseline_initial_state,
        roles_by_agent=roles_by_agent,
        required_followups=required_followups,
        policy_field=policy_field,
        guideline_rules=guideline_rules,
    )
    return [f.model_dump(mode="json") for f in findings]


def _normalize_rules(rules: Any) -> tuple[list, list[str], list[dict]]:
    """(rule objects for the library, judge strings, declared chaos entries)
    from whatever the caller passed as `rules`. Raises ValueError on a
    malformed spec so the error reaches the caller instead of being
    swallowed as a detector crash."""
    if rules is None:
        return [], [], []
    from drift import guidelines as _g

    if isinstance(rules, _g.GuidelineSpec):
        return list(rules.rules), list(rules.judge), list(rules.chaos)
    if isinstance(rules, (str, Path)):
        try:
            spec = _g.load(rules)
        except (OSError, _g.GuidelineError) as exc:
            raise ValueError(f"rules: {exc}") from exc
        return list(spec.rules), list(spec.judge), list(spec.chaos)
    try:
        return _g.parse_rules(list(rules)), [], []
    except _g.GuidelineError as exc:
        raise ValueError(f"rules: {exc}") from exc


def _build_declared_chaos(entries: list[dict], catalog: list) -> list:
    """Turn drift.yml `chaos:` entries into events, matched against the
    discovered catalogue so a typo is an error, not a silent skip.

        - {tool: web_search, fault: tool_timeout, on_call: 2}
        - {field: approved, pattern: flip_bool}
        - {pattern: inject_tool_result_instruction}        # messages field implied
        - {pattern: drop_last_user_message, step: 3}       # mid-run timing: fork after baseline step 3
    """
    if not entries:
        return []
    from drift.chaos.tools import TOOL_FAULTS, ToolChaosEvent

    by_name = {sp.pattern_name: sp for sp in catalog}
    events = []
    for i, raw in enumerate(entries, start=1):
        e = dict(raw)
        step = e.pop("step", None)
        if step is not None and (not isinstance(step, int) or isinstance(step, bool) or step < 1):
            raise ValueError(f"chaos #{i}: step must be an int >= 1 (a baseline root step to fork after)")
        if "tool" in e:
            tool, fault = str(e.pop("tool")), str(e.pop("fault", "tool_error"))
            on_call = e.pop("on_call", 1)
            if fault not in TOOL_FAULTS:
                raise ValueError(f"chaos #{i}: unknown tool fault {fault!r}; one of {', '.join(TOOL_FAULTS)}")
            if f"{fault}[{tool}]" not in by_name:
                have = sorted({sp.field for sp in catalog if sp.pattern_type in TOOL_FAULTS})
                raise ValueError(
                    f"chaos #{i}: tool {tool!r} is not reachable through a ToolNode of this graph "
                    f"(tools found: {', '.join(have) or 'none'})"
                )
            if not isinstance(on_call, int) or isinstance(on_call, bool) or on_call < 1:
                raise ValueError(f"chaos #{i}: on_call must be an int >= 1")
            events.append(ToolChaosEvent(tool, fault=fault, on_call=on_call, seed=i))
        elif "pattern" in e:
            pattern = str(e.pop("pattern"))
            fld = str(e.pop("field", "messages"))
            name = f"{pattern}[{fld}]"
            if name not in by_name:
                near = sorted(n for n in by_name if n.startswith(pattern + "[") or n.endswith(f"[{fld}]"))
                raise ValueError(
                    f"chaos #{i}: no pattern {name!r} for this state"
                    + (f"; close: {', '.join(near[:6])}" if near else "")
                )
            events.append(by_name[name].build())
        else:
            raise ValueError(f"chaos #{i}: needs `tool:` + `fault:` or `pattern:` (+ `field:`)")
        if e:
            raise ValueError(f"chaos #{i}: unknown key(s) {', '.join(sorted(e))}")
        if step is not None:
            events[-1].fork_rank = step
    return events


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
    divergence_mode: str = DEFAULT_DIVERGENCE_MODE,
    baseline_rollouts: int = DEFAULT_BASELINE_ROLLOUTS,
    perturbation_rollouts: int = DEFAULT_PERTURBATION_ROLLOUTS,
    max_judge_calls: int = DEFAULT_MAX_JUDGE_CALLS,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    run_coordination_detectors: bool = True,
    coordination_roles: dict[str, str] | None = None,
    required_followups: dict[str, str] | None = None,
    policy_field: str | None = None,
    rules: Any = None,
    tool_chaos: bool = True,
    chaos_timing: str = CHAOS_TIMING_INITIAL,
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
    if divergence_mode not in ("exact", "tiered", "off"):
        raise ValueError(
            f"divergence_mode must be 'exact', 'tiered', or 'off'; got {divergence_mode!r}"
        )
    if not isinstance(perturbation_rollouts, int) or perturbation_rollouts < 1:
        raise ValueError(
            f"perturbation_rollouts must be an int >= 1; got {perturbation_rollouts!r}"
        )
    if chaos_timing not in CHAOS_TIMINGS:
        raise ValueError(f"chaos_timing must be one of {', '.join(CHAOS_TIMINGS)}; got {chaos_timing!r}")
    _validate_graph(graph)
    if chaos_timing == CHAOS_TIMING_MID_RUN:
        ensure_checkpointer(graph)

    state_cls = _build_state_model(initial_state)
    level = _normalize_intensity(intensity)
    guidelines = [g for g in (user_guidelines or []) if g and str(g).strip()]
    # `rules` may be a GuidelineSpec, a path to a guideline file, or a list
    # of rule mappings / callables. A spec's judge strings join the judge's
    # guidelines; its rules run inside the coordination library.
    rule_list, spec_judge, declared_chaos = _normalize_rules(rules)
    for g in spec_judge:
        if g not in guidelines:
            guidelines.append(g)

    # A graph compiled with a checkpointer refuses to run without a thread_id,
    # and two runs on the SAME thread would share state (the second run's
    # messages land on top of the first's). Every run drift makes — baseline,
    # noise rollouts, each perturbation — gets its own thread.
    run_tag = uuid.uuid4().hex[:10]
    has_checkpointer = _checkpointer_of(graph) is not None

    def _cfg(label: str) -> dict | None:
        return _thread_config(f"drift-{run_tag}-{label}") if has_checkpointer else None

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
    # Tool-layer patterns: one per (tool the graph can call through a
    # ToolNode × fault). The auto plan draws from a curated subset of faults
    # so a three-tool graph is not all tool chaos; `exhaustive` takes every
    # fault; a drift.yml `chaos:` entry can name any of them.
    tool_specs_all = discover_tool_patterns(graph, seed=seed) if tool_chaos else []
    tool_specs_auto = (
        tool_specs_all if level == "exhaustive"
        else [sp for sp in tool_specs_all if sp.pattern_type in DEFAULT_AUTO_FAULTS]
    )
    all_specs = [*all_specs, *tool_specs_all]
    scheduled = plan_auto_chaos(
        state=seed_state,  # type: ignore[arg-type]
        steps=_PLANNER_HORIZON,
        intensity=level,
        seed=seed,
        exclude=auto_chaos_exclude,
        extra_catalog=tool_specs_auto,
    )

    # In exhaustive mode the user opted into "every applicable pattern" —
    # silently truncating to max_perturbations would defeat the entire
    # contract. Auto-raise the ceiling to the catalog size when the caller
    # left it at the default; if the caller explicitly passed a smaller
    # value, honor it but emit a warning via the diverged_summary plumbing
    # would be out of scope here — we just truncate as before.
    if level == "exhaustive" and max_perturbations == DEFAULT_MAX_PERTURBATIONS:
        max_perturbations = max(max_perturbations, len(scheduled))

    # Clamp to the per-call ceiling so an aggressive intensity on a noisy
    # schema doesn't silently rack up LLM cost (plus optional judge cost).
    if len(scheduled) > max_perturbations:
        scheduled = scheduled[:max_perturbations]

    # Chaos the user asked for by name (drift.yml `chaos:`) always runs,
    # first, on top of the plan and outside the ceiling — it is explicit.
    declared = _build_declared_chaos(declared_chaos, all_specs)
    if declared:
        names = {ev.name for ev in declared}
        mid = _PLANNER_HORIZON // 2   # mid-run timing forks a declared event mid-baseline unless `step:` says where
        scheduled = [(mid, ev) for ev in declared] + [(t, ev) for t, ev in scheduled if ev.name not in names]

    # Baseline run uses a fresh copy of the user's state so the graph can't
    # mutate the input dict and leak across perturbation runs.
    # In tiered mode with baseline_rollouts > 1, we measure natural variance
    # FIRST (N rollouts), then use the first successful one as THE baseline.
    # Otherwise we just run baseline once like before.
    noise_band: dict[str, FieldNoiseBand] = {}
    if divergence_mode == "tiered" and baseline_rollouts > 1:
        noise_band, finals = await _measure_noise_band(
            graph, initial_state, state_factory, baseline_rollouts, _cfg,
        )
        # Pick the first successful rollout as THE baseline. Time/trace/etc.
        # are re-derived from a one-shot run below so we have consistent
        # trace data — the noise band is derived from final-state variance,
        # not from per-step traces, so we don't need to keep all N traces.
        baseline_input = (
            deepcopy(state_factory()) if state_factory else deepcopy(initial_state)
        )
        baseline_cfg = _cfg("baseline")
        bfinal, btrace, betype, berr, btime = await _run_one(graph, baseline_input, baseline_cfg)
    else:
        baseline_input = (
            deepcopy(state_factory()) if state_factory else deepcopy(initial_state)
        )
        baseline_cfg = _cfg("baseline")
        bfinal, btrace, betype, berr, btime = await _run_one(graph, baseline_input, baseline_cfg)

    baseline_findings: list[dict] = []
    if judge_llm is not None and btrace:
        baseline_findings = await _run_judge_on_trace(
            judge_llm, guidelines, btrace, state_cls,
        )
    baseline_coord = _run_coordination_library(
        btrace, initial_state, bfinal,
        enabled=run_coordination_detectors,
        roles_by_agent=coordination_roles,
        required_followups=required_followups,
        policy_field=policy_field,
        guideline_rules=rule_list,
    )
    baseline = BaselineResult(
        initial_state=baseline_input,
        final_state=bfinal,
        crashed=bool(betype),
        error=berr,
        error_type=betype,
        crash_class=_classify_crash(betype, berr),
        duration_s=btime,
        trace=btrace,
        judge_findings=baseline_findings,
        coordination_findings=baseline_coord,
        thread_id=(baseline_cfg or {}).get("configurable", {}).get("thread_id", ""),
    )
    # Mid-run timing forks the baseline: pin its root checkpoints now,
    # before any fork writes a branch onto the same thread.
    root_snaps: dict[int, Any] = {}
    n_root_steps = sum(1 for rec in btrace if not rec.get("namespace"))
    midrun_blocker = ""
    if chaos_timing == CHAOS_TIMING_MID_RUN:
        if betype:
            midrun_blocker = "the baseline crashed, so there is no run to fork"
        elif not baseline.thread_id or not callable(getattr(graph, "update_state", None)):
            midrun_blocker = "the graph has no checkpointer / update_state (not a compiled LangGraph?)"
        elif n_root_steps < 2:
            midrun_blocker = f"the baseline has {n_root_steps} root step(s); forking needs at least 2"
        else:
            root_snaps = _capture_root_checkpoints(graph, baseline.thread_id)
            if not any(1 <= r <= n_root_steps - 1 for r in root_snaps):
                midrun_blocker = "no root checkpoints were recorded on the baseline thread"

    judge_calls_used = 0
    perturbations: list[PerturbationResult] = []

    async def _rollout(event: Any, record: Any, post_state: dict, label: str,
                       rollout_index: int, fork: dict | None = None,
                       timing_note: str = "") -> PerturbationResult:
        """Run ONE perturbed state once and score it. Called N times per
        perturbation when perturbation_rollouts > 1 — same perturbed input
        (deep-copied), its own thread, its own trace and findings.

        `fork` (mid-run timing) = {"snap", "rank", "edits"}: instead of a
        fresh run, write `edits` onto the baseline checkpoint `snap` with
        update_state (a new branch each call) and resume it."""
        nonlocal judge_calls_used
        pert_cfg = _cfg(label)
        timing = CHAOS_TIMING_INITIAL
        fork_step: int | None = None
        if fork is not None:
            try:
                cfg = _write_fork_edits(graph, fork)
                resumed = _normalize_messages(dict(graph.get_state(cfg).values or {}))
            except Exception as exc:  # noqa: BLE001
                # update_state writes the edit AS the last node and then
                # evaluates that node's outgoing edges, so a routing function
                # that cannot cope with the perturbed state raises here —
                # e.g. `state["messages"][-1]` after clear_list emptied the
                # channel. That is the graph crashing on the perturbation,
                # and it is reported as exactly that.
                return PerturbationResult(
                    event_name=event.name,
                    event_summary=record.summary + f" — applied mid-run, after baseline step {fork['rank']}"
                    + " — the graph raised while the edit was being written to the checkpoint (update_state "
                      "re-evaluates the last node's routing)",
                    perturbed_field=getattr(event, "field", ""),
                    pattern_type=getattr(event, "pattern", "auto_chaos"),
                    fault_class=getattr(event, "fault_class", ""),
                    perturbed_initial_state=post_state,
                    final_state=None, crashed=True, error=str(exc), error_type=type(exc).__name__,
                    crash_class=_classify_crash(type(exc).__name__, str(exc)),
                    thread_id=baseline.thread_id, rollout_index=rollout_index,
                    chaos_timing=CHAOS_TIMING_MID_RUN, fork_step=fork["rank"], timing_note=timing_note,
                )
            timing, fork_step, pert_cfg, post_state = CHAOS_TIMING_MID_RUN, fork["rank"], cfg, resumed
        # A tool-layer event lands while the graph runs: its wrapper sits on
        # the graph's ToolNodes for exactly this invocation.
        installer = event.install(graph) if is_tool_event(event) else contextlib.nullcontext()
        with installer:
            if fork is not None:
                final, ptrace, etype, err, took = await _run_one_resume(graph, pert_cfg, post_state)
            else:
                final, ptrace, etype, err, took = await _run_one(graph, deepcopy(post_state), pert_cfg)
        tool_note = event.outcome_note() if is_tool_event(event) else ""
        fault_fired = event.fired if is_tool_event(event) else None
        if timing == CHAOS_TIMING_MID_RUN:
            tool_note += f" — applied mid-run, after baseline step {fork_step}"
            # A resumed run's steps continue the baseline's numbering, so
            # step 5 in this trace is the same position as baseline step 5.
            for rec in ptrace:
                rec["step"] = int(rec.get("step", 0)) + int(fork_step or 0)
        if timing_note:
            tool_note += f" — {timing_note}"

        # Divergence detection: dispatch through the cost cascade.
        # `divfiltered` carries tier-2/3 candidates the cascade dropped —
        # preserved so the UI can audit UNCHANGED verdicts.
        divfiltered: list[FieldDivergence] = []
        # Compare what each run PRODUCED: reducer channels (messages) echo the
        # run's input at the front of the final value, and the perturbed run's
        # input differs by construction. See _output_view.
        baseline_out = _output_view(_normalize_messages(baseline_input), baseline.final_state)
        # A resumed run's final state carries the whole thread from step 1,
        # so it is viewed against the ORIGINAL input, like the baseline.
        perturbed_out = _output_view(
            _normalize_messages(baseline_input if timing == CHAOS_TIMING_MID_RUN else post_state), final,
        )
        if divergence_mode == "off":
            diverged, divsum, divdetails = (False, "", [])
        elif divergence_mode == "tiered":
            remaining = max(0, max_judge_calls - judge_calls_used)
            diverged, divsum, divdetails, divfiltered, used = await _diff_states_tiered(
                baseline_out, perturbed_out, noise_band,
                judge_llm if remaining > 0 else None,
                similarity_threshold, remaining,
            )
            judge_calls_used += used
        else:  # "exact"
            diverged, divsum, divdetails = _diff_states(baseline_out, perturbed_out)

        # Judge runs even on crashes — partial traces (steps before the crash)
        # are often the most diagnostic. Skipped only when there's literally
        # no trace data to feed it (graph doesn't stream, or crashed at step 0).
        pert_findings: list[dict] = []
        if judge_llm is not None and ptrace:
            pert_findings = await _run_judge_on_trace(
                judge_llm, guidelines, ptrace, state_cls,
            )

        # Coordination-failure detector library — deterministic, free, always
        # runs when a trace exists (unless the user opts out).
        pert_coord = _run_coordination_library(
            ptrace, post_state, baseline.final_state,
            enabled=run_coordination_detectors,
            baseline_trace=baseline.trace,
            baseline_initial_state=baseline.initial_state,
            roles_by_agent=coordination_roles,
            required_followups=required_followups,
            policy_field=policy_field,
            guideline_rules=rule_list,
        )

        return PerturbationResult(
            event_name=event.name,
            event_summary=record.summary + tool_note,
            perturbed_field=getattr(event, "field", ""),
            pattern_type=getattr(event, "pattern", "auto_chaos"),
            fault_class=getattr(event, "fault_class", ""),
            fault_layer=FAULT_LAYER_TOOL if is_tool_event(event) else FAULT_LAYER_STATE,
            fault_fired=fault_fired,
            perturbed_initial_state=post_state,
            final_state=final,
            crashed=bool(etype),
            error=err,
            error_type=etype,
            crash_class=_classify_crash(etype, err),
            diverged=diverged,
            divergence_summary=divsum,
            duration_s=took,
            trace=ptrace,
            judge_findings=pert_findings,
            divergence_details=divdetails,
            filtered_divergences=divfiltered,
            coordination_findings=pert_coord,
            thread_id=(pert_cfg or {}).get("configurable", {}).get("thread_id", ""),
            rollout_index=rollout_index,
            chaos_timing=timing,
            fork_step=fork_step,
            timing_note=timing_note,
        )

    for pert_index, (_t, event) in enumerate(scheduled):
        fork: dict | None = None
        timing_note = ""
        if chaos_timing == CHAOS_TIMING_MID_RUN and not is_tool_event(event):
            if midrun_blocker:
                timing_note = f"fell back to initial-state timing: {midrun_blocker}"
            else:
                wanted = getattr(event, "fork_rank", None)
                rank = min(n_root_steps - 1, wanted) if wanted else _fork_rank_for(_t, n_root_steps)
                if not wanted:
                    rank = _choose_fork_rank(rank, root_snaps, n_root_steps)
                snap = root_snaps.get(rank)
                if snap is None:
                    timing_note = f"fell back to initial-state timing: no checkpoint after baseline step {rank}"
                else:
                    # Apply the pattern to the checkpoint's own values (real
                    # message objects, ids intact) and express the change as
                    # an update_state payload. Applied once; every rollout
                    # writes the same edits onto a fresh branch.
                    before = {k: deepcopy(v) for k, v in dict(snap.values or {}).items() if k != "timestep"}
                    try:
                        world = _materialize_world(state_cls, before)
                        record = event.apply(world)
                        edits = _checkpoint_edits(before, _world_fields(world, before.keys()))
                    except Exception as exc:  # noqa: BLE001
                        timing_note = (f"fell back to initial-state timing: could not apply the pattern to "
                                       f"the checkpoint ({type(exc).__name__}: {exc})")
                    else:
                        root_nodes = [rec.get("node") for rec in btrace if not rec.get("namespace")]
                        as_node = str(root_nodes[rank - 1]) if 0 < rank <= len(root_nodes) else ""
                        fork = {"snap": snap, "rank": rank, "edits": edits, "as_node": as_node}
                        post_state = _normalize_messages(before)
        if fork is None:
            # Each perturbation gets its own World built from a fresh copy of
            # the user's state, then has one chaos event applied.
            pert_input = (
                deepcopy(state_factory()) if state_factory else deepcopy(initial_state)
            )
            world = _materialize_world(state_cls, pert_input)
            record = event.apply(world)
            post_state = _state_to_dict(world)

        # Rollout 0 keeps the historical thread label `p<i>`; repeats are
        # `p<i>r<j>`. Every rollout is an independent sample of the graph's
        # response to this exact input.
        rollouts: list[PerturbationResult] = []
        for j in range(perturbation_rollouts):
            label = f"p{pert_index}" if j == 0 else f"p{pert_index}r{j}"
            rollouts.append(await _rollout(event, record, post_state, label, j, fork=fork, timing_note=timing_note))
        # The representative is the most severe rollout, so the top-level
        # verdict reads "in at least one of N runs". The rest ride along.
        primary = min(rollouts, key=lambda r: (_OUTCOME_RANK[r.outcome], r.rollout_index))
        primary.rollouts = [r for r in rollouts if r is not primary]
        perturbations.append(primary)


    return AdapterResult(
        baseline=baseline,
        perturbations=perturbations,
        intensity=level,
        patterns_total=len(all_specs),
        divergence_mode=divergence_mode,
        baseline_rollouts=baseline_rollouts,
        perturbation_rollouts=perturbation_rollouts,
        noise_band=noise_band,
        judge_calls_used=judge_calls_used,
        judge_calls_budget=max_judge_calls,
        chaos_timing=chaos_timing,
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
    divergence_mode: str = DEFAULT_DIVERGENCE_MODE,
    baseline_rollouts: int = DEFAULT_BASELINE_ROLLOUTS,
    perturbation_rollouts: int = DEFAULT_PERTURBATION_ROLLOUTS,
    max_judge_calls: int = DEFAULT_MAX_JUDGE_CALLS,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    run_coordination_detectors: bool = True,
    coordination_roles: dict[str, str] | None = None,
    required_followups: dict[str, str] | None = None,
    policy_field: str | None = None,
    rules: Any = None,
    tool_chaos: bool = True,
    chaos_timing: str = CHAOS_TIMING_INITIAL,
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
            "aggressive" (~35%) | "exhaustive". Same scale as drift.run's
            auto_chaos. True is an alias for "moderate". "exhaustive"
            schedules EVERY applicable chaos pattern in the schema exactly
            once (deterministic; ignores seed for selection). Use it for
            pre-deploy gates where you want full schema coverage; cost
            scales with schema breadth, so a graph that LLM-calls per
            invocation will see one LLM call per fuzzable pattern.
        seed: RNG seed for reproducible perturbation selection.
        auto_chaos_exclude: substrings to skip when scheduling chaos.
            E.g. ["flip_bool"] disables all bool flips; ["messages"]
            disables every pattern targeting the `messages` field.
        max_perturbations: hard cap on perturbation runs per call.
            Each perturbation is a full graph invocation; if your graph
            calls an LLM, this directly bounds cost. Default 25.
            When intensity="exhaustive" AND max_perturbations is at the
            default, the ceiling is auto-raised to the catalog size so
            "exhaustive" actually means exhaustive. Explicit caps still win.
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
        rules: deterministic guideline rules (see drift.guidelines): a
            `GuidelineSpec`, a path to `drift.yml`, or a list of rule
            mappings / `@drift.guideline` functions. Run for free on the
            baseline and every perturbation; violations are coordination
            findings with `failure_type = "guideline:<id>"`, so
            `--fail-on guideline` / `finding:guideline:<id>` gate on them.
            A spec's `judge:` strings are added to `user_guidelines`; its
            `chaos:` entries (`{tool: web_search, fault: tool_timeout,
            on_call: 2}` / `{field: approved, pattern: flip_bool}`) are
            always scheduled, first, outside `max_perturbations`.
        tool_chaos: also discover tool-layer patterns (drift.chaos.tools):
            runtime faults on one call to one tool the graph reaches
            through a ToolNode — error, timeout, rate limit, empty,
            truncated, stale, injected instruction, hijack. The state is
            untouched; the graph meets the fault mid-run. Patterns are
            named `tool_timeout[web_search]` and excludable like any other.
            Default True; a graph without ToolNodes gets none.
        chaos_timing: "initial" (default) perturbs the initial state and
            reruns the graph. "mid_run" runs the baseline on a checkpointer,
            forks it at the step the planner scheduled each pattern for,
            applies the pattern to that checkpoint's state via update_state
            and resumes — the graph meets the fault mid-task, context in
            place (an injected tool result is then always "warm"). Needs a
            compiled LangGraph (drift attaches a MemorySaver if there is
            none); anything that cannot be forked falls back to "initial"
            for that perturbation, with `timing_note` saying why. Each
            result carries `chaos_timing` and `fork_step`.
        divergence_mode: how to compare baseline vs perturbed final state.
            "exact" (default) — equality on the whole dict; fast/free but
            noisy on LLM-driven graphs. "tiered" — cost cascade: structural
            → exact → similarity (vs measured noise floor) → LLM judge,
            with each tier filtering candidates before the next. Most
            comparisons resolve in tiers 0-1 for free; tier 3 only fires
            on survivors and is hard-capped. "off" — skip divergence
            detection entirely (crash detection still runs).
        baseline_rollouts: when divergence_mode="tiered", how many times to
            run the baseline to measure natural variance. >1 enables the
            noise floor that tier 2 consults. Cost = `baseline_rollouts`
            extra graph invocations once per drift_test call (amortized).
            Default 1 (no noise floor).
        perturbation_rollouts: how many times to run EACH perturbed state.
            Default 1. For an LLM-driven graph one perturbed run is a
            sample, not a verdict (a planted prompt injection was followed
            in 1 of 4 live runs). With N > 1 every PerturbationResult is
            the most severe of its N rollouts, carries the others in
            `.rollouts`, and reports `.outcome_counts` ("crashed 1/5 ·
            unchanged 4/5"), `.finding_rollout_counts` (detector → rollouts
            it fired in) and `.is_intermittent`. Cost = N × perturbation
            graph runs (+ N × judge calls when a judge is on).
        max_judge_calls: hard ceiling on tier-3 judge calls across all
            perturbations in this drift_test call. Default 10. Tier 0-2
            are free; only tier 3 costs LLM tokens for divergence
            equivalence checks.
        similarity_threshold: tier-2 lexical similarity cutoff for text
            fields, 0..1. Default 0.85. Effective threshold is
            max(this, measured baseline-vs-baseline floor) so text fields
            with naturally high variance get a tighter band automatically.
        coordination_roles: agent/node name -> role label (e.g. "verifier",
            "executor"). Powers role-aware library detectors (role_drift);
            falls back to name-pattern matching when omitted.
        required_followups: trigger agent/node name -> required follow-up
            agent/node name. Powers `premature_termination` — see that
            detector's docstring. Provisional single-hint config, not a
            general rule DSL; omit to leave that detector silent.
        policy_field: state field name holding a policy/config version
            (e.g. "policy_version"). Powers `policy_inconsistency_across_steps`
            — see that detector's docstring. Omit to leave it silent.

    Returns:
        AdapterResult with .baseline and .perturbations; see those
        classes for fields. .summary_lines() gives a quick stdout report.

        Each PerturbationResult carries:
          - divergence_details   — confirmed divergences after the cascade
          - filtered_divergences — tier-2/3 candidates the cascade dropped
            (within-noise-band matches OR judge-equivalent matches), each
            with the similarity score or judge_reasoning that justified
            the filter. Use this to audit UNCHANGED verdicts.

    Notes:
        This calls asyncio.run() internally — don't call from inside an
        already-running event loop. Use drift_test_async in that case.
        Cost in tiered mode with noise floor and budget B:
            baseline_rollouts + 1 + perturbation_rollouts × len(scheduled_perturbations) graph runs
            + up to (B + 1 if baseline has trace) judge calls for coord findings
            + up to B judge calls for divergence equivalence (capped).
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
            divergence_mode=divergence_mode,
            baseline_rollouts=baseline_rollouts,
            perturbation_rollouts=perturbation_rollouts,
            max_judge_calls=max_judge_calls,
            similarity_threshold=similarity_threshold,
            run_coordination_detectors=run_coordination_detectors,
            coordination_roles=coordination_roles,
            required_followups=required_followups,
            policy_field=policy_field,
            rules=rules,
            tool_chaos=tool_chaos,
            chaos_timing=chaos_timing,
        )
    )


# =============================================================================
# Fork-edit-replay — v1
# =============================================================================
#
# Design spec: docs/design/fork_edit_replay_v1.md.
# v1 scope: rerun-from-state execution, sparse deep-merge edits, state only,
# run-to-completion, optional top-vs-bottom compare. Deferred features
# (prompt editing, bounded replay, consistency check) live in
# memory/feature_ideas.md — do not forget them.
#
# The primitive is diagnostic, not therapeutic. Fix goes in the parent graph;
# fork-edit is how the developer isolates WHERE to fix.


@dataclass
class ForkBranch:
    """One branch of a fork-edit-replay run — either the fork-edited branch or
    the optional top-edited (initial-state-edited) branch. Same shape as a
    baseline run but scoped to what came out of THIS fork."""

    initial_state: dict            # the effective initial state we ran from
    trace: list[dict] = field(default_factory=list)
    final_state: dict | None = None
    crashed: bool = False
    error: str = ""
    error_type: str = ""
    # "" when not crashed; else "recursion_limit" | "exception" — see _classify_crash.
    crash_class: str = ""
    duration_s: float = 0.0
    coordination_findings: list[dict] = field(default_factory=list)
    # True when `max_steps` cut this branch short — final_state is the
    # accumulated state AT that step, not the graph's true completion
    # state. See drift_test_fork's `max_steps` docstring.
    truncated: bool = False
    # "checkpoint" — resumed a real LangGraph checkpoint on the parent's
    # thread after update_state (true time-travel; the graph continues from
    # exactly where it was, pending tasks included).
    # "rerun_from_state" — invoked the graph fresh with the edited
    # state_after as a new initial state (the v1 approximation).
    execution_mode: str = "rerun_from_state"
    thread_id: str = ""


FORK_MODE_CHECKPOINT = "checkpoint"
FORK_MODE_RERUN = "rerun_from_state"


@dataclass
class ForkResult:
    """Result of one drift_test_fork call.

    parent_summary carries only enough of the parent to render the diff view
    without holding a reference to the full AdapterResult (keeps this
    serializable + storable independently).

    fork_branch is always present — the branch produced by editing state at
    fork_step and running forward.

    top_edited_branch is populated only when the user passed
    `also_apply_at_initial=True`. It applied the same edits at the parent's
    initial state and ran fresh from step 0. Used for the design-diagnostic
    "would fixing this at initial design achieve the same outcome?" question.
    See feature_ideas.md → Fork-edit-replay augmentations → feature 2.
    """

    parent_run_id: str
    fork_step: int
    edits: dict[str, Any]
    fork_point_state: dict         # state_after[fork_step] BEFORE edits
    edited_state_at_fork: dict     # state_after[fork_step] AFTER edits (merged)
    fork_branch: ForkBranch = field(default_factory=lambda: ForkBranch(initial_state={}))
    top_edited_branch: ForkBranch | None = None
    # Cost / provenance telemetry for the fork call itself.
    duration_s: float = 0.0
    # Why the fork ran the way it did — e.g. "checkpoint fork: resumed step 2
    # on thread drift-…" or "rerun-from-state: graph has no checkpointer".
    execution_note: str = ""

    @property
    def n_coordination_findings(self) -> int:
        n = len(self.fork_branch.coordination_findings)
        if self.top_edited_branch:
            n += len(self.top_edited_branch.coordination_findings)
        return n

    @property
    def path_dependence(self) -> dict[str, Any] | None:
        """The design-diagnostic question made concrete: does applying the
        same edit at initial design converge to the same outcome as
        applying it at the fork point, or does the system carry real
        path-dependent state?

        None when `also_apply_at_initial` wasn't requested (no top branch
        to compare against), or when either branch crashed — a crash
        isn't a state-field diff, it's a different kind of divergence
        this comparison isn't trying to characterize.

        `is_path_dependent=True` means the two branches' final states
        differ after the same edit was applied at two different points —
        real signal the system isn't initial-state-pure (Camp A design
        philosophy: if a bug can ONLY be fixed by editing mid-run state,
        not by editing the initial design, the topology itself has
        state that accumulates path-dependent facts the initial state
        can't express). See feature_ideas.md → Fork-edit-replay
        augmentations → "Initial-state vs fork-state consistency check."
        """
        if self.top_edited_branch is None:
            return None
        if self.fork_branch.crashed or self.top_edited_branch.crashed:
            return None
        _, _, details = _diff_states(
            self.fork_branch.final_state, self.top_edited_branch.final_state,
        )
        return {
            "is_path_dependent": bool(details),
            "diverged_fields": [
                {
                    "name": d.name,
                    "fork_branch_value": d.baseline_value,
                    "top_edited_branch_value": d.perturbed_value,
                    "summary": d.summary,
                }
                for d in details
            ],
        }


def _deep_merge_edits(base: dict, edits: dict) -> dict:
    """Sparse deep-merge — `edits` overwrites at leaf level; dict values recurse.

    - Non-dict values in `edits` replace whatever's at the same key in `base`.
    - Dict values recurse (so `edits={"a": {"b": 1}}` doesn't wipe base's other
      "a" subkeys).
    - Lists are REPLACED wholesale — supporting list splice syntax needs the
      JSON-patch surface we deliberately deferred.

    Returns a new dict (deepcopy of `base` mutated in place). Does not modify
    the input.
    """
    out = deepcopy(base)

    def _apply(dst: dict, src: dict) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _apply(dst[k], v)
            else:
                dst[k] = deepcopy(v)

    if isinstance(edits, dict):
        _apply(out, edits)
    return out


async def _run_one_fork_branch(
    graph: Any,
    initial_state: dict,
    *,
    run_coordination_detectors: bool,
    coordination_roles: dict[str, str] | None,
    max_steps: int | None = None,
    config: dict | None = None,
    resume: bool = False,
) -> ForkBranch:
    """Run the graph once, capture trace + coord findings.

    `resume=True` means `config` names a checkpoint to continue from:
    LangGraph is invoked with input None and `initial_state` is only the
    (already-edited) checkpoint state used to seed the trace builder.
    Otherwise `initial_state` is fed as a fresh input."""
    t0 = time.perf_counter()
    sink: list[dict] = []
    mode = FORK_MODE_CHECKPOINT if resume else FORK_MODE_RERUN
    tid = (config or {}).get("configurable", {}).get("thread_id", "")
    try:
        final_state, trace, truncated = await _stream_or_invoke(
            graph, None if resume else initial_state, max_steps=max_steps,
            trace_sink=sink, config=config, resume_state=initial_state if resume else None,
        )
        duration = time.perf_counter() - t0
        coord: list[dict] = []
        if run_coordination_detectors and trace:
            findings = _run_library_on_trace(
                trace,
                initial_state=initial_state,
                baseline_state=final_state,
                roles_by_agent=coordination_roles,
            )
            coord = [f.model_dump(mode="json") for f in findings]
        return ForkBranch(
            initial_state=deepcopy(initial_state),
            trace=trace,
            final_state=final_state,
            duration_s=duration,
            coordination_findings=coord,
            truncated=truncated,
            execution_mode=mode,
            thread_id=tid,
        )
    except Exception as exc:  # noqa: BLE001
        return ForkBranch(
            initial_state=deepcopy(initial_state),
            trace=sink,  # partial trace up to the crash — not discarded
            final_state=None,
            crashed=True,
            error=str(exc)[:400],
            error_type=type(exc).__name__,
            crash_class=_classify_crash(type(exc).__name__, str(exc)),
            duration_s=time.perf_counter() - t0,
            execution_mode=mode,
            thread_id=tid,
        )


def _root_rank(trace: list[dict], fork_step: int) -> int | None:
    """How many ROOT-namespace super-steps the trace has up to and including
    `fork_step` (1-based). None if the record at `fork_step` is itself a
    subgraph step — LangGraph can't time-travel to a point inside a
    subgraph from the parent, so there is no root checkpoint to fork."""
    if trace[fork_step - 1].get("namespace"):
        return None
    return sum(1 for rec in trace[:fork_step] if not rec.get("namespace"))


def _find_checkpoint(graph: Any, thread_id: str, root_rank: int) -> Any:
    """The StateSnapshot on `thread_id` taken after `root_rank` root super-
    steps (LangGraph numbers them `metadata["step"]`, with -1 = input)."""
    history = getattr(graph, "get_state_history", None)
    if not callable(history):
        return None
    # History is newest-first and holds every branch on the thread: once a
    # fork has resumed, its own checkpoints carry the same `step` numbers as
    # the original run's later steps. The ORIGINAL run's checkpoint is the
    # oldest with that step, so keep the last match rather than the first.
    found = None
    for snap in history(_thread_config(thread_id)):
        meta = getattr(snap, "metadata", None) or {}
        if meta.get("step") == root_rank and not (snap.config.get("configurable") or {}).get("checkpoint_ns"):
            found = snap
    return found


def _capture_root_checkpoints(graph: Any, thread_id: str) -> dict[int, Any]:
    """`{root_rank: StateSnapshot}` for every root-namespace checkpoint on
    `thread_id` right now — taken straight after the baseline run, before
    any fork writes a branch to the thread."""
    history = getattr(graph, "get_state_history", None)
    if not callable(history) or not thread_id:
        return {}
    out: dict[int, Any] = {}
    try:
        for snap in history(_thread_config(thread_id)):
            meta = getattr(snap, "metadata", None) or {}
            step = meta.get("step")
            if isinstance(step, int) and step >= 0 and not (snap.config.get("configurable") or {}).get("checkpoint_ns"):
                out[step] = snap   # newest-first: the last write wins = the oldest = the baseline's
    except Exception:  # noqa: BLE001 — a saver that cannot list; mid-run falls back
        return {}
    return out


_PROCESS_CHECKPOINTER: Any = None


def _process_checkpointer() -> Any:
    """One in-memory saver shared by every graph drift runs in this process
    (the web server reuses it), so forks resume real checkpoints."""
    global _PROCESS_CHECKPOINTER
    if _PROCESS_CHECKPOINTER is None:
        try:
            from langgraph.checkpoint.memory import MemorySaver
        except Exception:  # noqa: BLE001
            return None
        _PROCESS_CHECKPOINTER = MemorySaver()
    return _PROCESS_CHECKPOINTER


def ensure_checkpointer(graph: Any) -> Any:
    """Give a compiled LangGraph graph without a checkpointer the process-wide
    one, in place. Leaves anything else — a graph compiled with its own
    saver, a hand-rolled stub, a plain callable — untouched. Returns graph."""
    if _checkpointer_of(graph) is not None:
        return graph
    looks_compiled = (
        hasattr(graph, "checkpointer")
        and callable(getattr(graph, "get_state_history", None))
        and callable(getattr(graph, "update_state", None))
    )
    if not looks_compiled:
        return graph
    saver = _process_checkpointer()
    if saver is None:
        return graph
    try:
        graph.checkpointer = saver
    except Exception:  # noqa: BLE001 — a frozen/odd Pregel subclass; keep the rerun path
        return graph
    return graph


def _fork_rank_for(planner_t: int, n_root: int) -> int:
    """Map the planner's step (2..horizon-1 over a synthetic horizon) onto
    a baseline root step to fork after, in [1, n_root - 1] — never the last
    step, so there is something left to resume."""
    span = max(1, _PLANNER_HORIZON - 3)
    frac = min(1.0, max(0.0, (planner_t - 2) / span))
    rank = 1 + round(frac * max(0, n_root - 2))
    return max(1, min(n_root - 1, rank))


def _has_pending_tool_calls(snap: Any) -> bool:
    """True when the checkpoint's last message is an AI message with tool
    calls — i.e. a ToolNode is about to run. Appending a chaos message
    there would both orphan that call (an AI tool call with no result is
    an invalid sequence for every chat API) and re-route the graph on the
    appended message instead; forking after the tool step is the right
    place for a mid-run message perturbation."""
    from drift.chaos import messages as _m

    values = dict(getattr(snap, "values", None) or {})
    for v in values.values():
        if _m.is_messages_list(v) and v:
            last = v[-1]
            return _m.message_role(last) == "ai" and bool(_m.message_tool_calls(last))
    return False


def _choose_fork_rank(preferred: int, root_snaps: dict[int, Any], n_root: int) -> int:
    """The root step to fork after: `preferred` unless its checkpoint has a
    pending tool call, in which case the nearest step (later first) whose
    checkpoint does not. Falls back to `preferred` when none qualifies."""
    hi = max(1, n_root - 1)
    order = sorted(range(1, hi + 1), key=lambda r: (abs(r - preferred), -r))
    for r in order:
        snap = root_snaps.get(r)
        if snap is not None and not _has_pending_tool_calls(snap):
            return r
    return preferred


def _world_fields(world: World, keys: Iterable[str]) -> dict:
    """The live (post-mutation) values of `keys` on the world's state — the
    objects themselves, not a model_dump, so langchain messages keep their
    class and id."""
    return {k: getattr(world.state, k) for k in keys if hasattr(world.state, k)}


def _checkpoint_edits(before: dict, after: dict) -> dict:
    """The `update_state` payload that turns checkpoint values `before` into
    `after`. Plain fields are written whole (replace channels). A messages
    list goes through add_messages, so it is expressed as a delta: removed
    ids become RemoveMessage, changed messages keep their id (replace),
    appended ones are new."""
    from drift.chaos import messages as _m

    edits: dict = {}
    for k, new in after.items():
        old = before.get(k)
        if k == "timestep":
            continue
        if isinstance(new, list) and (_m.is_messages_list(old) or _m.is_messages_list(new)):
            delta = _messages_delta(list(old or []), list(new))
            if delta:
                edits[k] = delta
            continue
        if old != new:
            edits[k] = new
    return edits


def _messages_delta(old: list, new: list) -> list:
    from drift.chaos import messages as _m

    def _id(m: Any) -> Any:
        if isinstance(m, dict):
            return m.get("id")
        return getattr(m, "id", None)

    def _sig(m: Any) -> tuple:
        return (_m.message_role(m), _m.message_text(m), repr(_m.message_tool_calls(m)))

    old_ids = [_id(m) for m in old]
    new_ids = [_id(m) for m in new]
    out: list = []
    removed = [i for i in old_ids if i is not None and i not in new_ids]
    if removed:
        try:
            from langchain_core.messages import RemoveMessage
            out.extend(RemoveMessage(id=i) for i in removed)
        except Exception:  # noqa: BLE001 — no langchain: cannot express a removal
            pass
    for m in new:
        mid = _id(m)
        if mid is None or mid not in old_ids:
            out.append(m)
        else:
            orig = old[old_ids.index(mid)]
            if orig is not m and _sig(orig) != _sig(m):
                out.append(m)
    return out


def _write_fork_edits(graph: Any, fork: dict) -> dict:
    """`update_state` the fork's edits onto its checkpoint, AS the node that
    produced that checkpoint. Left to infer `as_node`, LangGraph can pick
    the wrong writer and then evaluate the wrong node's outgoing edges —
    on `create_react_agent`, an injected tool result written "as agent"
    routes to END (no tool calls on the last message) and the resumed run
    is empty. Written as `tools`, the tools→agent edge fires and the agent
    sees the poisoned result. Falls back to inference if the name is not a
    node the graph accepts."""
    snap, edits, as_node = fork["snap"], fork["edits"], fork.get("as_node")
    if not edits:
        return snap.config
    if as_node:
        try:
            return graph.update_state(snap.config, edits, as_node=as_node)
        except (ValueError, KeyError, TypeError):
            pass
    return graph.update_state(snap.config, edits)


async def _run_one_resume(
    graph: Any, config: dict, resume_state: dict,
) -> tuple[dict | None, list[dict], str, str, float]:
    """Like `_run_one`, but continue a checkpoint (input None) — `config`
    names it; `resume_state` seeds the trace builder."""
    t0 = time.perf_counter()
    sink: list[dict] = []
    try:
        final, trace, _truncated = await _stream_or_invoke(
            graph, None, trace_sink=sink, config=config, resume_state=resume_state,
        )
        return (final, trace, "", "", time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001
        return (None, sink, type(exc).__name__, str(exc), time.perf_counter() - t0)


async def drift_test_fork_async(
    *,
    graph: Any,
    parent_result: AdapterResult,
    fork_step: int,
    edits: dict[str, Any],
    also_apply_at_initial: bool = False,
    run_coordination_detectors: bool = True,
    coordination_roles: dict[str, str] | None = None,
    parent_run_id: str | None = None,
    max_steps: int | None = None,
) -> ForkResult:
    """Fork a completed adapter run at `fork_step`, apply `edits`, re-run forward.

    Two execution models, chosen automatically (see `ForkResult.execution_note`
    and `ForkBranch.execution_mode`):

    * **checkpoint** — when the graph was compiled with a checkpointer and
      `parent_result` came from `drift_test` on this same graph object (so
      `baseline.thread_id` is set and the checkpoints are still in the
      saver). Drift finds the checkpoint after the fork step's root super-
      step, applies `edits` with `graph.update_state` (edits pass through the
      channel reducers — editing a `messages` channel APPENDS), and resumes
      with input None. This is real LangGraph time travel: pending tasks,
      tool-call resume position and all.
    * **rerun_from_state** — otherwise. Take the fork step's `state_after`,
      deep-merge `edits`, invoke the graph fresh with that as initial state.
      The branch "starts" from the fork-point state rather than resuming a
      paused execution; fine for most graphs, approximate for ones that
      depend on tool-call resume position.

    If `also_apply_at_initial=True`, we also invoke the graph with the same
    edits applied to the parent's original initial state. The two branches
    are returned together so callers can render the design-diagnostic compare:
    does editing at initial design converge to the same outcome as editing at
    fork_step, or does the state carry path-dependence?

    `max_steps`: bounded counterfactual replay — stop each branch after this
    many super-steps instead of running to completion. Cost control (a full
    replay can be an expensive LLM-driven run; "what happens 2 steps after
    my edit" is a much cheaper question) and focus (you often only care
    about the immediate aftermath of an edit, not the whole rest of the
    run). Only takes effect on graphs that support streaming — see
    `_stream_or_invoke`'s docstring. Each returned branch's `truncated`
    flag is True when this actually cut it short; `final_state` in that
    case is the state AT that step, not the graph's true completion state.

    Raises ValueError on out-of-range fork_step or crashed parent baseline.
    """
    if parent_result.baseline.crashed:
        raise ValueError(
            "cannot fork a run whose baseline crashed (no valid trace to fork from)"
        )
    trace = parent_result.baseline.trace
    if not trace:
        raise ValueError(
            "parent run has an empty baseline trace — the graph produced no "
            "streamable super-steps, so there's no fork point"
        )
    if fork_step < 1 or fork_step > len(trace):
        raise ValueError(
            f"fork_step={fork_step} out of range [1, {len(trace)}] "
            f"(parent baseline has {len(trace)} steps)"
        )

    t0 = time.perf_counter()
    fork_point_state = deepcopy(trace[fork_step - 1].get("state_after") or {})
    edited_state_at_fork = _deep_merge_edits(fork_point_state, edits or {})

    # Checkpoint path: the graph has a checkpointer AND the parent baseline
    # ran on a thread we know AND the fork step is a root super-step AND that
    # checkpoint is still in the saver. Then this is a real time-travel fork:
    # update_state on the parent's checkpoint (edits go THROUGH the channel
    # reducers — an edit to a `messages` channel appends), resume with input
    # None. Anything missing -> the v1 rerun-from-state approximation, and
    # `execution_note` says which and why.
    fork_branch: ForkBranch | None = None
    note = ""
    thread_id = getattr(parent_result.baseline, "thread_id", "") or ""
    checkpointer = _checkpointer_of(graph)
    if checkpointer is None:
        note = "rerun-from-state: graph has no checkpointer"
    elif not thread_id:
        note = "rerun-from-state: parent baseline has no thread_id (run it with this graph object to enable checkpoint forks)"
    elif not callable(getattr(graph, "update_state", None)):
        note = "rerun-from-state: graph exposes no update_state"
    else:
        rank = _root_rank(trace, fork_step)
        if rank is None:
            note = f"rerun-from-state: step {fork_step} is inside a subgraph; LangGraph can't fork there"
        else:
            snap = _find_checkpoint(graph, thread_id, rank)
            if snap is None:
                note = (
                    f"rerun-from-state: no checkpoint for root step {rank} on thread {thread_id!r} "
                    "(different graph object / checkpointer instance, or history evicted)"
                )
            else:
                try:
                    cfg = graph.update_state(snap.config, edits) if edits else snap.config
                    resumed_state = _normalize_messages(dict(graph.get_state(cfg).values or {}))
                except Exception as exc:  # noqa: BLE001
                    note = f"rerun-from-state: update_state failed ({type(exc).__name__}: {exc})"
                else:
                    edited_state_at_fork = resumed_state
                    note = f"checkpoint fork: resumed root step {rank} on thread {thread_id!r} after update_state"
                    fork_branch = await _run_one_fork_branch(
                        graph,
                        resumed_state,
                        run_coordination_detectors=run_coordination_detectors,
                        coordination_roles=coordination_roles,
                        max_steps=max_steps,
                        config=cfg,
                        resume=True,
                    )

    if fork_branch is None:
        fork_branch = await _run_one_fork_branch(
            graph,
            edited_state_at_fork,
            run_coordination_detectors=run_coordination_detectors,
            coordination_roles=coordination_roles,
            max_steps=max_steps,
            config=_thread_config(f"drift-fork-{uuid.uuid4().hex[:10]}") if checkpointer else None,
        )

    top_branch: ForkBranch | None = None
    if also_apply_at_initial:
        edited_initial = _deep_merge_edits(
            parent_result.baseline.initial_state or {},
            edits or {},
        )
        top_branch = await _run_one_fork_branch(
            graph,
            edited_initial,
            run_coordination_detectors=run_coordination_detectors,
            coordination_roles=coordination_roles,
            max_steps=max_steps,
            config=_thread_config(f"drift-fork-top-{uuid.uuid4().hex[:10]}") if checkpointer else None,
        )

    return ForkResult(
        parent_run_id=parent_run_id or "",
        fork_step=fork_step,
        edits=deepcopy(edits or {}),
        fork_point_state=fork_point_state,
        edited_state_at_fork=edited_state_at_fork,
        fork_branch=fork_branch,
        top_edited_branch=top_branch,
        duration_s=time.perf_counter() - t0,
        execution_note=note,
    )


def drift_test_fork(
    *,
    graph: Any,
    parent_result: AdapterResult,
    fork_step: int,
    edits: dict[str, Any],
    also_apply_at_initial: bool = False,
    run_coordination_detectors: bool = True,
    coordination_roles: dict[str, str] | None = None,
    parent_run_id: str | None = None,
    max_steps: int | None = None,
) -> ForkResult:
    """Synchronous wrapper around `drift_test_fork_async`.

    See `drift_test_fork_async` for arg semantics.
    """
    return asyncio.run(
        drift_test_fork_async(
            graph=graph,
            parent_result=parent_result,
            fork_step=fork_step,
            edits=edits,
            also_apply_at_initial=also_apply_at_initial,
            run_coordination_detectors=run_coordination_detectors,
            coordination_roles=coordination_roles,
            parent_run_id=parent_run_id,
            max_steps=max_steps,
        )
    )


__all__ = [
    "AdapterResult",
    "BaselineResult",
    "FORK_MODE_CHECKPOINT",
    "FORK_MODE_RERUN",
    "ForkBranch",
    "ForkResult",
    "PerturbationResult",
    "FieldDivergence",
    "FieldNoiseBand",
    "drift_test",
    "drift_test_async",
    "drift_test_fork",
    "drift_test_fork_async",
]
