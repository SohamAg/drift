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

from collections import Counter
from difflib import SequenceMatcher

from drift.agents.base import Action
from drift.chaos.engine import _normalize_intensity, plan_auto_chaos
from drift.chaos.fuzzer import discover_field_patterns
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
DEFAULT_BASELINE_ROLLOUTS = 1            # >1 enables noise-floor measurement
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
    final_state: dict | None = None
    crashed: bool = False
    error: str = ""
    error_type: str = ""
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
    coordination_findings: list[dict] = field(default_factory=list)


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
    noise_band: dict[str, FieldNoiseBand] = field(default_factory=dict)
    judge_calls_used: int = 0        # tier-3 calls actually fired (cost telemetry)
    judge_calls_budget: int = DEFAULT_MAX_JUDGE_CALLS

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
            f"  diverged        : {self.n_diverged}",
            f"  unchanged       : {self.n_unchanged}",
        ]
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

    # Text stats.
    if all(isinstance(v, str) for v in values) and len(values) >= 2:
        sims: list[float] = []
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                sims.append(_text_similarity(values[i], values[j]))
        if sims:
            band.text_min_similarity = min(sims)
            band.text_mean_similarity = sum(sims) / len(sims)

    return band


async def _measure_noise_band(
    graph: Any,
    initial_state: dict,
    state_factory: Callable[[], dict] | None,
    n_rollouts: int,
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
    for _ in range(n_rollouts):
        fresh = (
            deepcopy(state_factory()) if state_factory else deepcopy(initial_state)
        )
        final, _trace, etype, _err, _dur = await _run_one(graph, fresh)
        # If any rollout crashes we skip it for noise analysis but keep its
        # slot so callers can see counts didn't add up.
        finals.append(None if etype else final)

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
    if (
        band.text_min_similarity is not None
        and isinstance(perturbed_value, str)
    ):
        text_baselines = [v for v in band.distinct_values if isinstance(v, str)]
        if not text_baselines:
            return (False, None)
        best_sim = max(_text_similarity(perturbed_value, b) for b in text_baselines)
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


def _run_coordination_library(
    trace: list[dict],
    initial_state: dict | None,
    baseline_state: dict | None,
    *,
    enabled: bool = True,
    roles_by_agent: dict[str, str] | None = None,
) -> list[dict]:
    """Run the curated coordination-failure detector library over one trace.

    Free + deterministic — no LLM cost. Returns FailureRecord dicts. Empty
    when disabled, when the trace is empty (graph doesn't stream or crashed
    at step 0), or when no detector matched.
    """
    if not enabled or not trace:
        return []
    findings = _run_library_on_trace(
        trace,
        initial_state=initial_state,
        baseline_state=baseline_state,
        roles_by_agent=roles_by_agent,
    )
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
    divergence_mode: str = DEFAULT_DIVERGENCE_MODE,
    baseline_rollouts: int = DEFAULT_BASELINE_ROLLOUTS,
    max_judge_calls: int = DEFAULT_MAX_JUDGE_CALLS,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    run_coordination_detectors: bool = True,
    coordination_roles: dict[str, str] | None = None,
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

    # Baseline run uses a fresh copy of the user's state so the graph can't
    # mutate the input dict and leak across perturbation runs.
    # In tiered mode with baseline_rollouts > 1, we measure natural variance
    # FIRST (N rollouts), then use the first successful one as THE baseline.
    # Otherwise we just run baseline once like before.
    noise_band: dict[str, FieldNoiseBand] = {}
    if divergence_mode == "tiered" and baseline_rollouts > 1:
        noise_band, finals = await _measure_noise_band(
            graph, initial_state, state_factory, baseline_rollouts,
        )
        # Pick the first successful rollout as THE baseline. Time/trace/etc.
        # are re-derived from a one-shot run below so we have consistent
        # trace data — the noise band is derived from final-state variance,
        # not from per-step traces, so we don't need to keep all N traces.
        baseline_input = (
            deepcopy(state_factory()) if state_factory else deepcopy(initial_state)
        )
        bfinal, btrace, betype, berr, btime = await _run_one(graph, baseline_input)
    else:
        baseline_input = (
            deepcopy(state_factory()) if state_factory else deepcopy(initial_state)
        )
        bfinal, btrace, betype, berr, btime = await _run_one(graph, baseline_input)

    baseline_findings: list[dict] = []
    if judge_llm is not None and btrace:
        baseline_findings = await _run_judge_on_trace(
            judge_llm, guidelines, btrace, state_cls,
        )
    baseline_coord = _run_coordination_library(
        btrace, initial_state, bfinal,
        enabled=run_coordination_detectors,
        roles_by_agent=coordination_roles,
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
        coordination_findings=baseline_coord,
    )

    judge_calls_used = 0
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

        # Divergence detection: dispatch through the cost cascade.
        # `divfiltered` carries tier-2/3 candidates the cascade dropped —
        # preserved so the UI can audit UNCHANGED verdicts.
        divfiltered: list[FieldDivergence] = []
        if divergence_mode == "off":
            diverged, divsum, divdetails = (False, "", [])
        elif divergence_mode == "tiered":
            remaining = max(0, max_judge_calls - judge_calls_used)
            diverged, divsum, divdetails, divfiltered, used = await _diff_states_tiered(
                baseline.final_state, final, noise_band,
                judge_llm if remaining > 0 else None,
                similarity_threshold, remaining,
            )
            judge_calls_used += used
        else:  # "exact"
            diverged, divsum, divdetails = _diff_states(baseline.final_state, final)

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
            roles_by_agent=coordination_roles,
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
                divergence_details=divdetails,
                filtered_divergences=divfiltered,
                coordination_findings=pert_coord,
            )
        )

    return AdapterResult(
        baseline=baseline,
        perturbations=perturbations,
        intensity=level,
        patterns_total=len(all_specs),
        divergence_mode=divergence_mode,
        baseline_rollouts=baseline_rollouts,
        noise_band=noise_band,
        judge_calls_used=judge_calls_used,
        judge_calls_budget=max_judge_calls,
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
    max_judge_calls: int = DEFAULT_MAX_JUDGE_CALLS,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    run_coordination_detectors: bool = True,
    coordination_roles: dict[str, str] | None = None,
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
        max_judge_calls: hard ceiling on tier-3 judge calls across all
            perturbations in this drift_test call. Default 10. Tier 0-2
            are free; only tier 3 costs LLM tokens for divergence
            equivalence checks.
        similarity_threshold: tier-2 lexical similarity cutoff for text
            fields, 0..1. Default 0.85. Effective threshold is
            max(this, measured baseline-vs-baseline floor) so text fields
            with naturally high variance get a tighter band automatically.

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
            baseline_rollouts + 1 + len(scheduled_perturbations) graph runs
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
            max_judge_calls=max_judge_calls,
            similarity_threshold=similarity_threshold,
            run_coordination_detectors=run_coordination_detectors,
            coordination_roles=coordination_roles,
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
    duration_s: float = 0.0
    coordination_findings: list[dict] = field(default_factory=list)


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

    @property
    def n_coordination_findings(self) -> int:
        n = len(self.fork_branch.coordination_findings)
        if self.top_edited_branch:
            n += len(self.top_edited_branch.coordination_findings)
        return n


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
) -> ForkBranch:
    """Run the graph once from `initial_state`, capture trace + coord findings."""
    t0 = time.perf_counter()
    try:
        final_state, trace = await _stream_or_invoke(graph, initial_state)
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
        )
    except Exception as exc:  # noqa: BLE001
        return ForkBranch(
            initial_state=deepcopy(initial_state),
            trace=[],
            final_state=None,
            crashed=True,
            error=str(exc)[:400],
            error_type=type(exc).__name__,
            duration_s=time.perf_counter() - t0,
        )


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
) -> ForkResult:
    """Fork a completed adapter run at `fork_step`, apply `edits`, re-run forward.

    v1 execution model: rerun-from-state. We take the state_after of the fork
    step in the parent's baseline trace, deep-merge `edits` into it, and invoke
    the graph fresh with that as the initial state. This means the fork branch
    "starts" from the fork-point state — not resuming a paused execution. Good
    enough for the common developer case; graphs that depend on tool-call
    resume position will need the v2 checkpointer path.

    If `also_apply_at_initial=True`, we also invoke the graph with the same
    edits applied to the parent's original initial state. The two branches
    are returned together so callers can render the design-diagnostic compare:
    does editing at initial design converge to the same outcome as editing at
    fork_step, or does the state carry path-dependence?

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

    fork_branch = await _run_one_fork_branch(
        graph,
        edited_state_at_fork,
        run_coordination_detectors=run_coordination_detectors,
        coordination_roles=coordination_roles,
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
        )
    )


__all__ = [
    "AdapterResult",
    "BaselineResult",
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
