"""Auto-chaos scheduling.

Inputs:  a WorldState, a step count, an intensity setting, a seed.
Output:  a list of (timestep, Event) tuples ready to merge with the
         user-supplied `events` list and pass into drift's runner.

The intensity setting maps to a per-step injection probability:
  - off       -> 0%
  - light     -> 8%
  - moderate  -> 18%
  - aggressive-> 35%

At each step the engine rolls one die per the probability; on hit, it picks
a chaos spec from the catalog (weighted by spec type — see _weighted_pick)
and builds a fresh event targeted at the right field.

Scheduling avoids step 1 (gives agents one observation cycle to bind
expectations) and avoids the final step (so any failures the chaos surfaces
have at least one detector tick to catch them).
"""
from __future__ import annotations

import random
from typing import Iterable

from drift.chaos.fuzzer import ChaosSpec, discover_field_patterns
from drift.chaos.patterns import AutoChaosEvent
from drift.events.base import Event
from drift.world import WorldState

# Per-step probability of injecting an auto-chaos event. Tuned so a 30-step
# run sees ~5 (moderate) / ~10 (aggressive) injections — enough to surface
# interesting failures without drowning the agents in chaos.
INTENSITY_FREQUENCY: dict[str, float] = {
    "off": 0.0,
    "light": 0.08,
    "moderate": 0.18,
    "aggressive": 0.35,
}


def _normalize_intensity(value: str | bool | None) -> str:
    if value is None or value is False:
        return "off"
    if value is True:
        return "moderate"
    if isinstance(value, str):
        v = value.strip().lower()
        if v in INTENSITY_FREQUENCY:
            return v
        raise ValueError(
            f"unknown auto_chaos intensity {value!r}; "
            f"expected one of {sorted(INTENSITY_FREQUENCY)}"
        )
    raise TypeError(
        f"auto_chaos must be bool, str, or None; got {type(value).__name__}"
    )


def _exclude_match(pattern_name: str, exclude: Iterable[str]) -> bool:
    """Match exclusion entries as substrings against pattern_name.

    `flip_bool` excludes every flip_bool[*]; `flip_bool[approved]` excludes
    only that one. Substring match keeps the user API forgiving.
    """
    return any(tok in pattern_name for tok in exclude)


def plan_auto_chaos(
    *,
    state: WorldState,
    steps: int,
    intensity: str | bool | None,
    seed: int = 0,
    exclude: Iterable[str] | None = None,
) -> list[tuple[int, Event]]:
    """Build the auto-chaos schedule for one run.

    See `auto_chaos_events` for the parameter contract — this is the
    underlying implementation that returns concrete scheduled events.
    Returns an empty list when intensity is off, the state has no fuzzable
    fields, or all candidates are excluded.
    """
    level = _normalize_intensity(intensity)
    freq = INTENSITY_FREQUENCY[level]
    if freq <= 0.0 or steps < 2:
        return []

    exclude_set = list(exclude or [])
    catalog = [
        spec for spec in discover_field_patterns(state, seed=seed)
        if not _exclude_match(spec.pattern_name, exclude_set)
    ]
    if not catalog:
        return []

    rng = random.Random(seed ^ 0xC4A05)  # decorrelate from agent RNGs
    scheduled: list[tuple[int, Event]] = []

    # Skip step 1 (let agents see the initial state) and the very last step
    # (give detectors a chance to react).
    first_step = 2
    last_step = max(first_step, steps - 1)
    for t in range(first_step, last_step + 1):
        if rng.random() >= freq:
            continue
        spec = rng.choice(catalog)
        event = spec.build()
        scheduled.append((t, event))

    return scheduled


def auto_chaos_events(
    state: WorldState,
    steps: int,
    intensity: str | bool | None = "moderate",
    seed: int = 0,
    exclude: Iterable[str] | None = None,
) -> list[tuple[int, Event]]:
    """Public helper: get the auto-chaos schedule for a state + step count.

    Equivalent to `plan_auto_chaos(state=..., steps=..., intensity=...)` —
    positional form for ad-hoc use.

    Args:
        state: a drift.WorldState instance.
        steps: total simulation steps in the planned run.
        intensity: "off" | "light" | "moderate" | "aggressive"; True is an
            alias for "moderate"; False / None for "off".
        seed: RNG seed for both pattern selection and per-event randomness.
        exclude: iterable of pattern substrings to skip. "flip_bool" excludes
            every flip_bool[<field>] event; "flip_bool[approved]" excludes
            only the one targeting `approved`.

    Returns:
        A list of (timestep, Event) tuples; pass into drift.run via the
        existing `events=` kwarg, or let drift.run handle the auto_chaos
        kwarg directly.
    """
    return plan_auto_chaos(
        state=state,
        steps=steps,
        intensity=intensity,
        seed=seed,
        exclude=exclude,
    )


__all__ = [
    "INTENSITY_FREQUENCY",
    "auto_chaos_events",
    "plan_auto_chaos",
]
