"""Auto-chaos event patterns.

Each pattern is a drift Event subclass that mutates one named field of the
world state. The mutation is type-appropriate: a bool gets flipped, a dict
gets cleared / corrupted, a numeric field gets pushed to a boundary, etc.

Patterns target *any* WorldState subclass — they read and write through the
generic attribute API (`getattr` / `setattr`) plus `world.record_change` for
the audit trail. They don't depend on the shipped support / code_review /
ops fields, so they apply to user-defined WorldState subclasses unchanged.

Names are prefixed with `AutoChaos.` so the result can filter auto-injected
events out of user-supplied ones (see `RunResult.auto_chaos_injected`).
"""
from __future__ import annotations

import random
from typing import Any

from drift.events.base import Event, EventRecord
from drift.world import World

AUTO_CHAOS_PREFIX = "AutoChaos."


class AutoChaosEvent(Event):
    """Marker base for every auto-generated chaos event.

    Subclasses set `self.field` and `self.name` in __init__ and implement
    `_mutate(world)` which returns a one-line summary string describing the
    mutation (or None if the field wasn't present / wasn't the expected type
    — in which case the event becomes an auditable no-op).
    """

    pattern: str = "auto_chaos"

    def __init__(self, field: str, *, seed: int | None = None) -> None:
        super().__init__()
        self.field = field
        self.name = f"{AUTO_CHAOS_PREFIX}{self.pattern}[{field}]"
        self._rng = random.Random(seed)

    def apply(self, world: World) -> EventRecord:
        summary = self._mutate(world)
        if summary is None:
            summary = f"{self.field} not present or unexpected type — no-op"
        else:
            world.record_change("event", self.event_id, f"auto-chaos {self.field}: {summary}")
        return EventRecord(
            event_id=self.event_id,
            timestep=world.state.timestep,
            name=self.name,
            summary=summary,
        )

    def _mutate(self, world: World) -> str | None:  # pragma: no cover - abstract
        raise NotImplementedError


def _get(world: World, field: str) -> Any:
    return getattr(world.state, field, None)


def _set(world: World, field: str, value: Any) -> None:
    setattr(world.state, field, value)


# ---- bool ----------------------------------------------------------------


class FlipBoolField(AutoChaosEvent):
    pattern = "flip_bool"

    def _mutate(self, world: World) -> str | None:
        v = _get(world, self.field)
        if not isinstance(v, bool):
            return None
        _set(world, self.field, not v)
        return f"{v} -> {not v}"


# ---- numeric -------------------------------------------------------------


class BoundaryNumericField(AutoChaosEvent):
    """Push a numeric field to a boundary value (0, -1, very large, NaN-like).

    Caps are intentionally generic because we don't know the field's intended
    bounds. The point is to surface "what happens if this counter resets" or
    "what if this measurement blows up", not to honor any specific range.
    """

    pattern = "boundary_numeric"

    _INT_TARGETS = (0, -1, 1, 999_999_999)
    _FLOAT_TARGETS = (0.0, -1.0, 1.0, 1e9)

    def _mutate(self, world: World) -> str | None:
        v = _get(world, self.field)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            # bool subclasses int — exclude
            return None
        if isinstance(v, int):
            choice = self._rng.choice(self._INT_TARGETS)
        else:
            choice = self._rng.choice(self._FLOAT_TARGETS)
        _set(world, self.field, choice)
        return f"{v} -> {choice}"


# ---- str -----------------------------------------------------------------


class CorruptStringField(AutoChaosEvent):
    """Replace a string field with a degenerate value: empty, garbage, or
    a stale-looking copy of itself."""

    pattern = "corrupt_string"

    _GARBAGE = (
        "",
        "<<chaos>>",
        "null",
        "{}",
        "DELETED",
    )

    def _mutate(self, world: World) -> str | None:
        v = _get(world, self.field)
        if not isinstance(v, str):
            return None
        choice = self._rng.choice(self._GARBAGE)
        _set(world, self.field, choice)
        return f"{v!r} -> {choice!r}"


# ---- dict ----------------------------------------------------------------


class ClearDictField(AutoChaosEvent):
    """Empty a dict mid-run. Models a queue / open-cases / state-table
    suddenly going empty (cleared by another system, GC'd, etc.)."""

    pattern = "clear_dict"

    def _mutate(self, world: World) -> str | None:
        v = _get(world, self.field)
        if not isinstance(v, dict):
            return None
        size = len(v)
        if size == 0:
            return f"already empty (size 0)"
        _set(world, self.field, {})
        return f"cleared {size} entries"


class RemoveDictKey(AutoChaosEvent):
    """Remove a random key from a dict. Models a tracked entity (case, PR,
    incident, session) silently disappearing mid-run."""

    pattern = "remove_dict_key"

    def _mutate(self, world: World) -> str | None:
        v = _get(world, self.field)
        if not isinstance(v, dict) or not v:
            return None
        keys = list(v.keys())
        target = self._rng.choice(keys)
        # Mutate in place so any references to the dict still observe the change.
        del v[target]
        return f"removed key {target!r} (now {len(v)} entries)"


class InjectFakeDictKey(AutoChaosEvent):
    """Insert a sentinel key into a dict. Models bad data from an upstream
    system — agents may treat it as real."""

    pattern = "inject_fake_dict_key"

    def _mutate(self, world: World) -> str | None:
        v = _get(world, self.field)
        if not isinstance(v, dict):
            return None
        key = f"__phantom_{self._rng.randint(1000, 9999)}__"
        # Pick a sentinel value that mirrors an existing entry's type when possible
        # so consumers don't crash on type mismatch; otherwise use a string.
        if v:
            sample = next(iter(v.values()))
            try:
                fake_value: Any = type(sample)() if not isinstance(sample, bool) else False
            except Exception:
                fake_value = "<<phantom>>"
        else:
            fake_value = "<<phantom>>"
        v[key] = fake_value
        return f"injected key {key!r} = {fake_value!r}"


# ---- list ----------------------------------------------------------------


class ClearListField(AutoChaosEvent):
    """Drain a list mid-run. Models queue eviction / batch drop."""

    pattern = "clear_list"

    def _mutate(self, world: World) -> str | None:
        v = _get(world, self.field)
        if not isinstance(v, list):
            return None
        size = len(v)
        if size == 0:
            return "already empty (size 0)"
        _set(world, self.field, [])
        return f"cleared {size} entries"


class DuplicateListEntry(AutoChaosEvent):
    """Duplicate a random entry in a list. Models message duplication /
    double-delivery from an upstream bus."""

    pattern = "duplicate_list_entry"

    def _mutate(self, world: World) -> str | None:
        v = _get(world, self.field)
        if not isinstance(v, list) or not v:
            return None
        idx = self._rng.randrange(len(v))
        v.append(v[idx])
        return f"duplicated index {idx} (now {len(v)} entries)"


class ReverseListField(AutoChaosEvent):
    """Reverse a list. Models ordering violation (FIFO becomes LIFO etc.)."""

    pattern = "reverse_list"

    def _mutate(self, world: World) -> str | None:
        v = _get(world, self.field)
        if not isinstance(v, list) or len(v) < 2:
            return None
        v.reverse()
        return f"reversed {len(v)} entries"
