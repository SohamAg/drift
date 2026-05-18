"""State-schema introspection: WorldState -> chaos pattern catalog.

Given a WorldState instance, walk every field (declared + extra fields
allowed by Pydantic's `extra="allow"`) and emit a list of ChaosSpec
factories — each spec is a (pattern_name, factory) pair that, when called,
produces a fresh AutoChaosEvent targeting that field.

A few fields are always skipped because mutating them would corrupt the
simulation rather than test the agents:
  - `timestep` (the simulation clock)
  - any field whose name starts with `_`
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from drift.chaos.patterns import (
    AutoChaosEvent,
    BoundaryNumericField,
    ClearDictField,
    ClearListField,
    CorruptStringField,
    DuplicateListEntry,
    FlipBoolField,
    InjectFakeDictKey,
    RemoveDictKey,
    ReverseListField,
)
from drift.world import WorldState

# Fields drift's own machinery owns — fuzzing them would break the runner
# rather than stress the user's agents. Keep this list tiny.
_SKIP_FIELDS: frozenset[str] = frozenset({"timestep"})


@dataclass
class ChaosSpec:
    """A factory for one chaos event targeting one field.

    `pattern_name` is the human-readable identifier (e.g. "flip_bool[approved]")
    used for excluding patterns and for logging. `build()` returns a fresh
    AutoChaosEvent instance — call once per scheduled occurrence so each
    injection gets a unique event_id.
    """

    pattern_name: str
    field: str
    pattern_type: str
    factory: Callable[[], AutoChaosEvent]

    def build(self) -> AutoChaosEvent:
        return self.factory()


def _specs_for_field(field: str, value: Any, seed: int) -> list[ChaosSpec]:
    """Return all chaos specs appropriate for one field given its current value.

    Type dispatch is on the runtime value (not the declared annotation),
    because Pydantic's `extra="allow"` fields don't have one. Bool is checked
    before int because bool is a subclass of int in Python.
    """
    out: list[ChaosSpec] = []

    def _spec(pattern: str, cls: type[AutoChaosEvent]) -> ChaosSpec:
        return ChaosSpec(
            pattern_name=f"{pattern}[{field}]",
            field=field,
            pattern_type=pattern,
            factory=lambda f=field, s=seed: cls(f, seed=s),
        )

    if isinstance(value, bool):
        out.append(_spec("flip_bool", FlipBoolField))
    elif isinstance(value, (int, float)):
        out.append(_spec("boundary_numeric", BoundaryNumericField))
    elif isinstance(value, str):
        out.append(_spec("corrupt_string", CorruptStringField))
    elif isinstance(value, dict):
        out.append(_spec("clear_dict", ClearDictField))
        out.append(_spec("inject_fake_dict_key", InjectFakeDictKey))
        if value:
            out.append(_spec("remove_dict_key", RemoveDictKey))
    elif isinstance(value, list):
        out.append(_spec("clear_list", ClearListField))
        if value:
            out.append(_spec("duplicate_list_entry", DuplicateListEntry))
        if len(value) >= 2:
            out.append(_spec("reverse_list", ReverseListField))
    return out


def discover_field_patterns(
    state: WorldState,
    *,
    exclude_fields: set[str] | None = None,
    seed: int = 0,
) -> list[ChaosSpec]:
    """Walk every field on `state` and emit applicable chaos specs.

    Args:
        state: the initial WorldState. The fuzzer reads value types from this
            snapshot — fields added later (by events) won't get their own
            specs unless the user reruns discovery.
        exclude_fields: field names to skip entirely (no specs emitted).
        seed: base seed for the RNG inside each event. Each event still gets
            an independent stream because the events are constructed at
            schedule time, not now.

    Returns:
        A list of ChaosSpec, possibly empty if the state has no fuzzable
        fields (e.g. only a `timestep`).
    """
    skip = _SKIP_FIELDS | (exclude_fields or set())
    dump = state.model_dump()
    specs: list[ChaosSpec] = []
    for field, value in dump.items():
        if field in skip or field.startswith("_"):
            continue
        specs.extend(_specs_for_field(field, value, seed))
    return specs
