"""State-schema introspection: WorldState -> chaos pattern catalog.

Given a WorldState instance, walk every field (declared + extra fields
allowed by Pydantic's `extra="allow"`) and emit a list of ChaosSpec
factories — each spec is a (pattern_name, factory) pair that, when called,
produces a fresh AutoChaosEvent targeting that field.

A few fields are always skipped because mutating them would corrupt the
simulation rather than test the agents:
  - `timestep` (the simulation clock)
  - any field whose name starts with `_`

Two things beyond flat type dispatch:

  - A list that is a chat conversation (see drift.chaos.messages) gets the
    message-aware patterns (drop/truncate/garble the user's instruction,
    inject a failed tool result, inject an instruction inside a tool result,
    inject a fabricated "already done" turn) INSTEAD of the generic
    duplicate/reverse patterns, which produce API-invalid message orderings.
    `clear_list` still applies.
  - Dict fields are walked `max_depth` levels down (default 2); nested
    scalar leaves and nested dicts get their own specs under dotted paths
    (`config.max_retries`). Nested lists are left opaque.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from drift.chaos import messages as _msg
from drift.chaos.patterns import (
    MESSAGE_PATTERNS,
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

DEFAULT_MAX_DEPTH = 2

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
    fault_class: str = "value"

    def build(self) -> AutoChaosEvent:
        return self.factory()


def _specs_for_field(
    field: str, value: Any, seed: int, *, depth: int = 0, max_depth: int = DEFAULT_MAX_DEPTH,
) -> list[ChaosSpec]:
    """Return all chaos specs appropriate for one field given its current value.

    Type dispatch is on the runtime value (not the declared annotation),
    because Pydantic's `extra="allow"` fields don't have one. Bool is checked
    before int because bool is a subclass of int in Python. `field` may be a
    dotted path when recursing into a dict.
    """
    out: list[ChaosSpec] = []

    def _spec(pattern: str, cls: type[AutoChaosEvent]) -> ChaosSpec:
        return ChaosSpec(
            pattern_name=f"{pattern}[{field}]",
            field=field,
            pattern_type=pattern,
            factory=lambda f=field, s=seed: cls(f, seed=s),
            fault_class=getattr(cls, "fault_class", "value"),
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
        if depth < max_depth:
            for k, v in value.items():
                if not isinstance(k, str) or k.startswith("_") or "." in k:
                    continue
                if isinstance(v, list):
                    continue  # nested lists stay opaque
                out.extend(_specs_for_field(
                    f"{field}.{k}", v, seed, depth=depth + 1, max_depth=max_depth,
                ))
    elif isinstance(value, list):
        out.append(_spec("clear_list", ClearListField))
        if _msg.is_messages_list(value):
            for cls in MESSAGE_PATTERNS:
                out.append(_spec(cls.pattern, cls))
        else:
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
    max_depth: int = DEFAULT_MAX_DEPTH,
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
        max_depth: how many dict levels to walk for nested specs. 0 = top
            level only (the pre-nesting behaviour).

    Returns:
        A list of ChaosSpec, possibly empty if the state has no fuzzable
        fields (e.g. only a `timestep`).
    """
    skip = _SKIP_FIELDS | (exclude_fields or set())
    specs: list[ChaosSpec] = []
    # Read live attribute values, not model_dump(): a messages channel holding
    # langchain BaseMessage objects would be dumped to plain dicts and lose the
    # shape the message patterns dispatch on. model_dump() is still the key
    # source so extra="allow" fields are enumerated.
    for field in state.model_dump().keys():
        if field in skip or field.startswith("_"):
            continue
        value = getattr(state, field, None)
        specs.extend(_specs_for_field(field, value, seed, max_depth=max_depth))
    return specs
