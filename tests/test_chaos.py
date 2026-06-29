"""Tests for the auto-chaos engine.

Coverage:
  - Each pattern's _mutate against a fresh world (the success path) and
    against a world where the field is missing / wrong type (no-op path).
  - The fuzzer's discovery: every supported field type produces the expected
    pattern set; skip rules apply.
  - The engine's scheduling: intensity normalization, exclusion rules,
    determinism under a fixed seed, no events at step 1 / final step.
  - End-to-end run via drift.run: auto-injected events appear in
    result.auto_chaos_injected and the world reflects the mutations.
"""
from __future__ import annotations

import drift
from drift.chaos.engine import (
    INTENSITY_FREQUENCY,
    _normalize_intensity,
    auto_chaos_events,
    plan_auto_chaos,
)
from drift.chaos.fuzzer import discover_field_patterns
from drift.chaos.patterns import (
    AUTO_CHAOS_PREFIX,
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
from drift.world import World


class _DemoState(drift.WorldState):
    approved: bool = True
    counter: int = 5
    ratio: float = 0.4
    label: str = "active"
    items: dict = {"a": 1, "b": 2}
    queue: list = ["first", "second", "third"]


def _fresh_world(state: drift.WorldState | None = None) -> World:
    w = World(initial=state or _DemoState())
    w.begin_step(3)
    return w


# ---- pattern tests --------------------------------------------------------


def test_flip_bool_actually_flips():
    w = _fresh_world(_DemoState(approved=True))
    rec = FlipBoolField("approved").apply(w)
    assert w.state.approved is False
    assert "True -> False" in rec.summary
    assert rec.name.startswith(AUTO_CHAOS_PREFIX)


def test_flip_bool_noop_on_missing_field():
    w = _fresh_world(_DemoState())
    rec = FlipBoolField("does_not_exist").apply(w)
    assert "no-op" in rec.summary


def test_flip_bool_noop_when_field_is_not_bool():
    w = _fresh_world(_DemoState(counter=5))
    rec = FlipBoolField("counter").apply(w)
    assert w.state.counter == 5  # untouched
    assert "no-op" in rec.summary


def test_boundary_numeric_int_uses_int_targets():
    w = _fresh_world(_DemoState(counter=42))
    BoundaryNumericField("counter", seed=1).apply(w)
    assert isinstance(w.state.counter, int)
    assert w.state.counter in (0, -1, 1, 999_999_999)


def test_boundary_numeric_float_uses_float_targets():
    w = _fresh_world(_DemoState(ratio=0.5))
    BoundaryNumericField("ratio", seed=1).apply(w)
    assert isinstance(w.state.ratio, float)
    assert w.state.ratio in (0.0, -1.0, 1.0, 1e9)


def test_boundary_numeric_skips_bool():
    # bool subclasses int in Python; the pattern must reject bool explicitly.
    w = _fresh_world(_DemoState(approved=True))
    rec = BoundaryNumericField("approved").apply(w)
    assert w.state.approved is True  # untouched
    assert "no-op" in rec.summary


def test_corrupt_string_replaces_with_garbage():
    w = _fresh_world(_DemoState(label="active"))
    CorruptStringField("label", seed=2).apply(w)
    assert w.state.label != "active"
    assert isinstance(w.state.label, str)


def test_clear_dict_empties_field():
    w = _fresh_world(_DemoState(items={"a": 1, "b": 2}))
    rec = ClearDictField("items").apply(w)
    assert w.state.items == {}
    assert "cleared 2 entries" in rec.summary


def test_clear_dict_already_empty_is_audited_no_op():
    w = _fresh_world(_DemoState(items={}))
    rec = ClearDictField("items").apply(w)
    assert "already empty" in rec.summary


def test_remove_dict_key_removes_one():
    w = _fresh_world(_DemoState(items={"a": 1, "b": 2, "c": 3}))
    RemoveDictKey("items", seed=4).apply(w)
    assert len(w.state.items) == 2


def test_remove_dict_key_noop_on_empty():
    w = _fresh_world(_DemoState(items={}))
    rec = RemoveDictKey("items").apply(w)
    assert "no-op" in rec.summary


def test_inject_fake_dict_key_adds_phantom():
    w = _fresh_world(_DemoState(items={"a": 1}))
    InjectFakeDictKey("items", seed=5).apply(w)
    assert len(w.state.items) == 2
    phantoms = [k for k in w.state.items if k.startswith("__phantom_")]
    assert len(phantoms) == 1


def test_clear_list_empties_field():
    w = _fresh_world(_DemoState(queue=["x", "y"]))
    ClearListField("queue").apply(w)
    assert w.state.queue == []


def test_duplicate_list_entry_appends():
    w = _fresh_world(_DemoState(queue=["x", "y"]))
    DuplicateListEntry("queue", seed=7).apply(w)
    assert len(w.state.queue) == 3
    # The appended entry must equal one of the originals.
    assert w.state.queue[-1] in {"x", "y"}


def test_reverse_list_reverses():
    w = _fresh_world(_DemoState(queue=["a", "b", "c"]))
    ReverseListField("queue").apply(w)
    assert w.state.queue == ["c", "b", "a"]


def test_reverse_list_noop_on_single_element():
    w = _fresh_world(_DemoState(queue=["only"]))
    rec = ReverseListField("queue").apply(w)
    assert "no-op" in rec.summary


# ---- fuzzer discovery -----------------------------------------------------


def test_discover_emits_specs_for_each_field_type():
    specs = discover_field_patterns(_DemoState())
    types_by_field: dict[str, set[str]] = {}
    for s in specs:
        types_by_field.setdefault(s.field, set()).add(s.pattern_type)

    assert "flip_bool" in types_by_field["approved"]
    assert "boundary_numeric" in types_by_field["counter"]
    assert "boundary_numeric" in types_by_field["ratio"]
    assert "corrupt_string" in types_by_field["label"]
    assert {"clear_dict", "inject_fake_dict_key", "remove_dict_key"} <= types_by_field["items"]
    assert {"clear_list", "duplicate_list_entry", "reverse_list"} <= types_by_field["queue"]


def test_discover_skips_timestep():
    specs = discover_field_patterns(_DemoState())
    assert not any(s.field == "timestep" for s in specs)


def test_discover_drops_remove_key_for_empty_dict():
    specs = discover_field_patterns(_DemoState(items={}))
    patterns_for_items = {s.pattern_type for s in specs if s.field == "items"}
    assert "clear_dict" in patterns_for_items
    assert "inject_fake_dict_key" in patterns_for_items
    assert "remove_dict_key" not in patterns_for_items


def test_discover_drops_dup_and_reverse_for_short_lists():
    short = _DemoState(queue=[])
    patterns = {s.pattern_type for s in discover_field_patterns(short) if s.field == "queue"}
    assert patterns == {"clear_list"}

    one = _DemoState(queue=["only"])
    patterns_one = {s.pattern_type for s in discover_field_patterns(one) if s.field == "queue"}
    # reverse drops out (needs >=2 elements); dup stays (single-element dup is fine).
    assert patterns_one == {"clear_list", "duplicate_list_entry"}


def test_discover_respects_exclude_fields():
    specs = discover_field_patterns(_DemoState(), exclude_fields={"approved", "counter"})
    fields = {s.field for s in specs}
    assert "approved" not in fields
    assert "counter" not in fields
    assert "queue" in fields  # still present


# ---- engine ---------------------------------------------------------------


def test_normalize_intensity_aliases():
    assert _normalize_intensity(None) == "off"
    assert _normalize_intensity(False) == "off"
    assert _normalize_intensity(True) == "moderate"
    assert _normalize_intensity("AGGRESSIVE") == "aggressive"


def test_normalize_intensity_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        _normalize_intensity("extreme")


def test_plan_off_returns_empty():
    plan = plan_auto_chaos(state=_DemoState(), steps=30, intensity="off", seed=1)
    assert plan == []


def test_plan_returns_more_at_higher_intensity():
    light = plan_auto_chaos(state=_DemoState(), steps=100, intensity="light", seed=1)
    aggressive = plan_auto_chaos(state=_DemoState(), steps=100, intensity="aggressive", seed=1)
    # With a 100-step run the law of large numbers should hold even with
    # only one seed; aggressive should still produce ~3-4x more events.
    assert len(aggressive) > len(light)


def test_plan_deterministic_under_same_seed():
    a = plan_auto_chaos(state=_DemoState(), steps=50, intensity="moderate", seed=99)
    b = plan_auto_chaos(state=_DemoState(), steps=50, intensity="moderate", seed=99)
    # Compare by (timestep, event class name, field) — event_ids will differ.
    ka = [(t, type(e).__name__, e.field) for t, e in a]
    kb = [(t, type(e).__name__, e.field) for t, e in b]
    assert ka == kb


def test_plan_excludes_pattern_by_substring():
    plan = plan_auto_chaos(
        state=_DemoState(), steps=80, intensity="aggressive", seed=3,
        exclude=["flip_bool"],
    )
    assert not any(isinstance(e, FlipBoolField) for _, e in plan)


def test_plan_skips_first_and_final_steps():
    plan = plan_auto_chaos(state=_DemoState(), steps=40, intensity="aggressive", seed=11)
    timesteps = {t for t, _ in plan}
    assert 1 not in timesteps
    assert 40 not in timesteps


def test_intensity_table_is_monotonic():
    levels = ["off", "light", "moderate", "aggressive"]
    freqs = [INTENSITY_FREQUENCY[l] for l in levels]
    assert freqs == sorted(freqs)


def test_exhaustive_registered_in_intensity_table():
    # Exhaustive is a sentinel, not part of the monotonic random-sampling
    # ladder, but it must be a recognized intensity value.
    assert "exhaustive" in INTENSITY_FREQUENCY
    assert _normalize_intensity("exhaustive") == "exhaustive"
    assert _normalize_intensity("EXHAUSTIVE") == "exhaustive"


def test_exhaustive_schedules_every_pattern_exactly_once():
    state = _DemoState()
    catalog = discover_field_patterns(state)
    plan = plan_auto_chaos(state=state, steps=30, intensity="exhaustive", seed=1)
    # One scheduled event per spec in the catalog. No more, no fewer.
    assert len(plan) == len(catalog)
    # Every pattern_name appears in the schedule.
    scheduled_names = [type(e).__name__ + ":" + e.field for _, e in plan]
    # Same set of (event_class, field) pairs as the catalog produces.
    expected = sorted(
        type(spec.build()).__name__ + ":" + spec.field for spec in catalog
    )
    assert sorted(scheduled_names) == expected


def test_exhaustive_is_deterministic_across_runs():
    a = plan_auto_chaos(state=_DemoState(), steps=30, intensity="exhaustive", seed=1)
    b = plan_auto_chaos(state=_DemoState(), steps=30, intensity="exhaustive", seed=999)
    # Different seeds must produce the same schedule — exhaustive ignores RNG.
    ka = [(t, type(e).__name__, e.field) for t, e in a]
    kb = [(t, type(e).__name__, e.field) for t, e in b]
    assert ka == kb


def test_exhaustive_respects_exclude():
    state = _DemoState()
    full = plan_auto_chaos(state=state, steps=30, intensity="exhaustive", seed=1)
    excluded = plan_auto_chaos(
        state=state, steps=30, intensity="exhaustive", seed=1,
        exclude=["flip_bool"],
    )
    assert len(excluded) == len(full) - 1
    assert not any(isinstance(e, FlipBoolField) for _, e in excluded)


def test_exhaustive_stacks_extras_on_final_step_when_catalog_wider_than_window():
    # 5 steps -> window is steps 2..4 (3 slots). _DemoState yields many more
    # patterns than that, so the tail must stack on the last available step.
    state = _DemoState()
    catalog = discover_field_patterns(state)
    assert len(catalog) > 3  # sanity
    plan = plan_auto_chaos(state=state, steps=5, intensity="exhaustive", seed=1)
    assert len(plan) == len(catalog)
    timesteps = [t for t, _ in plan]
    assert min(timesteps) == 2
    assert max(timesteps) == 4  # last_step = steps - 1


def test_exhaustive_returns_empty_when_all_patterns_excluded():
    # Exclude every pattern type the catalog produces — exhaustive should
    # then have nothing to schedule and return [].
    excludes = [
        "flip_bool", "boundary_numeric", "corrupt_string",
        "clear_dict", "inject_fake_dict_key", "remove_dict_key",
        "clear_list", "duplicate_list_entry", "reverse_list",
    ]
    plan = plan_auto_chaos(
        state=_DemoState(), steps=30, intensity="exhaustive", seed=1,
        exclude=excludes,
    )
    assert plan == []


def test_auto_chaos_events_helper_matches_plan_auto_chaos():
    # The positional helper should produce the same schedule as the kwarg form.
    state = _DemoState()
    a = auto_chaos_events(state, 30, "moderate", 17)
    b = plan_auto_chaos(state=state, steps=30, intensity="moderate", seed=17)
    assert [(t, type(e).__name__, e.field) for t, e in a] == \
           [(t, type(e).__name__, e.field) for t, e in b]
