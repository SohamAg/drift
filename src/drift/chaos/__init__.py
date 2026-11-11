"""Auto-chaos: drift generates chaos events instead of asking the user to.

The user-facing entry point is the `auto_chaos` keyword on `drift.run`:

    drift.run(
        agents=[...],
        state=MyState(...),
        steps=30,
        auto_chaos="moderate",   # or "aggressive", "off" / False / None
    )

What drift does at run start when `auto_chaos` is set:
  1. Introspects the WorldState instance to discover every field and type.
  2. Builds a catalog of type-appropriate chaos patterns
     (flip_bool[field], clear_dict[field], boundary_numeric[field], etc.).
  3. Schedules a subset across the run's timesteps at a frequency that
     scales with the intensity setting.
  4. Tags each injected EventRecord with the prefix `AutoChaos.` so the
     result can attribute fired failures back to specific chaos patterns.

The auto-injected events are merged with any user-supplied `events=[...]`,
so handcrafted chaos still works alongside auto-chaos.

Result reporting: after a run, `RunResult.auto_chaos_injected` lists every
auto-generated EventRecord that fired. Mapping those back to failures lets
the user see which auto-injected chaos pattern surfaced which failure.

See `examples/adapters/langgraph_demo.py` for a worked end-to-end example.
"""
from drift.chaos.engine import (
    INTENSITY_FREQUENCY,
    auto_chaos_events,
    plan_auto_chaos,
)
from drift.chaos.fuzzer import ChaosSpec, discover_field_patterns
from drift.chaos.tools import (
    TOOL_FAULTS,
    ToolChaosEvent,
    discover_tool_patterns,
    tool_names,
)
from drift.chaos.patterns import (
    AUTO_CHAOS_PREFIX,
    CANARY_PATTERN,
    CANARY_PREFIX,
    MESSAGE_PATTERNS,
    AutoChaosEvent,
    BoundaryNumericField,
    ClearDictField,
    ClearListField,
    CorruptStringField,
    DropLastUserMessage,
    DuplicateListEntry,
    FlipBoolField,
    GarbleLastUserMessage,
    InjectFabricatedPriorTurn,
    InjectFakeDictKey,
    InjectToolErrorResult,
    InjectToolResultInstruction,
    InjectToolResultToolHijack,
    MessagesChaosEvent,
    RemoveDictKey,
    ReverseListField,
    TruncateLastUserMessage,
)

__all__ = [
    "AUTO_CHAOS_PREFIX",
    "CANARY_PATTERN",
    "CANARY_PREFIX",
    "MESSAGE_PATTERNS",
    "AutoChaosEvent",
    "BoundaryNumericField",
    "ChaosSpec",
    "ClearDictField",
    "ClearListField",
    "CorruptStringField",
    "DropLastUserMessage",
    "DuplicateListEntry",
    "FlipBoolField",
    "GarbleLastUserMessage",
    "INTENSITY_FREQUENCY",
    "InjectFabricatedPriorTurn",
    "InjectFakeDictKey",
    "InjectToolErrorResult",
    "InjectToolResultInstruction",
    "InjectToolResultToolHijack",
    "MessagesChaosEvent",
    "RemoveDictKey",
    "ReverseListField",
    "TruncateLastUserMessage",
    "auto_chaos_events",
    "discover_field_patterns",
    "plan_auto_chaos",
]
