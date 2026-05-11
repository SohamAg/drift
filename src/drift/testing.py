"""Test utilities — reset all module-level counters so two runs in the
same process produce identical IDs (essential for determinism tests).

Production runs typically don't need this; tests and parameter sweeps do.
"""
from __future__ import annotations

from drift.agents.base import reset_action_counter
from drift.events.base import reset_event_counter
from drift.events.library import _case_counter as _support_case_counter  # noqa: F401
from drift.failures.base import reset_failure_counter


def reset_all_counters() -> None:
    """Reset every monotonic counter drift uses, including topology-specific
    ones. Call before each run in tests / sweeps where determinism matters."""
    import itertools as _it

    reset_action_counter()
    reset_event_counter()
    reset_failure_counter()

    # Topology-specific case/PR/incident counters live in their own modules.
    # We reset them by name so this stays a single chokepoint.
    import drift.events.library as _support_lib
    _support_lib._case_counter = _it.count(1)

    try:
        import drift.topologies.code_review.agents as _cr_agents
        _cr_agents._pr_counter = _it.count(1)
    except ImportError:
        pass

    try:
        import drift.topologies.ops.agents as _ops_agents
        _ops_agents._inc_counter = _it.count(1)
    except ImportError:
        pass
