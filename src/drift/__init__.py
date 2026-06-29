"""drift — pre-deploy chaos testing for LangGraph multi-agent systems.

After the 2026-06-29 cleanup the native per-tick simulator is gone. The
primary surface is now the LangGraph adapter:

    from drift.adapters.langgraph import drift_test

    result = drift_test(
        graph=my_compiled_graph,
        initial_state={"messages": [...], ...},
        intensity="aggressive",
    )

The BYOA `@drift.agent` decorator survives but its runtime path (`drift.run`)
was removed alongside the simulator. The decorator is preserved as a shape
for future re-wiring against the adapter.
"""
__version__ = "0.2.0"

from drift.agents.base import Action, Agent
from drift.events.base import Event, EventRecord
from drift.sdk import agent
from drift.world import World, WorldState

__all__ = [
    "Action",
    "Agent",
    "Event",
    "EventRecord",
    "World",
    "WorldState",
    "agent",
    "__version__",
]
