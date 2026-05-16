"""drift — chaos engineering for multi-agent AI systems.

Two public surfaces:

  1. The BYOA / BYOE SDK (the product) — decorate your agents, subclass
     WorldState for your domain, optionally define chaos events, call
     drift.run(). See drift.sdk and examples/byoa_minimal.py.

  2. The shipped topologies (support / code_review / ops) — scaffolding
     for exploring the idea via the CLI + web UI. Run them with
     `python -m drift run --topology support`.

Top-level re-exports below are the BYOA SDK. Power users who need stateful
agents (where actions mutate world state directly) should subclass
drift.agents.base.Agent rather than using the @drift.agent decorator.
"""
__version__ = "0.1.0"

from drift.sdk import (
    Action,
    Agent,
    Event,
    EventRecord,
    World,
    WorldState,
    agent,
    run,
)

__all__ = [
    "Action",
    "Agent",
    "Event",
    "EventRecord",
    "World",
    "WorldState",
    "agent",
    "run",
    "__version__",
]
