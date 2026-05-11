"""Event abstractions.

Events are the world's source of exogenous change. Each Event has an
`apply(world)` that performs its effect through the World API, so the
audit trail captures it the same way agent actions do.
"""
from __future__ import annotations

import itertools
from abc import ABC, abstractmethod

from pydantic import BaseModel

from drift.world import World

_event_counter = itertools.count(1)


def reset_event_counter() -> None:
    global _event_counter
    _event_counter = itertools.count(1)


class EventRecord(BaseModel):
    event_id: str
    timestep: int
    name: str
    summary: str


class Event(ABC):
    name: str = "event"

    def __init__(self) -> None:
        self.event_id = f"e{next(_event_counter):06d}"

    @abstractmethod
    def apply(self, world: World) -> EventRecord: ...
