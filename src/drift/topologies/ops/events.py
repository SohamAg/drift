"""Ops topology events."""
from __future__ import annotations

from drift.events.base import Event, EventRecord
from drift.topologies.ops.agents import make_incident
from drift.world import World


class SeveritySpike(Event):
    """A new high-severity incident drops in."""
    name = "SeveritySpike"

    def apply(self, world: World) -> EventRecord:
        case = make_incident("sev1", world.state.timestep)
        world.add_case(case, source="event", source_id=self.event_id)
        world.adjust_load(+0.20, source="event", source_id=self.event_id)
        world.adjust_sentiment(-0.06, source="event", source_id=self.event_id)
        return EventRecord(
            event_id=self.event_id, timestep=world.state.timestep, name=self.name,
            summary="+1 sev1 incident, +0.20 load, -0.06 trust",
        )


class DependentServiceDown(Event):
    """An upstream service died — multiple incidents simultaneously."""
    name = "DependentServiceDown"

    def apply(self, world: World) -> EventRecord:
        for sev in ("high", "high", "low"):
            world.add_case(make_incident(sev, world.state.timestep), source="event", source_id=self.event_id)
        world.adjust_load(+0.25, source="event", source_id=self.event_id)
        world.adjust_sentiment(-0.08, source="event", source_id=self.event_id)
        return EventRecord(
            event_id=self.event_id, timestep=world.state.timestep, name=self.name,
            summary="upstream outage: +3 incidents, +0.25 load",
        )


class CustomerNoise(Event):
    """Public confusion / noise on social channels — sentiment hit even without new incidents."""
    name = "CustomerNoise"

    def apply(self, world: World) -> EventRecord:
        world.adjust_sentiment(-0.05, source="event", source_id=self.event_id)
        return EventRecord(
            event_id=self.event_id, timestep=world.state.timestep, name=self.name,
            summary="-0.05 trust from external noise",
        )


class PartialOutage(Event):
    """A small subset of customers affected — low-sev incident plus mild trust drag."""
    name = "PartialOutage"

    def apply(self, world: World) -> EventRecord:
        case = make_incident("low", world.state.timestep)
        world.add_case(case, source="event", source_id=self.event_id)
        world.adjust_load(+0.05, source="event", source_id=self.event_id)
        world.adjust_sentiment(-0.02, source="event", source_id=self.event_id)
        return EventRecord(
            event_id=self.event_id, timestep=world.state.timestep, name=self.name,
            summary="+1 low-sev incident, -0.02 trust",
        )


OPS_EVENTS: dict[str, type[Event]] = {
    "SeveritySpike":         SeveritySpike,
    "DependentServiceDown":  DependentServiceDown,
    "CustomerNoise":         CustomerNoise,
    "PartialOutage":         PartialOutage,
}
