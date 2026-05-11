"""Code-review topology events."""
from __future__ import annotations

from drift.events.base import Event, EventRecord
from drift.world import World


class UrgentFixRequest(Event):
    name = "UrgentFixRequest"

    def apply(self, world: World) -> EventRecord:
        # External pressure to ship — drives reviewer/merge agents to be looser.
        world.adjust_load(+0.15, source="event", source_id=self.event_id)
        new_pressure = min(1.0, getattr(world.state, "deadline_pressure", 0.0) + 0.2)
        world.state.deadline_pressure = new_pressure
        world.record_change("event", self.event_id, f"deadline_pressure -> {new_pressure:.2f}")
        return EventRecord(
            event_id=self.event_id, timestep=world.state.timestep, name=self.name,
            summary="+0.15 load, +0.20 deadline_pressure",
        )


class DependencyCVE(Event):
    name = "DependencyCVE"

    def apply(self, world: World) -> EventRecord:
        # A new CVE means SecurityAgent has more work. Mark all unreviewed PRs
        # as needing security re-review.
        affected = 0
        for case in world.state.open_cases.values():
            if case.extra.get("security_status") == "cleared":
                case.extra["security_status"] = "unreviewed"
                affected += 1
        world.adjust_load(+0.10, source="event", source_id=self.event_id)
        world.record_change("event", self.event_id, f"CVE: {affected} PRs back to unreviewed")
        return EventRecord(
            event_id=self.event_id, timestep=world.state.timestep, name=self.name,
            summary=f"CVE published; {affected} PRs need re-review",
        )


class DeadlinePressure(Event):
    name = "DeadlinePressure"

    def apply(self, world: World) -> EventRecord:
        new_pressure = min(1.0, getattr(world.state, "deadline_pressure", 0.0) + 0.15)
        world.state.deadline_pressure = new_pressure
        world.adjust_sentiment(-0.05, source="event", source_id=self.event_id)
        world.record_change("event", self.event_id, f"deadline_pressure -> {new_pressure:.2f}")
        return EventRecord(
            event_id=self.event_id, timestep=world.state.timestep, name=self.name,
            summary="+0.15 deadline_pressure, -0.05 morale",
        )


class ConflictingRebase(Event):
    name = "ConflictingRebase"

    def apply(self, world: World) -> EventRecord:
        # Some PRs lose their approvals when the base branch shifts.
        invalidated = 0
        for case in world.state.open_cases.values():
            if case.extra.get("approvals", 0) > 0:
                case.extra["approvals"] = max(0, case.extra["approvals"] - 1)
                invalidated += 1
        world.adjust_sentiment(-0.03, source="event", source_id=self.event_id)
        world.record_change("event", self.event_id, f"rebase invalidated {invalidated} approvals")
        return EventRecord(
            event_id=self.event_id, timestep=world.state.timestep, name=self.name,
            summary=f"rebase: {invalidated} approvals invalidated",
        )


CODE_REVIEW_EVENTS: dict[str, type[Event]] = {
    "UrgentFixRequest":  UrgentFixRequest,
    "DependencyCVE":     DependencyCVE,
    "DeadlinePressure":  DeadlinePressure,
    "ConflictingRebase": ConflictingRebase,
}
