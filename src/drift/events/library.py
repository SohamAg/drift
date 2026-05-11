"""The default event library.

Events mutate world state through the World API. They never emit Actions —
they are exogenous, not agent-driven.
"""
from __future__ import annotations

import itertools

from drift.events.base import Event, EventRecord
from drift.world import Case, World

_case_counter = itertools.count(1)


def _new_case_id() -> str:
    return f"c{next(_case_counter):05d}"


class BlackFridaySpike(Event):
    name = "BlackFridaySpike"

    def apply(self, world: World) -> EventRecord:
        world.adjust_load(+0.3, source="event", source_id=self.event_id)
        world.adjust_sentiment(-0.05, source="event", source_id=self.event_id)
        for _ in range(5):
            cid = _new_case_id()
            world.add_case(
                Case(case_id=cid, customer_id=f"u{cid}", issue="order delay", refund_requested=False, opened_at_step=world.state.timestep),
                source="event",
                source_id=self.event_id,
            )
        return EventRecord(
            event_id=self.event_id, timestep=world.state.timestep, name=self.name,
            summary="+0.3 load, +5 cases, -0.05 sentiment",
        )


class RefundPolicyChange(Event):
    name = "RefundPolicyChange"

    def apply(self, world: World) -> EventRecord:
        new_v = world.state.refund_policy_version + 1
        world.set_policy_version(new_v, source="event", source_id=self.event_id)
        return EventRecord(
            event_id=self.event_id, timestep=world.state.timestep, name=self.name,
            summary=f"policy -> v{new_v}",
        )


class InventoryAPIDelay(Event):
    name = "InventoryAPIDelay"

    def apply(self, world: World) -> EventRecord:
        world.adjust_inventory_delay(+30, source="event", source_id=self.event_id)
        world.adjust_load(+0.05, source="event", source_id=self.event_id)
        return EventRecord(
            event_id=self.event_id, timestep=world.state.timestep, name=self.name,
            summary="+30min inventory delay, +0.05 load",
        )


class AngryCustomerSurge(Event):
    name = "AngryCustomerSurge"

    def apply(self, world: World) -> EventRecord:
        world.adjust_sentiment(-0.15, source="event", source_id=self.event_id)
        for _ in range(3):
            cid = _new_case_id()
            world.add_case(
                Case(
                    case_id=cid, customer_id=f"u{cid}", issue="angry refund demand",
                    refund_requested=True, opened_at_step=world.state.timestep,
                ),
                source="event",
                source_id=self.event_id,
            )
        return EventRecord(
            event_id=self.event_id, timestep=world.state.timestep, name=self.name,
            summary="-0.15 sentiment, +3 angry cases",
        )


EVENT_REGISTRY: dict[str, type[Event]] = {
    "BlackFridaySpike": BlackFridaySpike,
    "RefundPolicyChange": RefundPolicyChange,
    "InventoryAPIDelay": InventoryAPIDelay,
    "AngryCustomerSurge": AngryCustomerSurge,
}
