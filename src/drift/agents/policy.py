from __future__ import annotations

from drift.agents.base import Action, Agent, ObservationView
from drift.world import World, WorldState


class PolicyAgent(Agent):
    role = "policy"
    system_prompt = (
        "You are the policy steward. Periodically refresh the refund policy "
        "version when the environment shifts."
    )

    def observe(self, state: WorldState) -> ObservationView:
        return ObservationView(
            timestep=state.timestep,
            refund_policy_version=state.refund_policy_version,
            system_load=state.system_load,
            customer_sentiment=state.customer_sentiment,
        )

    def apply(self, action: Action, world: World) -> None:
        if action.kind == "policy_update":
            world.set_policy_version(
                world.state.refund_policy_version + 1,
                source="action",
                source_id=action.action_id,
            )
