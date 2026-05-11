from __future__ import annotations

from drift.agents.base import Action, Agent, ObservationView
from drift.world import World, WorldState


class SupportAgent(Agent):
    role = "support"
    system_prompt = (
        "You are a frontline support agent. Triage open cases. "
        "Respond directly when you can; escalate when load or sentiment requires it."
    )

    def observe(self, state: WorldState) -> ObservationView:
        return ObservationView(
            timestep=state.timestep,
            customer_sentiment=state.customer_sentiment,
            system_load=state.system_load,
            open_case_ids=list(state.open_cases.keys()),
        )

    def apply(self, action: Action, world: World) -> None:
        if action.kind == "escalate" and action.target_case_id:
            if action.target_case_id in world.state.open_cases:
                world.enqueue_escalation(action.target_case_id, source="action", source_id=action.action_id)
        elif action.kind == "respond" and action.target_case_id:
            if action.target_case_id in world.state.open_cases:
                # Direct response nudges sentiment up slightly.
                world.adjust_sentiment(+0.01, source="action", source_id=action.action_id)
