from __future__ import annotations

from drift.agents.base import Action, Agent, ObservationView
from drift.world import World, WorldState


class EscalationAgent(Agent):
    role = "escalation"
    system_prompt = (
        "You manage the escalation queue. Resolve cases when you can; "
        "rebound them only if you genuinely cannot proceed."
    )

    def observe(self, state: WorldState) -> ObservationView:
        return ObservationView(
            timestep=state.timestep,
            system_load=state.system_load,
            queue_case_ids=[r.case_id for r in state.escalation_queue],
        )

    def apply(self, action: Action, world: World) -> None:
        if action.kind == "resolve" and action.target_case_id:
            if action.target_case_id in world.state.open_cases:
                world.remove_case(action.target_case_id, source="action", source_id=action.action_id)
                world.adjust_sentiment(+0.02, source="action", source_id=action.action_id)
            else:
                # Stale queue entry — clear it from the queue.
                world.state.escalation_queue = [
                    r for r in world.state.escalation_queue if r.case_id != action.target_case_id
                ]
        elif action.kind == "rebound" and action.target_case_id:
            # Pop from front and re-enqueue at the back. This is what produces escalation loops.
            world.state.escalation_queue = [
                r for r in world.state.escalation_queue if r.case_id != action.target_case_id
            ]
            if action.target_case_id in world.state.open_cases:
                world.enqueue_escalation(action.target_case_id, source="action", source_id=action.action_id)
                world.adjust_sentiment(-0.02, source="action", source_id=action.action_id)
