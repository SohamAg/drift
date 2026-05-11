from __future__ import annotations

from drift.agents.base import Action, Agent, ObservationView
from drift.world import World, WorldState


class RefundAgent(Agent):
    role = "refund"
    system_prompt = (
        "You are a refund decision agent. Approve or deny refunds based on "
        "current policy and customer sentiment. Always cite the policy version you used."
    )

    def observe(self, state: WorldState) -> ObservationView:
        return ObservationView(
            timestep=state.timestep,
            customer_sentiment=state.customer_sentiment,
            refund_policy_version=state.refund_policy_version,
            open_case_ids=list(state.open_cases.keys()),
        )

    def _build_action(self, timestep: int, resp) -> Action:  # type: ignore[override]
        # The mock returns "approve"/"deny" — translate to typed kinds.
        kind_map = {"approve": "refund_approve", "deny": "refund_deny", "no_op": "no_op"}
        kind = kind_map.get(resp.decision, "no_op")
        return Action(
            timestep=timestep,
            agent_name=self.name,
            kind=kind,  # type: ignore[arg-type]
            target_case_id=resp.target_case_id,
            rationale=resp.rationale,
            referenced_policy_version=resp.referenced_policy_version,
        )

    def apply(self, action: Action, world: World) -> None:
        if action.kind not in ("refund_approve", "refund_deny"):
            return
        if not action.target_case_id or action.target_case_id not in world.state.open_cases:
            # Hallucinated reference — detector picks this up; world is not mutated.
            return
        # Approvals close the case and bump sentiment; denials drop sentiment.
        if action.kind == "refund_approve":
            world.adjust_sentiment(+0.05, source="action", source_id=action.action_id)
            world.remove_case(action.target_case_id, source="action", source_id=action.action_id)
        else:
            world.adjust_sentiment(-0.04, source="action", source_id=action.action_id)
