"""Code-review topology agents.

A "PR" is modeled as a Case in `world.open_cases`, with PR-specific data
held in `case.extra`:
  - approvals: int
  - rejections: int
  - security_status: 'unreviewed' | 'blocked' | 'cleared'
  - urgency: 'low' | 'high'
  - merged: bool (set true on merge so detectors can see it; case removed on merge)
"""
from __future__ import annotations

import itertools

from drift.agents.base import Action, Agent, ObservationView
from drift.world import Case, World, WorldState

_pr_counter = itertools.count(1)


def _new_pr_id() -> str:
    return f"pr{next(_pr_counter):04d}"


class ProposerAgent(Agent):
    role = "proposer"
    system_prompt = "You open new pull requests."

    def observe(self, state: WorldState) -> ObservationView:
        # Proposer needs to know team load and deadline pressure to decide cadence.
        return ObservationView(
            timestep=state.timestep,
            system_load=state.system_load,
            customer_sentiment=state.customer_sentiment,  # team morale
            open_case_ids=list(state.open_cases.keys()),
        )

    def apply(self, action: Action, world: World) -> None:
        if action.kind == "propose_change":
            pr_id = _new_pr_id()
            case = Case(case_id=pr_id, customer_id="", issue="new PR", opened_at_step=world.state.timestep)
            case.extra = {
                "approvals": 0,
                "rejections": 0,
                "security_status": "unreviewed",
                "urgency": "high" if (world.state.system_load > 0.6) else "low",
                "merged": False,
            }
            world.add_case(case, source="action", source_id=action.action_id)


class ReviewerAgent(Agent):
    role = "reviewer"
    system_prompt = "You review pull requests."

    def observe(self, state: WorldState) -> ObservationView:
        return ObservationView(
            timestep=state.timestep,
            system_load=state.system_load,
            open_case_ids=list(state.open_cases.keys()),
        )

    def _build_action(self, timestep: int, resp) -> Action:  # type: ignore[override]
        kind_map = {"approve": "approve_review", "reject": "reject_review", "no_op": "no_op"}
        return Action(
            timestep=timestep,
            agent_name=self.name,
            kind=kind_map.get(resp.decision, "no_op"),
            target_case_id=resp.target_case_id,
            rationale=resp.rationale,
        )

    def apply(self, action: Action, world: World) -> None:
        if not action.target_case_id or action.target_case_id not in world.state.open_cases:
            return
        case = world.state.open_cases[action.target_case_id]
        if action.kind == "approve_review":
            case.extra["approvals"] = case.extra.get("approvals", 0) + 1
            world.record_change("action", action.action_id, f"approve {case.case_id}")
        elif action.kind == "reject_review":
            case.extra["rejections"] = case.extra.get("rejections", 0) + 1
            world.record_change("action", action.action_id, f"reject {case.case_id}")
            world.adjust_sentiment(-0.01, source="action", source_id=action.action_id)


class SecurityAgent(Agent):
    role = "security"
    system_prompt = "You guard the codebase from security regressions."

    def observe(self, state: WorldState) -> ObservationView:
        return ObservationView(
            timestep=state.timestep,
            system_load=state.system_load,
            open_case_ids=list(state.open_cases.keys()),
        )

    def _build_action(self, timestep: int, resp) -> Action:  # type: ignore[override]
        kind_map = {"block": "security_block", "clear": "security_clear", "no_op": "no_op"}
        return Action(
            timestep=timestep,
            agent_name=self.name,
            kind=kind_map.get(resp.decision, "no_op"),
            target_case_id=resp.target_case_id,
            rationale=resp.rationale,
        )

    def apply(self, action: Action, world: World) -> None:
        if not action.target_case_id or action.target_case_id not in world.state.open_cases:
            return
        case = world.state.open_cases[action.target_case_id]
        if action.kind == "security_block":
            case.extra["security_status"] = "blocked"
            world.record_change("action", action.action_id, f"block {case.case_id}")
        elif action.kind == "security_clear":
            case.extra["security_status"] = "cleared"
            world.record_change("action", action.action_id, f"clear {case.case_id}")


class MergeAgent(Agent):
    role = "merge"
    system_prompt = "You merge approved pull requests."

    def observe(self, state: WorldState) -> ObservationView:
        return ObservationView(
            timestep=state.timestep,
            system_load=state.system_load,
            open_case_ids=list(state.open_cases.keys()),
        )

    def _build_action(self, timestep: int, resp) -> Action:  # type: ignore[override]
        kind_map = {"merge": "merge", "defer": "defer", "no_op": "no_op"}
        return Action(
            timestep=timestep,
            agent_name=self.name,
            kind=kind_map.get(resp.decision, "no_op"),
            target_case_id=resp.target_case_id,
            rationale=resp.rationale,
        )

    def apply(self, action: Action, world: World) -> None:
        if not action.target_case_id or action.target_case_id not in world.state.open_cases:
            return
        case = world.state.open_cases[action.target_case_id]
        if action.kind == "merge":
            case.extra["merged"] = True
            # Tag as merged but DON'T remove yet — detectors need to see the merge.
            # Remove next step via recording the case as closed.
            world.adjust_sentiment(+0.02, source="action", source_id=action.action_id)
            world.record_change("action", action.action_id, f"merge {case.case_id}")
            # Clean up: pop the case from open_cases; the merge is already recorded.
            world.remove_case(case.case_id, source="action", source_id=action.action_id)
        elif action.kind == "defer":
            world.record_change("action", action.action_id, f"defer {case.case_id}")
