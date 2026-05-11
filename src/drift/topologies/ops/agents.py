"""Ops topology agents.

An incident is modeled as a Case with extras:
  - severity: 'low' | 'high' | 'sev1'
  - triaged: bool
  - diagnosis: str | None
  - remediated: bool
  - comms_sent_at: int | None
  - root_cause_hypothesis: str | None
"""
from __future__ import annotations

import itertools

from drift.agents.base import Action, Agent, ObservationView
from drift.world import Case, World, WorldState

_inc_counter = itertools.count(1)


def _new_inc_id() -> str:
    return f"inc{next(_inc_counter):04d}"


def make_incident(severity: str, opened_at: int) -> Case:
    iid = _new_inc_id()
    case = Case(case_id=iid, customer_id="", issue=f"{severity} incident", opened_at_step=opened_at)
    case.extra = {
        "severity": severity,
        "triaged": False,
        "diagnosis": None,
        "remediated": False,
        "comms_sent_at": None,
        "root_cause_hypothesis": None,
    }
    return case


class TriageAgent(Agent):
    role = "triage"
    system_prompt = "You triage incoming incidents."

    def observe(self, state: WorldState) -> ObservationView:
        return ObservationView(
            timestep=state.timestep,
            system_load=state.system_load,
            open_case_ids=[k for k, c in state.open_cases.items() if not c.extra.get("triaged")],
        )

    def _build_action(self, timestep: int, resp) -> Action:  # type: ignore[override]
        kind_map = {"triage": "triage", "no_op": "no_op"}
        return Action(
            timestep=timestep,
            agent_name=self.name,
            kind=kind_map.get(resp.decision, "no_op"),
            target_case_id=resp.target_case_id,
            rationale=resp.rationale,
        )

    def apply(self, action: Action, world: World) -> None:
        if action.kind != "triage" or not action.target_case_id:
            return
        case = world.state.open_cases.get(action.target_case_id)
        if not case:
            return
        case.extra["triaged"] = True
        world.record_change("action", action.action_id, f"triaged {case.case_id}")


class DiagnosisAgent(Agent):
    role = "diagnosis"
    system_prompt = "You propose a root-cause hypothesis for incidents."

    def observe(self, state: WorldState) -> ObservationView:
        return ObservationView(
            timestep=state.timestep,
            system_load=state.system_load,
            open_case_ids=[k for k, c in state.open_cases.items() if c.extra.get("triaged")],
        )

    def _build_action(self, timestep: int, resp) -> Action:  # type: ignore[override]
        kind_map = {"diagnose": "diagnose", "no_op": "no_op"}
        return Action(
            timestep=timestep,
            agent_name=self.name,
            kind=kind_map.get(resp.decision, "no_op"),
            target_case_id=resp.target_case_id,
            rationale=resp.rationale,
        )

    def apply(self, action: Action, world: World) -> None:
        if action.kind != "diagnose" or not action.target_case_id:
            return
        case = world.state.open_cases.get(action.target_case_id)
        if not case:
            return
        # The diagnosis "hypothesis" is encoded as the action's rationale prefix —
        # detectors compare these strings for inconsistency between diagnoses.
        case.extra["diagnosis"] = action.rationale[:40]
        case.extra["root_cause_hypothesis"] = action.rationale[:40]
        world.record_change("action", action.action_id, f"diagnosed {case.case_id}")


class RemediationAgent(Agent):
    role = "remediation"
    system_prompt = "You apply fixes to live incidents."

    def observe(self, state: WorldState) -> ObservationView:
        return ObservationView(
            timestep=state.timestep,
            system_load=state.system_load,
            open_case_ids=[k for k, c in state.open_cases.items() if c.extra.get("diagnosis")],
        )

    def _build_action(self, timestep: int, resp) -> Action:  # type: ignore[override]
        kind_map = {"remediate": "remediate", "no_op": "no_op"}
        return Action(
            timestep=timestep,
            agent_name=self.name,
            kind=kind_map.get(resp.decision, "no_op"),
            target_case_id=resp.target_case_id,
            rationale=resp.rationale,
        )

    def apply(self, action: Action, world: World) -> None:
        if action.kind != "remediate" or not action.target_case_id:
            return
        case = world.state.open_cases.get(action.target_case_id)
        if not case:
            return
        case.extra["remediated"] = True
        world.adjust_load(-0.05, source="action", source_id=action.action_id)
        world.record_change("action", action.action_id, f"remediated {case.case_id}")
        # Resolved incidents stay around briefly for comms then close.


class CommsAgent(Agent):
    role = "comms"
    system_prompt = "You communicate with customers and stakeholders during incidents."

    def observe(self, state: WorldState) -> ObservationView:
        return ObservationView(
            timestep=state.timestep,
            customer_sentiment=state.customer_sentiment,
            open_case_ids=[k for k, c in state.open_cases.items() if c.extra.get("triaged")],
        )

    def _build_action(self, timestep: int, resp) -> Action:  # type: ignore[override]
        kind_map = {"communicate": "communicate", "no_op": "no_op"}
        return Action(
            timestep=timestep,
            agent_name=self.name,
            kind=kind_map.get(resp.decision, "no_op"),
            target_case_id=resp.target_case_id,
            rationale=resp.rationale,
        )

    def apply(self, action: Action, world: World) -> None:
        if action.kind != "communicate" or not action.target_case_id:
            return
        case = world.state.open_cases.get(action.target_case_id)
        if not case:
            return
        case.extra["comms_sent_at"] = world.state.timestep
        world.adjust_sentiment(+0.02, source="action", source_id=action.action_id)
        world.record_change("action", action.action_id, f"comms on {case.case_id}")
        # Close incidents that have been remediated AND communicated.
        if case.extra.get("remediated"):
            world.remove_case(case.case_id, source="action", source_id=action.action_id)
