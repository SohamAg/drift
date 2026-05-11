"""Ops/incident-response topology.

4 agents triage and respond to evolving production incidents:
  - TriageAgent       — classifies severity and assigns
  - DiagnosisAgent    — proposes a root-cause hypothesis
  - RemediationAgent  — applies fixes (rollback, restart, scale, etc.)
  - CommsAgent        — communicates internally and externally

Failure modes specific to this topology:
  - silent_remediation: a fix was applied but no comms went out for N steps
  - comms_lag: high-severity incident with no comms for N steps after triage
  - contradictory_diagnosis: two diagnoses on the same incident name different root causes

Plus the general detectors (sentiment_collapse — public/customer trust here,
queue_explosion — incident backlog, hallucinated_reference, stale_snapshot).
"""
from __future__ import annotations

from drift.failures.detectors import GENERAL_DETECTORS
from drift.llm.base import LLMClient
from drift.topologies import Topology, register
from drift.topologies.ops.agents import (
    CommsAgent,
    DiagnosisAgent,
    RemediationAgent,
    TriageAgent,
)
from drift.topologies.ops.detectors import OPS_DETECTORS
from drift.topologies.ops.events import OPS_EVENTS
from drift.topologies.ops.mock import OPS_MOCK_HANDLERS
from drift.topologies.ops.prompts import OPS_PROMPTS
from drift.world import World, WorldState


def _agents(llm: LLMClient):
    return [
        TriageAgent(name="triage", llm=llm),
        DiagnosisAgent(name="diagnosis", llm=llm),
        RemediationAgent(name="remediation", llm=llm),
        CommsAgent(name="comms", llm=llm),
    ]


def _initial_world() -> World:
    initial = WorldState(
        customer_sentiment=0.8,   # public trust / status-page reputation
        system_load=0.3,          # ops on-call workload
        refund_policy_version=1,  # unused
    )
    return World(initial=initial)


OPS = Topology(
    name="ops",
    description="4 agents triaging, diagnosing, fixing, and communicating production incidents.",
    agent_factory=_agents,
    event_registry=OPS_EVENTS,
    detectors=GENERAL_DETECTORS + OPS_DETECTORS,
    prompts=OPS_PROMPTS,
    mock_handlers=OPS_MOCK_HANDLERS,
    initial_world=_initial_world,
)

register(OPS)
