"""Customer-support topology — bundles the original 4 agents."""
from __future__ import annotations

from drift.agents import EscalationAgent, PolicyAgent, RefundAgent, SupportAgent
from drift.agents.prompts import PROMPTS
from drift.events.library import EVENT_REGISTRY
from drift.failures.detectors import GENERAL_DETECTORS, SUPPORT_DETECTORS
from drift.llm.base import LLMClient
from drift.topologies import Topology, register
from drift.world import World, WorldState


def _agents(llm: LLMClient):
    return [
        SupportAgent(name="support", llm=llm),
        RefundAgent(name="refund", llm=llm),
        EscalationAgent(name="escalation", llm=llm),
        PolicyAgent(name="policy", llm=llm),
    ]


def _initial_world() -> World:
    return World(initial=WorldState())


# The mock handlers for support live on ScriptedMockLLM as defaults
# (kept there for back-compat). Empty dict means "use the defaults."
_MOCK_HANDLERS: dict = {}


SUPPORT = Topology(
    name="support",
    description="Customer-support: 4 agents handling cases, refunds, escalation, and policy.",
    agent_factory=_agents,
    event_registry=EVENT_REGISTRY,
    detectors=GENERAL_DETECTORS + SUPPORT_DETECTORS,
    prompts=PROMPTS,
    mock_handlers=_MOCK_HANDLERS,
    initial_world=_initial_world,
)

register(SUPPORT)
