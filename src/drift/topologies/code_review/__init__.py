"""Code-review topology.

4 agents collaborate on a stream of pull requests under deadline + CVE pressure:
  - ProposerAgent  — opens PRs (engineers under pressure to ship)
  - ReviewerAgent  — approves or rejects PRs
  - SecurityAgent  — blocks PRs with security concerns; clears them after review
  - MergeAgent     — merges PRs that are ready

Failure modes this topology surfaces:
  - contradictory_review: same PR gets both approve and reject from different reviewers
  - security_bypass: PR merged while security had it blocked
  - merge_without_approval: PR merged with no approvals

Plus the general detectors (hallucinated_reference, stale_snapshot_reference,
queue_explosion, sentiment_collapse — sentiment here represents team morale).
"""
from __future__ import annotations

from drift.failures.detectors import GENERAL_DETECTORS
from drift.llm.base import LLMClient
from drift.topologies import Topology, register
from drift.topologies.code_review.agents import (
    MergeAgent,
    ProposerAgent,
    ReviewerAgent,
    SecurityAgent,
)
from drift.topologies.code_review.detectors import CODE_REVIEW_DETECTORS
from drift.topologies.code_review.events import CODE_REVIEW_EVENTS
from drift.topologies.code_review.mock import CODE_REVIEW_MOCK_HANDLERS
from drift.topologies.code_review.prompts import CODE_REVIEW_PROMPTS
from drift.world import World, WorldState


def _agents(llm: LLMClient):
    return [
        ProposerAgent(name="proposer", llm=llm),
        ReviewerAgent(name="reviewer", llm=llm),
        SecurityAgent(name="security", llm=llm),
        MergeAgent(name="merge", llm=llm),
    ]


def _initial_world() -> World:
    # Re-purpose existing fields where natural; topology-specific state lives
    # on each PR's Case.extra dict and on WorldState's extras (deadline_pressure).
    initial = WorldState(
        customer_sentiment=0.7,    # team morale
        system_load=0.3,           # CI/review queue load
        refund_policy_version=1,   # not used; left at default
    )
    initial.deadline_pressure = 0.2  # extra field, allowed by ConfigDict(extra="allow")
    return World(initial=initial)


CODE_REVIEW = Topology(
    name="code_review",
    description="4 agents reviewing/merging PRs under deadline and security pressure.",
    agent_factory=_agents,
    event_registry=CODE_REVIEW_EVENTS,
    detectors=GENERAL_DETECTORS + CODE_REVIEW_DETECTORS,
    prompts=CODE_REVIEW_PROMPTS,
    mock_handlers=CODE_REVIEW_MOCK_HANDLERS,
    initial_world=_initial_world,
)

register(CODE_REVIEW)
