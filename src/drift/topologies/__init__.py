"""Topology = a self-contained bundle that turns drift into a simulator
for one specific multi-agent system shape.

A topology brings:
  - agent factory (how to construct the agents that participate)
  - event registry (the exogenous events that can fire in this domain)
  - detectors (failure modes specific to the topology, plus general ones)
  - prompts (naive + hardened variants per role)
  - mock-LLM role handlers (so demos work without API keys)
  - initial world factory (any topology-specific WorldState fields)

Each topology is a self-contained Python module under `drift.topologies`.
The CLI selects one via `--topology`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from drift.agents.base import Agent
from drift.events.base import Event
from drift.failures.base import Detector
from drift.llm.base import LLMClient
from drift.llm.mock import RoleHandler
from drift.world import World


@dataclass
class Topology:
    name: str
    description: str
    agent_factory: Callable[[LLMClient], list[Agent]]
    event_registry: dict[str, type[Event]]
    detectors: list[Detector]
    prompts: dict[tuple[str, str], str]
    mock_handlers: dict[str, RoleHandler]
    initial_world: Callable[[], World]


# Populated by importing each topology module below. Done lazily inside
# `get_topology()` so failures in one topology module don't break the others.
_TOPOLOGIES: dict[str, Topology] = {}


def register(topology: Topology) -> None:
    _TOPOLOGIES[topology.name] = topology


def get_topology(name: str) -> Topology:
    if not _TOPOLOGIES:
        _load_all()
    if name not in _TOPOLOGIES:
        raise KeyError(
            f"unknown topology {name!r}. Available: {sorted(_TOPOLOGIES.keys())}"
        )
    return _TOPOLOGIES[name]


def list_topologies() -> list[str]:
    if not _TOPOLOGIES:
        _load_all()
    return sorted(_TOPOLOGIES.keys())


def _load_all() -> None:
    # Import each topology so it self-registers.
    from drift.topologies import support  # noqa: F401
    from drift.topologies import code_review  # noqa: F401
    from drift.topologies import ops  # noqa: F401
