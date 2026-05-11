"""Agent abstractions.

Each agent has three hooks:
  - observe(world)   -> a slice of world state relevant to its role
  - decide(...)      -> async, calls the LLM, emits an Action
  - update_memory    -> append outcome to its bounded memory

Actions are typed Pydantic models. Detectors read action fields directly,
which is why we don't hand around free-form text.
"""
from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from drift.llm.base import LLMClient
from drift.memory import AgentMemory
from drift.world import World, WorldState

# ActionKind is a free-form string so each topology can declare its own
# vocabulary (e.g. "merge", "remediate", "diagnose"). Detectors filter by
# kind names; agents are responsible for using the right ones.
ActionKind = str

_action_counter = itertools.count(1)


def reset_action_counter() -> None:
    """Reset the module-level action ID counter — for tests/sweeps."""
    global _action_counter
    _action_counter = itertools.count(1)


class Action(BaseModel):
    action_id: str = Field(default_factory=lambda: f"a{next(_action_counter):06d}")
    timestep: int
    agent_name: str
    kind: ActionKind
    target_case_id: str | None = None
    rationale: str = ""
    referenced_policy_version: int | None = None


class ObservationView(BaseModel):
    """What an agent sees this step. Roles can shadow fields they shouldn't see."""
    timestep: int
    customer_sentiment: float | None = None
    refund_policy_version: int | None = None
    inventory_delay_minutes: int | None = None
    system_load: float | None = None
    open_case_ids: list[str] = Field(default_factory=list)
    queue_case_ids: list[str] = Field(default_factory=list)


class Agent(ABC):
    role: ClassVar[str] = "base"
    system_prompt: ClassVar[str] = "You are an agent in a simulation."

    def __init__(self, name: str, llm: LLMClient, memory_capacity: int = 32) -> None:
        self.name = name
        self.llm = llm
        self.memory = AgentMemory(capacity=memory_capacity)

    @abstractmethod
    def observe(self, state: WorldState) -> ObservationView: ...

    def _ctx(self, obs: ObservationView) -> dict[str, Any]:
        d = obs.model_dump()
        d["agent_role"] = self.role
        return d

    async def step(self, world: World) -> Action:
        obs = self.observe(world.state)
        self.memory.remember(obs.timestep, "observation", obs.model_dump_json(exclude_none=True))
        resp = await self.llm.generate(system=self.system_prompt, user=self._user_prompt(obs), ctx=self._ctx(obs))
        action = self._build_action(obs.timestep, resp)
        self.memory.remember(obs.timestep, "action", f"{action.kind} {action.target_case_id or ''} :: {action.rationale}")
        return action

    def _user_prompt(self, obs: ObservationView) -> str:
        return f"World snapshot at t={obs.timestep}:\n{obs.model_dump_json(exclude_none=True)}\n{self.memory.render()}"

    def _build_action(self, timestep: int, resp: Any) -> Action:
        return Action(
            timestep=timestep,
            agent_name=self.name,
            kind=resp.decision,  # type: ignore[arg-type]
            target_case_id=resp.target_case_id,
            rationale=resp.rationale,
            referenced_policy_version=resp.referenced_policy_version,
        )

    @abstractmethod
    def apply(self, action: Action, world: World) -> None:
        """Translate an action into world mutations. Called sequentially after all agents decide."""
