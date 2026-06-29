"""BYOA SDK — `@drift.agent` decorator (currently in transition).

History: this module used to wrap a per-tick native simulator runtime so
users could decorate async functions as drift agents and call `drift.run()`
to simulate them. After the 2026-06-29 cleanup the native simulator was
removed; the decorator and `_BYOAgent` data class survive as the BYOA
surface, but the runtime path (`drift.run` / `drift.run_async`) is removed.

Re-implementation TODO: a `@drift.agent`-shaped function should be
wrappable into a LangGraph-shaped object that drift's adapter
(`drift.adapters.langgraph.drift_test`) can run against. That re-wire is
tracked in memory + FUTURE_DIRECTIONS but is NOT part of this cleanup.

Until then, the decorator exists for users to mark agents in their code;
they can be plugged into a graph manually or wait for the adapter rewire.
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from drift.agents.base import Action, Agent, ObservationView
from drift.memory import AgentMemory
from drift.world import World, WorldState


AgentFunc = Callable[[WorldState, AgentMemory], Awaitable[Action | dict]]


class _BYOAgent(Agent):
    """An Agent built from a user-decorated async function.

    The decorated function makes its own LLM/tool calls; this wrapper handles
    observation, memory, and action coercion. Stateless on the drift side —
    state mutations are owned by the user's graph runtime (when wired into
    an adapter), not by this wrapper.
    """

    def __init__(self, func: AgentFunc, name: str, role: str, memory_capacity: int = 32) -> None:
        # Note: not calling super().__init__ because we don't have an LLMClient.
        self.name = name
        self.role = role  # instance-level override of the ClassVar
        self.memory = AgentMemory(capacity=memory_capacity)
        self.system_prompt = ""
        self._func = func

    def observe(self, state: WorldState) -> WorldState:
        return state.model_copy(deep=True)

    async def step(self, world: World) -> Action:
        obs = self.observe(world.state)
        self.memory.remember(
            world.state.timestep,
            "observation",
            obs.model_dump_json(exclude_none=True)[:500],
        )
        result = await self._func(obs, self.memory)
        action = self._coerce_to_action(result, world.state.timestep)
        self.memory.remember(
            world.state.timestep,
            "action",
            f"{action.kind} {action.target_case_id or ''} :: {action.rationale or ''}",
        )
        return action

    def apply(self, action: Action, world: World) -> None:
        return None

    def _coerce_to_action(self, result: Any, timestep: int) -> Action:
        if isinstance(result, Action):
            return result.model_copy(update={
                "agent_name": self.name,
                "timestep": timestep,
            })
        if isinstance(result, dict):
            payload = {**result, "agent_name": self.name, "timestep": timestep}
            return Action.model_validate(payload)
        raise TypeError(
            f"@drift.agent {self.name!r} returned {type(result).__name__}; "
            "expected drift.Action or dict"
        )


def agent(role: str, *, name: str | None = None, memory_capacity: int = 32):
    """Decorator: turn an async function into a drift Agent instance.

    The decorated function must be async with signature:
        async def my_agent(state: WorldState, memory: AgentMemory) -> Action

    NOTE: as of the 2026-06-29 cleanup, drift.run() no longer exists. The
    decorator survives as a data shape so downstream code can identify
    drift-decorated agents and (in a future re-wire) plug them into the
    LangGraph adapter. Don't expect a working end-to-end runtime from
    `@drift.agent` today.
    """
    def deco(func: AgentFunc) -> _BYOAgent:
        if not asyncio.iscoroutinefunction(func):
            raise TypeError(
                f"@drift.agent requires an async function; {func.__name__!r} is sync."
            )
        return _BYOAgent(
            func=func,
            name=name or func.__name__,
            role=role,
            memory_capacity=memory_capacity,
        )
    return deco


__all__ = ["Action", "Agent", "ObservationView", "agent"]
