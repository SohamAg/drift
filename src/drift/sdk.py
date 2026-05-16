"""BYOA / BYOE public API — let users plug their own multi-agent system into drift.

The shipped topologies (support / code_review / ops) are scaffolding. The
product surface for external users is here: decorate your agents, subclass
WorldState for your domain, optionally define chaos events, call drift.run().

Usage shape (see examples/byoa_minimal.py for a worked end-to-end example):

    import drift

    class MyState(drift.WorldState):
        open_prs: dict = {}

    @drift.agent(role="reviewer")
    async def reviewer(state, memory):
        # user calls their own LLM / tools however they like
        return drift.Action(kind="approve_review", target_case_id="PR-1")

    drift.run(
        agents=[reviewer, ...],
        state=MyState(),
        events=[(5, MyEvent())],
        steps=30,
    )

The decorator pattern matters because drift owns the runtime loop. Drift
calls the user's agents (not the other way around), which is what makes
chaos-event injection possible at all — you can't inject events into a
system you don't drive.
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Iterable

from drift.agents.base import Action, Agent
from drift.events.base import Event, EventRecord
from drift.events.scheduler import EventScheduler, Scenario
from drift.failures.detectors import GENERAL_DETECTORS
from drift.memory import AgentMemory
from drift.simulation import RunResult, SimulationRunner
from drift.world import World, WorldState


# Type alias for the user's agent function signature.
AgentFunc = Callable[[WorldState, AgentMemory], Awaitable[Action | dict]]


class _BYOAgent(Agent):
    """An Agent built from a user-decorated async function.

    Differs from the shipped Agent subclasses in two ways:
      1. No LLM injection — the user's function makes its own LLM/tool calls.
      2. apply() is a no-op — the BYOA pattern is stateless. World state
         mutations come from chaos events, not agent actions. Detectors fire
         on action patterns + the state changes those events cause.

    Users who need stateful actions (action -> world mutation) should subclass
    drift.Agent directly instead of using the decorator.
    """

    def __init__(self, func: AgentFunc, name: str, role: str, memory_capacity: int = 32) -> None:
        # Note: not calling super().__init__ because we don't have an LLMClient.
        self.name = name
        self.role = role  # instance-level override of the ClassVar
        self.memory = AgentMemory(capacity=memory_capacity)
        self.system_prompt = ""
        self._func = func

    def observe(self, state: WorldState) -> WorldState:
        # BYOA agents see the full state; they slice what they care about.
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
        # Stateless — drop the action into the log, world state stays put.
        # Chaos events are responsible for state changes.
        return None

    def _coerce_to_action(self, result: Any, timestep: int) -> Action:
        """User can return either an Action or a dict shorthand."""
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

    Drift calls it each timestep with the current world state and the agent's
    rolling memory. The function should return a drift.Action (or a dict
    convertible to one).

    Args:
        role: the agent's role label (e.g. "reviewer", "security"). Detectors
              filter by role for some failure-mode rules.
        name: instance name; defaults to the function's __name__. Distinct
              names matter when you have multiple agents of the same role.
        memory_capacity: how many recent observations + actions to retain.

    Returns:
        A _BYOAgent instance ready to be passed to drift.run(agents=[...]).
    """
    def deco(func: AgentFunc) -> _BYOAgent:
        if not asyncio.iscoroutinefunction(func):
            raise TypeError(
                f"@drift.agent requires an async function; {func.__name__!r} is sync. "
                "Use `async def` for your agent."
            )
        return _BYOAgent(
            func=func,
            name=name or func.__name__,
            role=role,
            memory_capacity=memory_capacity,
        )
    return deco


class _InlineScheduler(EventScheduler):
    """Scheduler that emits a fixed list of (timestep, event_instance) pairs.

    Skips the YAML + registry indirection — useful when users define events
    in Python and want to inject them at known timesteps without writing a
    scenario file. Stochastic events are not supported here.
    """

    def __init__(self, events_by_step: dict[int, list[Event]], seed: int = 0) -> None:
        # Bypass the parent constructor's registry resolution.
        self.scenario = Scenario()
        self._rng = None
        self._registry = {}
        self._scripted_by_step = {}
        self._events_by_step = events_by_step

    def tick(self, timestep: int) -> list[Event]:
        # Return fresh copies of the events at this step; the runner will call
        # apply() on each. We don't deep-copy because Events are stateless
        # apart from their event_id which is set in __init__ — but since the
        # user passes pre-built instances, just return them once.
        return list(self._events_by_step.get(timestep, []))


def _build_runner(
    agents: Iterable[_BYOAgent | Agent],
    state: WorldState | None,
    events: Iterable[tuple[int, Event]] | None,
    steps: int,
    seed: int,
    detectors: Iterable | None,
) -> SimulationRunner:
    agent_list = list(agents)
    if not agent_list:
        raise ValueError("drift.run requires at least one agent")

    initial_world = World(initial=state if state is not None else WorldState())

    events_by_step: dict[int, list[Event]] = {}
    for t, ev in (events or []):
        events_by_step.setdefault(int(t), []).append(ev)
    scheduler = _InlineScheduler(events_by_step=events_by_step, seed=seed)

    detector_list = list(detectors) if detectors is not None else list(GENERAL_DETECTORS)

    return SimulationRunner(
        agents=agent_list,
        scheduler=scheduler,
        steps=steps,
        detectors=detector_list,
        logger=None,
        initial_world=initial_world,
    )


async def run_async(
    *,
    agents: Iterable[_BYOAgent | Agent],
    state: WorldState | None = None,
    events: Iterable[tuple[int, Event]] | None = None,
    steps: int = 30,
    seed: int = 42,
    detectors: Iterable | None = None,
) -> RunResult:
    """Async version of drift.run(). Use this when calling from inside an
    already-running event loop (e.g., a FastAPI endpoint handler)."""
    runner = _build_runner(agents, state, events, steps, seed, detectors)
    return await runner.run()


def run(
    *,
    agents: Iterable[_BYOAgent | Agent],
    state: WorldState | None = None,
    events: Iterable[tuple[int, Event]] | None = None,
    steps: int = 30,
    seed: int = 42,
    detectors: Iterable | None = None,
) -> RunResult:
    """Run drift's simulator with user-supplied agents, state, and events.

    Args:
        agents:    list of @drift.agent-decorated functions (or Agent subclass
                   instances for power users).
        state:     a drift.WorldState instance (subclass it to add your own
                   fields). Defaults to a bare WorldState if omitted.
        events:    optional list of (timestep, event_instance) pairs. Drift
                   applies each event at the given timestep. Skip for a
                   pure-decisions-only run with no chaos.
        steps:     how many timesteps to simulate.
        seed:      RNG seed for any stochastic agent behavior.
        detectors: which detectors to run. Defaults to GENERAL_DETECTORS
                   (the cross-topology ones — sentiment_collapse,
                   hallucinated_reference, stale_snapshot_reference,
                   queue_explosion). Pass an explicit list to use
                   domain-specific detectors.

    Returns:
        drift.RunResult with .actions, .events, .failures, .final_state.

    Notes:
        This calls asyncio.run() internally; do not call from inside an
        already-running event loop. Use drift.run_async() in that case.
    """
    runner = _build_runner(agents, state, events, steps, seed, detectors)
    return asyncio.run(runner.run())


__all__ = [
    "Action",
    "Agent",
    "Event",
    "EventRecord",
    "WorldState",
    "World",
    "agent",
    "run",
    "run_async",
]
