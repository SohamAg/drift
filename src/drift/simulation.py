"""Simulation runner.

Per-timestep loop:
  1. scheduler.tick(t)               -> events; each event.apply(world)
  2. asyncio.gather(agents.step)     -> all agents observe+decide concurrently
  3. for action in actions:           -> sequential merge (deterministic order)
       agent.apply(action, world)
  4. detectors run                    -> append new failures
  5. logger writes per-step records
  6. metrics updated

Concurrent observe-decide reflects the property we're modeling — agents
act on the *same* world snapshot. Sequential application keeps mutation
deterministic.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from drift.agents.base import Action, Agent
from drift.events.base import EventRecord
from drift.events.scheduler import EventScheduler
from drift.failures.base import Detector, DetectorContext, FailureRecord
from drift.failures.detectors import ALL_DETECTORS
from drift.observability.logger import RunLogger
from drift.observability.metrics import Metrics
from drift.world import World, WorldState


@dataclass
class RunResult:
    final_state: WorldState
    events: list[EventRecord] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    failures: list[FailureRecord] = field(default_factory=list)
    metrics: Metrics = field(default_factory=Metrics)
    run_dir: Path | None = None
    # EventRecords that came from drift's auto-chaos engine (a subset of
    # `events`). Populated by drift.sdk.run when auto_chaos is enabled;
    # always empty otherwise. Names are prefixed `AutoChaos.<pattern>[field]`.
    auto_chaos_injected: list[EventRecord] = field(default_factory=list)


class SimulationRunner:
    def __init__(
        self,
        agents: list[Agent],
        scheduler: EventScheduler,
        steps: int = 50,
        detectors: Iterable[Detector] | None = None,
        logger: RunLogger | None = None,
        initial_world: World | None = None,
        start_step: int = 0,
        disabled_agents: Iterable[str] | None = None,
    ) -> None:
        self.agents = agents
        self.scheduler = scheduler
        self.steps = steps
        self.detectors = list(detectors) if detectors is not None else list(ALL_DETECTORS)
        self.logger = logger
        self.world = initial_world or World()
        self.metrics = Metrics()
        self.events: list[EventRecord] = []
        self.actions: list[Action] = []
        self.failures: list[FailureRecord] = []
        self._reported: set[str] = set()
        # Fork support: loop runs (start_step, start_step + steps]. The world's
        # timestep is reset on each begin_step so an injected initial world
        # gets advanced properly.
        self.start_step = start_step
        self.disabled_agents: set[str] = set(disabled_agents or ())

    async def run(self) -> RunResult:
        start = self.start_step + 1
        end = self.start_step + self.steps + 1
        for t in range(start, end):
            await self._tick(t)

        return RunResult(
            final_state=self.world.state.model_copy(deep=True),
            events=list(self.events),
            actions=list(self.actions),
            failures=list(self.failures),
            metrics=self.metrics,
            run_dir=self.logger.run_dir if self.logger else None,
        )

    async def _tick(self, t: int) -> None:
        self.world.begin_step(t)

        # 1. events
        for event in self.scheduler.tick(t):
            record = event.apply(self.world)
            self.events.append(record)
            self.metrics.record_event(record.name)
            if self.logger:
                self.logger.log_event(record)

        # 2. agents observe + decide concurrently (snapshot-of-world view).
        # Disabled agents are skipped entirely — they don't observe, decide, or act.
        active = [a for a in self.agents if a.name not in self.disabled_agents]
        actions: list[Action] = await asyncio.gather(*(a.step(self.world) for a in active))

        # 3. apply actions sequentially in agent-name order so runs stay deterministic
        for action, agent in sorted(zip(actions, active), key=lambda pair: pair[0].agent_name):
            agent.apply(action, self.world)
            self.actions.append(action)
            self.metrics.record_action(action)
            if self.logger:
                self.logger.log_action(action)

        # commit snapshot before detectors so they see the post-step world
        self.world.commit_step()
        if self.logger:
            self.logger.log_snapshot(self.world.state)

        # 4. detectors
        ctx = DetectorContext(
            timestep=t,
            history=self.world.history,
            actions=self.actions,
            events=self.events,
            already_reported=self._reported,
        )
        for detector in self.detectors:
            for failure in detector(ctx):
                self.failures.append(failure)
                self.metrics.record_failure(failure)
                if self.logger:
                    self.logger.log_failure(failure)
