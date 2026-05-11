"""Event scheduler.

Combines:
  - a scripted timeline loaded from a YAML scenario file
  - a stochastic injector that fires events at a configurable rate

Both run every timestep. The scheduler is seedable so runs are reproducible.
The event registry is supplied per-topology — `EVENT_REGISTRY` is the
default support-topology fallback for backward compat with old code paths.
"""
from __future__ import annotations

import random
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from drift.events.base import Event
from drift.events.library import EVENT_REGISTRY


class StochasticEntry(BaseModel):
    name: str
    probability: float = Field(ge=0.0, le=1.0)


class ScriptedEntry(BaseModel):
    timestep: int
    name: str


class Scenario(BaseModel):
    name: str = "unnamed"
    scripted: list[ScriptedEntry] = Field(default_factory=list)
    stochastic: list[StochasticEntry] = Field(default_factory=list)


class EventScheduler:
    def __init__(
        self,
        scenario: Scenario,
        seed: int = 0,
        event_registry: dict[str, type[Event]] | None = None,
    ) -> None:
        self.scenario = scenario
        self._rng = random.Random(seed)
        self._registry = event_registry if event_registry is not None else EVENT_REGISTRY
        self._scripted_by_step: dict[int, list[str]] = {}
        for entry in scenario.scripted:
            self._scripted_by_step.setdefault(entry.timestep, []).append(entry.name)

    @classmethod
    def from_yaml(
        cls,
        path: Path | str,
        seed: int = 0,
        event_registry: dict[str, type[Event]] | None = None,
    ) -> "EventScheduler":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(Scenario(**data), seed=seed, event_registry=event_registry)

    @classmethod
    def empty(
        cls,
        seed: int = 0,
        event_registry: dict[str, type[Event]] | None = None,
    ) -> "EventScheduler":
        return cls(Scenario(), seed=seed, event_registry=event_registry)

    def tick(self, timestep: int) -> list[Event]:
        out: list[Event] = []
        for name in self._scripted_by_step.get(timestep, []):
            cls = self._registry.get(name)
            if cls is None:
                continue
            out.append(cls())
        for entry in self.scenario.stochastic:
            if self._rng.random() < entry.probability:
                cls = self._registry.get(entry.name)
                if cls is None:
                    continue
                out.append(cls())
        return out
