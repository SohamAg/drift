"""Failure detection abstractions."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, Field

from drift.agents.base import Action
from drift.events.base import EventRecord
from drift.world import WorldHistory

_failure_counter = itertools.count(1)


def _new_failure_id() -> str:
    return f"f{next(_failure_counter):06d}"


def reset_failure_counter() -> None:
    global _failure_counter
    _failure_counter = itertools.count(1)


class FailureRecord(BaseModel):
    failure_id: str = Field(default_factory=_new_failure_id)
    timestep: int
    failure_type: str
    agents_involved: list[str] = Field(default_factory=list)
    evidence_action_ids: list[str] = Field(default_factory=list)
    summary: str
    snapshot_timestep: int  # which world snapshot the failure references


@dataclass
class DetectorContext:
    """Inputs each detector gets. Detectors are stateless functions over this."""
    timestep: int
    history: WorldHistory
    actions: list[Action]
    events: list[EventRecord]
    already_reported: set[str]  # set of "type:fingerprint" entries to avoid duplicate fires


Detector = Callable[[DetectorContext], list[FailureRecord]]
