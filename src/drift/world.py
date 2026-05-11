"""World state — the single source of mutable truth in a simulation.

All state changes flow through `WorldState.apply(...)`, which appends a
snapshot to history. This is the only mutation point so the audit trail
stays clean.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field

CaseId = str


class Case(BaseModel):
    """A unit of work being processed (a support case, a PR, an incident).

    Topologies attach domain-specific data via the `extra` dict rather than
    by subclassing — keeps the World API uniform across topologies.
    """
    model_config = ConfigDict(extra="allow")

    case_id: CaseId
    customer_id: str = ""
    issue: str = ""
    refund_requested: bool = False
    opened_at_step: int = 0
    escalation_count: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


class CaseRef(BaseModel):
    """Lightweight pointer used inside the escalation queue."""
    case_id: CaseId
    enqueued_at_step: int


class WorldState(BaseModel):
    """Mutable simulation state. Topologies may add extra fields.

    Common fields (timestep, system_load, open_cases, escalation_queue)
    work across all topologies; domain-specific fields go in `extra` or
    are added directly via Pydantic's `extra="allow"`.
    """
    model_config = ConfigDict(extra="allow")

    timestep: int = 0
    customer_sentiment: float = Field(default=0.7, ge=0.0, le=1.0)
    refund_policy_version: int = 1
    inventory_delay_minutes: int = 0
    system_load: float = Field(default=0.3, ge=0.0, le=1.0)
    escalation_queue: list[CaseRef] = Field(default_factory=list)
    open_cases: dict[CaseId, Case] = Field(default_factory=dict)


class WorldChange(BaseModel):
    """Records a delta applied to the world. Stored alongside snapshots."""
    timestep: int
    source: Literal["event", "action", "tick"]
    source_id: str
    summary: str


class WorldHistory:
    """Bounded ring of (snapshot, changes_at_step). In-memory only."""

    def __init__(self, maxlen: int = 1024) -> None:
        self._snapshots: Deque[WorldState] = deque(maxlen=maxlen)
        self._changes: Deque[list[WorldChange]] = deque(maxlen=maxlen)

    def record(self, state: WorldState, changes: list[WorldChange]) -> None:
        self._snapshots.append(state.model_copy(deep=True))
        self._changes.append(list(changes))

    def latest(self) -> WorldState | None:
        return self._snapshots[-1] if self._snapshots else None

    def at(self, timestep: int) -> WorldState | None:
        for snap in self._snapshots:
            if snap.timestep == timestep:
                return snap
        return None

    def window(self, steps: int) -> list[WorldState]:
        if steps <= 0:
            return []
        return list(self._snapshots)[-steps:]

    def all_snapshots(self) -> Iterable[WorldState]:
        return iter(self._snapshots)

    def __len__(self) -> int:
        return len(self._snapshots)


class World:
    """Wraps WorldState with a controlled mutation API."""

    def __init__(self, initial: WorldState | None = None, history_size: int = 1024) -> None:
        self.state = initial or WorldState()
        self.history = WorldHistory(maxlen=history_size)
        self._pending_changes: list[WorldChange] = []

    def begin_step(self, timestep: int) -> None:
        self.state.timestep = timestep
        self._pending_changes.clear()

    def commit_step(self) -> None:
        self.history.record(self.state, self._pending_changes)
        self._pending_changes.clear()

    def record_change(self, source: Literal["event", "action", "tick"], source_id: str, summary: str) -> None:
        self._pending_changes.append(
            WorldChange(timestep=self.state.timestep, source=source, source_id=source_id, summary=summary)
        )

    # Mutation helpers — agents/events call these instead of touching fields directly.
    def adjust_sentiment(self, delta: float, *, source: str, source_id: str) -> None:
        new = max(0.0, min(1.0, self.state.customer_sentiment + delta))
        self.record_change(source, source_id, f"sentiment {self.state.customer_sentiment:.2f}->{new:.2f}")  # type: ignore[arg-type]
        self.state.customer_sentiment = new

    def set_policy_version(self, version: int, *, source: str, source_id: str) -> None:
        self.record_change(source, source_id, f"policy {self.state.refund_policy_version}->{version}")  # type: ignore[arg-type]
        self.state.refund_policy_version = version

    def adjust_inventory_delay(self, minutes: int, *, source: str, source_id: str) -> None:
        new = max(0, self.state.inventory_delay_minutes + minutes)
        self.record_change(source, source_id, f"inventory_delay {self.state.inventory_delay_minutes}->{new}")  # type: ignore[arg-type]
        self.state.inventory_delay_minutes = new

    def adjust_load(self, delta: float, *, source: str, source_id: str) -> None:
        new = max(0.0, min(1.0, self.state.system_load + delta))
        self.record_change(source, source_id, f"load {self.state.system_load:.2f}->{new:.2f}")  # type: ignore[arg-type]
        self.state.system_load = new

    def add_case(self, case: Case, *, source: str, source_id: str) -> None:
        self.state.open_cases[case.case_id] = case
        self.record_change(source, source_id, f"case+ {case.case_id}")  # type: ignore[arg-type]

    def remove_case(self, case_id: CaseId, *, source: str, source_id: str) -> None:
        if case_id in self.state.open_cases:
            del self.state.open_cases[case_id]
            self.record_change(source, source_id, f"case- {case_id}")  # type: ignore[arg-type]
        self.state.escalation_queue = [r for r in self.state.escalation_queue if r.case_id != case_id]

    def enqueue_escalation(self, case_id: CaseId, *, source: str, source_id: str) -> None:
        self.state.escalation_queue.append(CaseRef(case_id=case_id, enqueued_at_step=self.state.timestep))
        if case_id in self.state.open_cases:
            self.state.open_cases[case_id].escalation_count += 1
        self.record_change(source, source_id, f"escalate {case_id}")  # type: ignore[arg-type]

    def dequeue_escalation(self, *, source: str, source_id: str) -> CaseRef | None:
        if not self.state.escalation_queue:
            return None
        ref = self.state.escalation_queue.pop(0)
        self.record_change(source, source_id, f"dequeue {ref.case_id}")  # type: ignore[arg-type]
        return ref
