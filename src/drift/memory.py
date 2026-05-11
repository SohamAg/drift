"""Lightweight bounded memory for agents."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque


@dataclass
class MemoryEntry:
    timestep: int
    kind: str  # "observation" | "action" | "outcome" | "note"
    content: str


@dataclass
class AgentMemory:
    """Bounded log of recent entries plus a free-form scratchpad.

    The cap matters: an unbounded memory wouldn't surface drift, because
    everything would always be in scope. A bounded window mirrors the
    real situation where context is limited and agents forget.
    """

    capacity: int = 32
    log: Deque[MemoryEntry] = field(default_factory=deque)
    scratchpad: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.log = deque(self.log, maxlen=self.capacity)

    def remember(self, timestep: int, kind: str, content: str) -> None:
        self.log.append(MemoryEntry(timestep=timestep, kind=kind, content=content))

    def recent(self, n: int | None = None) -> list[MemoryEntry]:
        items = list(self.log)
        return items if n is None else items[-n:]

    def render(self, n: int = 8) -> str:
        return "\n".join(f"[t={e.timestep}] {e.kind}: {e.content}" for e in self.recent(n))
