"""Lightweight metrics tracker.

Used to populate the final-report behavior summaries. Plain counters and
per-agent action histograms — no time series.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from drift.agents.base import Action
from drift.failures.base import FailureRecord


@dataclass
class Metrics:
    actions_by_agent: Counter = field(default_factory=Counter)
    actions_by_kind: Counter = field(default_factory=Counter)
    per_agent_kinds: dict[str, Counter] = field(default_factory=dict)
    failures_by_type: Counter = field(default_factory=Counter)
    events_by_name: Counter = field(default_factory=Counter)

    def record_action(self, action: Action) -> None:
        self.actions_by_agent[action.agent_name] += 1
        self.actions_by_kind[action.kind] += 1
        self.per_agent_kinds.setdefault(action.agent_name, Counter())[action.kind] += 1

    def record_failure(self, failure: FailureRecord) -> None:
        self.failures_by_type[failure.failure_type] += 1

    def record_event(self, name: str) -> None:
        self.events_by_name[name] += 1

    def agent_summary(self, agent_name: str) -> str:
        kinds = self.per_agent_kinds.get(agent_name, Counter())
        if not kinds:
            return f"{agent_name}: no actions"
        parts = [f"{k}={v}" for k, v in kinds.most_common()]
        return f"{agent_name}: " + ", ".join(parts)
