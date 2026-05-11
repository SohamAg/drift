"""Scripted mock LLM.

The mock is intentionally flaky. Without flakiness, the detectors would
never fire — the whole point of the simulator is to surface emergent
failures, so the mock must occasionally:
  - cite a stale policy version (drift)
  - reference a case_id that isn't open (hallucination)
  - contradict its own prior decision on a case (contradiction)
  - bounce cases back to escalation under load (loops)

All randomness flows through a seeded `random.Random` so runs are
reproducible. Topologies provide their own role handlers via
`role_handlers={role: callable(rng, prior_decisions, ctx) -> LLMResponse}`.
The default support-topology handlers are kept as methods on this class
for backward compatibility.
"""
from __future__ import annotations

import random
from typing import Any, Callable

from drift.llm.base import LLMClient, LLMResponse

# A role handler is a function the mock calls for each agent role.
# Signature: (rng, prior_decisions_by_target, ctx) -> LLMResponse
RoleHandler = Callable[[random.Random, dict[str, str], dict[str, Any]], LLMResponse]


class ScriptedMockLLM(LLMClient):
    def __init__(
        self,
        seed: int = 0,
        role_handlers: dict[str, RoleHandler] | None = None,
    ) -> None:
        self._rng = random.Random(seed)
        self._prior_decisions: dict[str, str] = {}
        self._role_handlers: dict[str, RoleHandler] = role_handlers or {}

    async def generate(self, *, system: str, user: str, ctx: dict[str, Any]) -> LLMResponse:
        role = ctx.get("agent_role", "unknown")
        if role in self._role_handlers:
            return self._role_handlers[role](self._rng, self._prior_decisions, ctx)
        # Default fallbacks for the support topology.
        if role == "support":
            return self._support(ctx)
        if role == "refund":
            return self._refund(ctx)
        if role == "escalation":
            return self._escalation(ctx)
        if role == "policy":
            return self._policy(ctx)
        return LLMResponse(decision="no_op", rationale=f"unknown role {role}")

    # --- per-role behavior --------------------------------------------------

    def _support(self, ctx: dict[str, Any]) -> LLMResponse:
        load = ctx.get("system_load", 0.0)
        sentiment = ctx.get("customer_sentiment", 0.7)
        open_cases: list[str] = ctx.get("open_case_ids", [])

        if not open_cases:
            return LLMResponse(decision="no_op", rationale="no cases to triage")

        target = self._rng.choice(open_cases)

        # Hallucination: occasionally invent a case_id by mutating the chosen one.
        if self._rng.random() < 0.04:
            target = target + "_X"

        # Under high load or low sentiment, lean toward escalating.
        escalate_p = 0.2 + max(0.0, load - 0.5) + max(0.0, 0.5 - sentiment) * 0.5
        if self._rng.random() < min(escalate_p, 0.85):
            return LLMResponse(
                decision="escalate",
                rationale=f"load={load:.2f} sentiment={sentiment:.2f} — needs escalation",
                target_case_id=target,
            )
        return LLMResponse(
            decision="respond",
            rationale="standard triage",
            target_case_id=target,
        )

    def _refund(self, ctx: dict[str, Any]) -> LLMResponse:
        sentiment = ctx.get("customer_sentiment", 0.7)
        policy_version = ctx.get("refund_policy_version", 1)
        open_cases: list[str] = ctx.get("open_case_ids", [])

        if not open_cases:
            return LLMResponse(decision="no_op", rationale="nothing to refund")

        target = self._rng.choice(open_cases)

        # Drift: 12% chance of citing a stale policy version (off by one).
        cited_version = policy_version
        if self._rng.random() < 0.12 and policy_version > 1:
            cited_version = policy_version - 1

        # Contradiction: 10% chance of flipping our previous call on this case.
        prior = self._prior_decisions.get(target)
        approve_p = 0.4 + (sentiment - 0.5) * 0.6  # higher sentiment -> approve
        if self._rng.random() < 0.10 and prior in ("approve", "deny"):
            decision = "deny" if prior == "approve" else "approve"
        else:
            decision = "approve" if self._rng.random() < approve_p else "deny"

        # Hallucination: 3% chance of inventing a case_id.
        if self._rng.random() < 0.03:
            target = target + "_GHOST"

        self._prior_decisions[target] = decision
        return LLMResponse(
            decision=decision,
            rationale=f"sentiment={sentiment:.2f} policy_v={cited_version}",
            referenced_policy_version=cited_version,
            target_case_id=target,
        )

    def _escalation(self, ctx: dict[str, Any]) -> LLMResponse:
        queue: list[str] = ctx.get("queue_case_ids", [])
        load = ctx.get("system_load", 0.0)
        if not queue:
            return LLMResponse(decision="no_op", rationale="empty queue")

        target = queue[0]

        # Under load, occasionally bounce the case back into escalation
        # rather than resolving — produces escalation loops.
        rebound_p = 0.15 + max(0.0, load - 0.5)
        if self._rng.random() < min(rebound_p, 0.6):
            return LLMResponse(
                decision="rebound",
                rationale=f"load={load:.2f}, deferring",
                target_case_id=target,
            )
        return LLMResponse(
            decision="resolve",
            rationale="resolving from queue",
            target_case_id=target,
        )

    def _policy(self, ctx: dict[str, Any]) -> LLMResponse:
        timestep = ctx.get("timestep", 0)
        # Roughly every ~17 steps, bump the policy.
        if timestep > 0 and self._rng.random() < 0.06:
            return LLMResponse(
                decision="policy_update",
                rationale="periodic refresh",
            )
        return LLMResponse(decision="no_op", rationale="policy stable")
