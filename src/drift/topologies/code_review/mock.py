"""Mock-LLM role handlers for the code-review topology.

Same intentional-flakiness philosophy as the support topology: the mock
must misbehave often enough that detectors fire on a 30-step run. Each
handler reads `ctx` and returns an LLMResponse.
"""
from __future__ import annotations

import random
from typing import Any

from drift.llm.base import LLMResponse


def _proposer(rng: random.Random, prior: dict[str, str], ctx: dict[str, Any]) -> LLMResponse:
    load = ctx.get("system_load", 0.0)
    open_ids = ctx.get("open_case_ids", [])
    # Under pressure, opens more. Naive prompt has nothing stopping this.
    propose_p = 0.4 + max(0.0, load - 0.5) * 0.6
    if rng.random() < propose_p:
        return LLMResponse(decision="propose_change", rationale=f"opening PR (load={load:.2f})")
    return LLMResponse(decision="no_op", rationale="no proposal this step")


def _reviewer(rng: random.Random, prior: dict[str, str], ctx: dict[str, Any]) -> LLMResponse:
    open_ids = ctx.get("open_case_ids", [])
    if not open_ids:
        return LLMResponse(decision="no_op", rationale="nothing to review")
    target = rng.choice(open_ids)

    # Hallucination: 4% chance of inventing a PR id
    if rng.random() < 0.04:
        target = target + "_X"

    # Contradiction: 12% chance of flipping our prior decision on the PR
    prior_d = prior.get(target)
    if rng.random() < 0.12 and prior_d in ("approve", "reject"):
        decision = "reject" if prior_d == "approve" else "approve"
    else:
        # Under load, more lax: approve more often
        load = ctx.get("system_load", 0.0)
        approve_p = 0.55 + max(0.0, load - 0.5) * 0.4
        decision = "approve" if rng.random() < approve_p else "reject"

    prior[target] = decision
    return LLMResponse(
        decision=decision,
        rationale=f"reviewer call on {target}",
        target_case_id=target,
    )


def _security(rng: random.Random, prior: dict[str, str], ctx: dict[str, Any]) -> LLMResponse:
    open_ids = ctx.get("open_case_ids", [])
    if not open_ids:
        return LLMResponse(decision="no_op", rationale="no PRs to inspect")
    target = rng.choice(open_ids)

    load = ctx.get("system_load", 0.0)
    # Under load, security is more likely to clear without inspecting (the failure mode)
    clear_p = 0.35 + max(0.0, load - 0.5) * 0.3
    if rng.random() < clear_p:
        return LLMResponse(decision="clear", rationale="appears clean", target_case_id=target)
    if rng.random() < 0.40:
        return LLMResponse(decision="block", rationale="needs deeper review", target_case_id=target)
    return LLMResponse(decision="no_op", rationale="defer security review", target_case_id=target)


def _merge(rng: random.Random, prior: dict[str, str], ctx: dict[str, Any]) -> LLMResponse:
    open_ids = ctx.get("open_case_ids", [])
    if not open_ids:
        return LLMResponse(decision="no_op", rationale="nothing to merge")
    target = rng.choice(open_ids)

    load = ctx.get("system_load", 0.0)
    # Under high load (deadline pressure), the naive merger merges aggressively —
    # even without approval / with security blocked. That's the failure mode.
    aggressive_p = 0.30 + max(0.0, load - 0.5) * 0.6
    if rng.random() < aggressive_p:
        return LLMResponse(decision="merge", rationale=f"shipping under load={load:.2f}", target_case_id=target)
    if rng.random() < 0.30:
        return LLMResponse(decision="defer", rationale="not ready", target_case_id=target)
    return LLMResponse(decision="no_op", rationale="awaiting reviews", target_case_id=target)


CODE_REVIEW_MOCK_HANDLERS = {
    "proposer": _proposer,
    "reviewer": _reviewer,
    "security": _security,
    "merge":    _merge,
}
