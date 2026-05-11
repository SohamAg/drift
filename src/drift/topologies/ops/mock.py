"""Mock-LLM role handlers for the ops topology.

Tuned to misbehave often enough to surface comms_lag, silent_remediation,
and contradictory_diagnosis on a 30-50 step run.
"""
from __future__ import annotations

import random
from typing import Any

from drift.llm.base import LLMResponse


def _triage(rng: random.Random, prior: dict[str, str], ctx: dict[str, Any]) -> LLMResponse:
    open_ids = ctx.get("open_case_ids", [])
    if not open_ids:
        return LLMResponse(decision="no_op", rationale="no incidents to triage")
    target = rng.choice(open_ids)
    if rng.random() < 0.85:
        return LLMResponse(decision="triage", rationale="triaging", target_case_id=target)
    return LLMResponse(decision="no_op", rationale="busy", target_case_id=target)


_HYPOTHESES = ["bad deploy", "DB locks", "third-party API", "cache stampede", "config drift", "GC pause"]


def _diagnosis(rng: random.Random, prior: dict[str, str], ctx: dict[str, Any]) -> LLMResponse:
    open_ids = ctx.get("open_case_ids", [])
    if not open_ids:
        return LLMResponse(decision="no_op", rationale="no diagnosable incidents")
    target = rng.choice(open_ids)
    # Naive failure mode: 30% chance of re-diagnosing with a different hypothesis,
    # producing contradictory_diagnosis findings.
    h = rng.choice(_HYPOTHESES)
    if rng.random() < 0.55:
        return LLMResponse(decision="diagnose", rationale=h, target_case_id=target)
    return LLMResponse(decision="no_op", rationale="watching", target_case_id=target)


def _remediation(rng: random.Random, prior: dict[str, str], ctx: dict[str, Any]) -> LLMResponse:
    open_ids = ctx.get("open_case_ids", [])
    if not open_ids:
        return LLMResponse(decision="no_op", rationale="nothing to remediate")
    target = rng.choice(open_ids)
    load = ctx.get("system_load", 0.0)
    fix_p = 0.55 + max(0.0, load - 0.5) * 0.4
    if rng.random() < fix_p:
        return LLMResponse(decision="remediate", rationale="applying fix", target_case_id=target)
    return LLMResponse(decision="no_op", rationale="not yet", target_case_id=target)


def _comms(rng: random.Random, prior: dict[str, str], ctx: dict[str, Any]) -> LLMResponse:
    open_ids = ctx.get("open_case_ids", [])
    if not open_ids:
        return LLMResponse(decision="no_op", rationale="nothing public-facing")
    target = rng.choice(open_ids)
    # The naive failure mode: low base rate of comms = many incidents get
    # remediated silently, triggering silent_remediation + comms_lag.
    sentiment = ctx.get("customer_sentiment", 0.7)
    comms_p = 0.30 + max(0.0, 0.5 - sentiment) * 0.5  # only comms more when trust is already crumbling
    if rng.random() < comms_p:
        return LLMResponse(decision="communicate", rationale="public update", target_case_id=target)
    return LLMResponse(decision="no_op", rationale="holding", target_case_id=target)


OPS_MOCK_HANDLERS = {
    "triage":      _triage,
    "diagnosis":   _diagnosis,
    "remediation": _remediation,
    "comms":       _comms,
}
