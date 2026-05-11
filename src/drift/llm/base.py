"""LLM client abstraction.

The simulator never depends on a particular vendor. Agents call into
`LLMClient.generate(...)` and receive a structured `LLMResponse`. The
mock and the real Anthropic adapter both satisfy this protocol.
"""
from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel


class LLMResponse(BaseModel):
    """What an agent gets back from the LLM.

    The mock returns this directly; the real adapter would parse a
    Claude tool-use response into the same shape. Keeping it typed
    means agents don't have to parse free text.
    """
    decision: str          # e.g. "approve", "deny", "escalate", "no_op"
    rationale: str
    referenced_policy_version: int | None = None
    target_case_id: str | None = None
    raw: dict[str, Any] | None = None


class LLMClient(Protocol):
    async def generate(self, *, system: str, user: str, ctx: dict[str, Any]) -> LLMResponse: ...
