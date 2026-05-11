"""Anthropic Claude adapter — scaffolded for v0, not wired.

To enable: install with `pip install drift[anthropic]`, set
ANTHROPIC_API_KEY, and pass `--llm anthropic` (after wiring it into
cli.py). The shape below mirrors what would be written; the actual
client construction is intentionally deferred.
"""
from __future__ import annotations

from typing import Any

from drift.llm.base import LLMClient, LLMResponse


class AnthropicLLM(LLMClient):
    def __init__(self, model: str = "claude-haiku-4-5", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key
        # Deliberately not constructing the client here — kept as a stub so
        # the import doesn't require the anthropic package to be installed.

    async def generate(self, *, system: str, user: str, ctx: dict[str, Any]) -> LLMResponse:
        raise NotImplementedError(
            "Anthropic adapter is scaffolded but not wired in v0. "
            "Use --llm mock for now, or implement this method against "
            "anthropic.AsyncAnthropic with a tool-use schema that maps "
            "directly onto LLMResponse fields."
        )
