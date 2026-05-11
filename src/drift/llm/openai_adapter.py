"""OpenAI adapter using structured JSON output.

Each agent role gets a system prompt + a JSON schema that constrains
the LLM's output to the fields LLMResponse expects. This keeps real-LLM
behavior compatible with the rest of the simulator (detectors,
scheduler, comparison) without parsing free-form text.
"""
from __future__ import annotations

import json
import os
from typing import Any

from drift.llm.base import LLMClient, LLMResponse

# Per-role decision vocabulary. Constraining the model up front prevents
# free-form decisions like "investigate" that we'd have to map.
_ALLOWED_DECISIONS = {
    # support topology
    "support":    ["respond", "escalate", "no_op"],
    "refund":     ["approve", "deny", "no_op"],
    "escalation": ["resolve", "rebound", "no_op"],
    "policy":     ["policy_update", "no_op"],
    # code-review topology
    "proposer":   ["propose_change", "no_op"],
    "reviewer":   ["approve", "reject", "no_op"],
    "security":   ["block", "clear", "no_op"],
    "merge":      ["merge", "defer", "no_op"],
    # ops topology
    "triage":      ["triage", "no_op"],
    "diagnosis":   ["diagnose", "no_op"],
    "remediation": ["remediate", "no_op"],
    "comms":       ["communicate", "no_op"],
}


def _schema_for(role: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": _ALLOWED_DECISIONS.get(role, ["no_op"])},
            "rationale": {"type": "string"},
            "target_case_id": {"type": ["string", "null"]},
            "referenced_policy_version": {"type": ["integer", "null"]},
        },
        "required": ["decision", "rationale", "target_case_id", "referenced_policy_version"],
    }


class OpenAILLM(LLMClient):
    """Calls OpenAI Chat Completions with response_format=json_schema.

    Errors on the first call (e.g. invalid key, no quota) are raised so
    misconfigurations surface immediately instead of silently producing
    a run full of `no_op` actions. Subsequent transient errors fall back
    to `no_op` so a flaky network can't kill a long simulation.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        from openai import AsyncOpenAI  # imported lazily so the package is optional
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._first_call_succeeded = False

    async def generate(self, *, system: str, user: str, ctx: dict[str, Any]) -> LLMResponse:
        role = ctx.get("agent_role", "unknown")
        schema = _schema_for(role)

        # The user message gives the LLM the slice of world state and the
        # vocabulary it must choose from. Keeping it terse minimizes tokens.
        rendered = (
            f"{user}\n\n"
            f"You are the {role} agent.\n"
            f"Allowed decisions: {_ALLOWED_DECISIONS.get(role, ['no_op'])}.\n"
            "Pick exactly one decision. If you reference a case, target_case_id "
            "must be one currently in your observation. If your role doesn't use "
            "policy_version, set referenced_policy_version to null."
        )

        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": rendered},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"{role}_decision",
                        "schema": schema,
                        "strict": True,
                    },
                },
                temperature=0.7,
                max_tokens=256,
            )
            payload = json.loads(resp.choices[0].message.content or "{}")
            self._first_call_succeeded = True
        except Exception as e:
            if not self._first_call_succeeded:
                # Configuration error (bad key, no quota, wrong model). Crash loudly
                # so users don't waste a run thinking everything ran with no_ops.
                raise RuntimeError(
                    f"OpenAI first call failed ({type(e).__name__}): {e}. "
                    "Check OPENAI_API_KEY, billing, and model name."
                ) from e
            return LLMResponse(decision="no_op", rationale=f"llm_error: {type(e).__name__}: {e}")

        return LLMResponse(
            decision=payload.get("decision", "no_op"),
            rationale=payload.get("rationale", ""),
            target_case_id=payload.get("target_case_id"),
            referenced_policy_version=payload.get("referenced_policy_version"),
            raw=payload,
        )
