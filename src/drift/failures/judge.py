"""LLM-judged failure detection.

The deterministic detectors in `failures.detectors` fire on crisp, named
patterns specific to drift's shipped topologies (contradictory_refund,
security_bypass, etc.). They don't generalize to arbitrary user domains —
when a real user runs drift on their own multi-agent system, the
deterministic rules don't know what counts as a coordination failure in
that domain.

LLM-judged detection closes that gap. A judge LLM gets a sliding window of
recent actions / events / state snapshots and is asked: "did any of the
five failure families occur here?" It returns structured JSON which the
detector parses into `FailureRecord`s with `llm:` prefixed types.

The pillar this implements (drift-context SKILL.md pillar 3 — hybrid
detection) was previously aspirational; this module makes it real.

Usage from the SDK:

    from drift.failures.judge import LLMJudgeDetector, build_judge

    judge = build_judge("openai", model="gpt-4o-mini")
    drift.run(..., judge_llm=judge, judge_every=5)

The judge runs every `judge_every` steps (default 5) over the trailing
window, with a signature dedupe so each fired failure only reports once.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Awaitable, Protocol

from drift.failures.base import DetectorContext, FailureRecord

# The judge maps every reported failure into one of these families.
# The first five are the stable drift coordination taxonomy (SKILL.md
# "Core failure families") — domain-independent so the judge can be reused
# across any user topology. The sixth (`user_guideline`) is the umbrella
# family for matches against user-supplied guideline strings; the specific
# guideline index is appended to the failure_type as `llm:user_guideline:<i>`.
USER_GUIDELINE_FAMILY = "user_guideline"

JUDGE_FAMILIES: tuple[str, ...] = (
    "coordination_contradiction",
    "grounding_failure",
    "state_drift",
    "emergent_decay",
    "gate_bypass",
    USER_GUIDELINE_FAMILY,
)

JUDGE_PREFIX = "llm:"

# Default judge prompt for the standard five-family taxonomy. When the user
# supplies guidelines via `user_guidelines=`, an extra block is appended at
# run time describing the additional patterns and the user_guideline family.
DEFAULT_JUDGE_SYSTEM = (
    "You analyze a window of one multi-agent system's recent activity and "
    "report coordination failures. You return strict JSON only — never prose.\n\n"
    "Failure families:\n"
    "  - coordination_contradiction: two agents reach opposing decisions on the same target.\n"
    "  - grounding_failure:         an action references a target that doesn't exist or no longer exists.\n"
    "  - state_drift:               an agent acted on outdated world state (e.g. wrong policy version).\n"
    "  - emergent_decay:            the system is trending bad over time across snapshots.\n"
    "  - gate_bypass:               a well-formed action that wasn't allowed (skipped approval, blocked op executed).\n\n"
    "Be CONSERVATIVE. Only report failures you can cite specific evidence for. "
    "If nothing rises to that bar, return an empty list. Hallucinated failures "
    "destroy this tool's value.\n\n"
    "Output format (strict JSON, no prose):\n"
    '{"failures": [{"family": "<one of the five>", '
    '"summary": "<one sentence>", '
    '"evidence_action_ids": ["a000001", ...], '
    '"agents_involved": ["agent_name", ...]}]}\n'
    'If no failures: {"failures": []}'
)


def render_user_guidelines_block(guidelines: list[str]) -> str:
    """Render the user-guideline block appended to the system prompt.

    Returns empty string when no guidelines are supplied so the default
    prompt is byte-equivalent to the pre-guideline behaviour.

    The block tells the judge two things: the patterns to additionally
    watch for, and how to report a match (family=`user_guideline` with the
    1-based `guideline_id` matching the index here). 1-based because LLMs
    confuse 0-based on small lists.
    """
    if not guidelines:
        return ""
    # Filter to non-blank lines FIRST, then enumerate — so the 1-based numbers
    # the judge sees match the indices the user would count themselves and any
    # downstream `llm:user_guideline:<n>` failure_type rows resolve correctly.
    cleaned = [g.strip() for g in guidelines if g and g.strip()]
    if not cleaned:
        return ""
    bullets = "\n".join(f"  {i + 1}. {g}" for i, g in enumerate(cleaned))
    return (
        "\n\nAdditional user-specified patterns to flag — treat each as an additional "
        "failure type in scope. If you find one of these, report it under "
        f"`family: \"{USER_GUIDELINE_FAMILY}\"` and include a `guideline_id` field equal to "
        "the 1-based number in this list:\n"
        + bullets
        + "\n\nWhen reporting a `user_guideline` failure, the output schema becomes:\n"
        '{"family": "user_guideline", "guideline_id": <int>, "summary": "...", '
        '"evidence_action_ids": [...], "agents_involved": [...]}'
    )


def build_system_prompt(
    user_guidelines: list[str] | None = None,
    *,
    base: str = DEFAULT_JUDGE_SYSTEM,
) -> str:
    """Compose the full judge system prompt, appending user guidelines if any."""
    return base + render_user_guidelines_block(list(user_guidelines or []))


# ---- Judge LLM protocol + implementations --------------------------------


class JudgeLLM(Protocol):
    """A minimal LLM client for the judge — separate from agent-side
    `drift.llm.base.LLMClient` because the response shape is different.
    Returns raw text (expected to be JSON)."""

    async def judge(self, *, system: str, user: str) -> str: ...


class ScriptedMockJudge:
    """Placeholder judge for use without API keys.

    The first time it's called in a process, it emits a one-line warning to
    stderr and returns a single failure record of type `llm:placeholder` so
    users can see in the UI that the judge actually ran but isn't doing real
    work. On subsequent calls it returns no failures. Intended for tests,
    smoke runs, and demos that can't reach the network.
    """

    _warned: bool = False
    _fired: bool = False

    async def judge(self, *, system: str, user: str) -> str:
        if not type(self)._warned:
            print(
                "[drift] scripted mock judge is a placeholder — use "
                "openai/anthropic for real LLM-judged detection.",
                file=sys.stderr,
            )
            type(self)._warned = True
        if not self._fired:
            self._fired = True
            return json.dumps({
                "failures": [{
                    "family": "emergent_decay",
                    "summary": (
                        "scripted mock judge placeholder — configure a real "
                        "judge LLM (openai/anthropic) to get real detection"
                    ),
                    "evidence_action_ids": [],
                    "agents_involved": [],
                }]
            })
        return json.dumps({"failures": []})


class OpenAIJudge:
    """OpenAI judge via Chat Completions with JSON-mode response."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        from openai import AsyncOpenAI  # lazy import — openai is optional
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._first_call_succeeded = False

    async def judge(self, *, system: str, user: str) -> str:
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=600,
            )
            self._first_call_succeeded = True
            return resp.choices[0].message.content or '{"failures": []}'
        except Exception as e:
            if not self._first_call_succeeded:
                raise RuntimeError(
                    f"OpenAI judge first call failed ({type(e).__name__}): {e}. "
                    "Check OPENAI_API_KEY, billing, and model name."
                ) from e
            # Transient errors degrade to no-fire so one network blip can't
            # blow up a long run.
            return '{"failures": []}'


class AnthropicJudge:
    """Anthropic Claude judge — scaffolded but not wired in v0.

    To enable: implement against anthropic.AsyncAnthropic with a tool-use
    schema that mirrors the JSON shape DEFAULT_JUDGE_SYSTEM describes.
    """

    def __init__(self, model: str = "claude-haiku-4-5", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key

    async def judge(self, *, system: str, user: str) -> str:  # pragma: no cover - stubbed
        raise NotImplementedError(
            "Anthropic judge is scaffolded but not wired in v0. "
            "Use judge='openai' or judge='mock' for now."
        )


def build_judge(spec: str, *, model: str | None = None) -> JudgeLLM | None:
    """Factory: map a string spec to a JudgeLLM instance.

    Spec values:
      - "off" / "" / None-ish → returns None (no judge runs)
      - "mock"                → ScriptedMockJudge (placeholder, no network)
      - "openai"              → OpenAIJudge with optional model override
      - "anthropic"           → AnthropicJudge (stubbed, raises on call)

    Raises ValueError for unrecognized specs.
    """
    if not spec or spec == "off":
        return None
    s = spec.strip().lower()
    if s == "mock":
        return ScriptedMockJudge()
    if s == "openai":
        return OpenAIJudge(model=model or "gpt-4o-mini")
    if s == "anthropic":
        return AnthropicJudge(model=model or "claude-haiku-4-5")
    raise ValueError(f"unknown judge spec {spec!r}; expected off/mock/openai/anthropic")


# ---- Window rendering ----------------------------------------------------


def _render_window(ctx: DetectorContext, window_steps: int) -> str:
    """Render the last `window_steps` worth of actions, events, snapshots
    into a compact text the judge can read.

    Trimming to a window keeps the prompt small and lets the judge focus on
    recent activity rather than re-analyzing the entire run.
    """
    start_step = max(1, ctx.timestep - window_steps + 1)

    actions = [a for a in ctx.actions if a.timestep >= start_step]
    events = [e for e in ctx.events if e.timestep >= start_step]
    snapshots = [s for s in ctx.history.window(window_steps)]

    parts: list[str] = [f"# Trace window: t={start_step}..{ctx.timestep}"]

    if events:
        parts.append("\n## Events (exogenous changes)")
        for e in events:
            parts.append(f"  t={e.timestep} {e.name}: {e.summary}")

    if actions:
        parts.append("\n## Actions (agent decisions)")
        for a in actions:
            target = f" -> {a.target_case_id}" if a.target_case_id else ""
            policy = f" (policy v{a.referenced_policy_version})" if a.referenced_policy_version is not None else ""
            rationale = f" :: {a.rationale}" if a.rationale else ""
            parts.append(f"  t={a.timestep} [{a.action_id}] {a.agent_name} {a.kind}{target}{policy}{rationale}")

    if snapshots:
        parts.append("\n## Snapshots (world state per step)")
        for s in snapshots:
            # Only include keys whose value isn't the default-zeroish to keep
            # the prompt focused on signal.
            dump = s.model_dump(mode="json")
            interesting = {
                k: v for k, v in dump.items()
                if k != "timestep" and v not in (None, 0, 0.0, "", [], {})
            }
            parts.append(f"  t={s.timestep}: {json.dumps(interesting, default=str)[:400]}")

    return "\n".join(parts)


def _parse_judge_response(
    raw: str, ctx: DetectorContext,
) -> list[FailureRecord]:
    """Parse a judge JSON response into FailureRecord objects.

    Tolerates: trailing prose around the JSON (extracts the first `{...}`),
    unknown family names (skipped with a stderr warning), missing fields.
    """
    if not raw:
        return []
    # Extract a JSON object — some models still wrap responses in prose.
    text = raw.strip()
    if not text.startswith("{"):
        i = text.find("{")
        if i == -1:
            return []
        text = text[i:]
    if not text.endswith("}"):
        j = text.rfind("}")
        if j != -1:
            text = text[: j + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        print(f"[drift] judge returned non-JSON, skipping: {raw[:200]!r}", file=sys.stderr)
        return []

    raw_failures = payload.get("failures") or []
    if not isinstance(raw_failures, list):
        return []

    out: list[FailureRecord] = []
    for item in raw_failures:
        if not isinstance(item, dict):
            continue
        family = str(item.get("family") or "").strip()
        if family not in JUDGE_FAMILIES:
            print(f"[drift] judge returned unknown family {family!r}, skipping", file=sys.stderr)
            continue

        failure_type = f"{JUDGE_PREFIX}{family}"
        summary = str(item.get("summary") or "").strip() or f"{family} (no summary)"

        # user_guideline matches carry a 1-based guideline_id pointer back to
        # the user's guideline list. Embed it in the failure_type so callers
        # can resolve which specific pattern fired without changing the
        # FailureRecord schema. Skip the row entirely if the id is missing
        # or unparseable — better to drop than to mis-attribute.
        if family == USER_GUIDELINE_FAMILY:
            gid_raw = item.get("guideline_id")
            try:
                gid = int(gid_raw)
            except (TypeError, ValueError):
                print(
                    f"[drift] user_guideline match missing parseable guideline_id "
                    f"({gid_raw!r}), skipping",
                    file=sys.stderr,
                )
                continue
            if gid < 1:
                continue
            failure_type = f"{JUDGE_PREFIX}{family}:{gid}"
            summary = f"[guideline #{gid}] {summary}"

        out.append(FailureRecord(
            timestep=ctx.timestep,
            failure_type=failure_type,
            agents_involved=[str(a) for a in (item.get("agents_involved") or [])],
            evidence_action_ids=[str(a) for a in (item.get("evidence_action_ids") or [])],
            summary=summary,
            snapshot_timestep=ctx.timestep,
        ))
    return out


# ---- Detector ------------------------------------------------------------


class LLMJudgeDetector:
    """Async detector that consults an LLM judge over a sliding window.

    Plugs into the same detector pipeline as the sync deterministic ones —
    the simulation runner awaits if the detector returns a coroutine.

    Args:
        judge: a JudgeLLM (use `build_judge(...)`).
        every: run the judge every N timesteps. Default 5 keeps token cost
               bounded for long runs.
        window: how many recent steps to include in each judge prompt.
                Default 5 matches `every` so consecutive judgments cover
                disjoint windows.
        system_prompt: override the default judge system prompt base. The
                       user-guideline block (if any) is appended to whatever
                       base is supplied.
        user_guidelines: optional list of plain-English patterns the user
                         wants drift to additionally flag. Each becomes a
                         user_guideline match candidate; matches are reported
                         under failure_type `llm:user_guideline:<index>`.
                         This is pillar 4 — the programmable test surface
                         that differentiates drift from fixed-primitive tools.
    """

    def __init__(
        self,
        judge: JudgeLLM,
        *,
        every: int = 5,
        window: int = 5,
        system_prompt: str | None = None,
        user_guidelines: list[str] | None = None,
    ) -> None:
        self.judge = judge
        self.every = max(1, int(every))
        self.window = max(1, int(window))
        self.user_guidelines = [g for g in (user_guidelines or []) if g and g.strip()]
        base = system_prompt or DEFAULT_JUDGE_SYSTEM
        self.system_prompt = build_system_prompt(self.user_guidelines, base=base)

    async def __call__(self, ctx: DetectorContext) -> list[FailureRecord]:
        if ctx.timestep % self.every != 0:
            return []
        user = _render_window(ctx, self.window)
        raw = await self.judge.judge(system=self.system_prompt, user=user)
        failures = _parse_judge_response(raw, ctx)

        # Dedupe via the runner's already_reported set so the same finding
        # doesn't re-fire on every window. Fingerprint = (type, summary hash)
        # so different findings of the same type still report independently.
        out: list[FailureRecord] = []
        for f in failures:
            digest = hashlib.sha1(f.summary.encode("utf-8")).hexdigest()[:10]
            fp = f"{f.failure_type}:{digest}"
            if fp in ctx.already_reported:
                continue
            ctx.already_reported.add(fp)
            out.append(f)
        return out


__all__ = [
    "DEFAULT_JUDGE_SYSTEM",
    "JUDGE_FAMILIES",
    "JUDGE_PREFIX",
    "USER_GUIDELINE_FAMILY",
    "JudgeLLM",
    "LLMJudgeDetector",
    "OpenAIJudge",
    "AnthropicJudge",
    "ScriptedMockJudge",
    "build_judge",
    "build_system_prompt",
    "render_user_guidelines_block",
]
