"""Historical grounding for scenario simulations — real precedent, not
pure LLM imagination, plus a way to check whether that precedent
actually helps.

Two capabilities on top of target_builder.py's scenario generation:

1. Retrieval-grounding (`retrieve_similar_events` + `build_grounded_query`)
   — before a simulation runs, pull real historical analogues (an event
   that actually happened, and what actually happened to the stock
   afterward) and fold them into the query, so agents reason from real
   precedent instead of guessing.

2. Calibration (see drift.calibration) — run a simulation against a
   held-out historical event whose outcome is already known, and check
   whether its structured verdict's predicted direction matches what
   actually happened. This is the only way "grounded in historical
   data" means anything verifiable, rather than just sounding more
   credible.

HISTORICAL_EVENTS below is real, verified data, not fabricated: each
entry's baseline_close/reaction_close/actual_move_pct was fetched live
from Yahoo Finance (yfinance) on 2026-10-22 for well-documented,
unambiguous single-trigger market events (an earnings report, an
incident, an announcement with a clear date) — never a multi-week
phenomenon like the GameStop squeeze, which doesn't reduce to one
honest baseline/reaction date pair. Event descriptions are plain
statements of well-documented public record, not speculation.

7 events is a small seed, not a statistically powerful calibration set
— treat calibration scores from this module as a sanity check, not a
confidence interval. Growing the dataset (more real events, verified
the same way) is the obvious next step before trusting this beyond a
demo.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

DEFAULT_MODEL = "gpt-4o-mini"

# Stock-reaction-prediction verdict shape — what a generated simulation's
# final node is asked to produce when grounded against this dataset. See
# drift.target_builder.generate_target_source's verdict_schema parameter.
STOCK_VERDICT_SCHEMA: dict[str, str] = {
    "direction": "one of: up, down, flat",
    "magnitude_pct": "your best-guess percentage price move as a number, e.g. -12.5",
    "confidence": "one of: low, medium, high",
    "reasoning": "1-2 sentence explanation citing what the team found",
}

# A fixed, reusable roster for calibration (drift.calibration) — unlike
# /api/simulate-scenario's per-scenario AI-proposed roster, calibration
# needs ONE consistent system under repeated test across all held-out
# events, not 7 different ad-hoc rosters that would each need their own
# calibration run to mean anything.
DEFAULT_STOCK_REACTION_TOPOLOGY = "fanout"
DEFAULT_STOCK_REACTION_AGENTS: tuple[dict[str, str], ...] = (
    {"name": "dispatcher", "role": (
        "Split the scenario into the angles the team should each "
        "investigate: market/trading reaction, public sentiment, and "
        "underlying business fundamentals."
    )},
    {"name": "market_analyst", "role": "Assess the likely market/trading reaction to this scenario."},
    {"name": "sentiment_analyst", "role": "Assess the likely public and social sentiment reaction to this scenario."},
    {"name": "fundamentals_analyst", "role": "Assess this scenario's impact on the underlying business fundamentals."},
    {"name": "verdict_synth", "role": "Synthesize all analysts' findings into one final stock-reaction prediction."},
)


@dataclass(frozen=True)
class HistoricalEvent:
    ticker: str
    baseline_date: str   # YYYY-MM-DD, the trading day before the reaction
    reaction_date: str   # YYYY-MM-DD, the trading day the move shows up in
    description: str     # plain statement of what happened, no speculation
    baseline_close: float
    reaction_close: float
    actual_move_pct: float  # (reaction_close - baseline_close) / baseline_close * 100

    @property
    def actual_direction(self) -> str:
        """Same up/down/flat vocabulary as STOCK_VERDICT_SCHEMA's
        "direction" field, so a verdict and an actual outcome are directly
        comparable. +/-1% is the flat band — small enough that real noise
        inside it isn't a meaningful "prediction," large enough that
        real-event moves in this dataset (all several points or more)
        never land in it by chance."""
        if self.actual_move_pct > 1.0:
            return "up"
        if self.actual_move_pct < -1.0:
            return "down"
        return "flat"

    def to_json(self) -> dict:
        return {
            "ticker": self.ticker,
            "baseline_date": self.baseline_date,
            "reaction_date": self.reaction_date,
            "description": self.description,
            "baseline_close": self.baseline_close,
            "reaction_close": self.reaction_close,
            "actual_move_pct": self.actual_move_pct,
            "actual_direction": self.actual_direction,
        }


# Fetched live via yfinance on 2026-10-22 — see this module's docstring.
# Each baseline_date/reaction_date pair is the trading day the news broke
# (or the day before, when reported after market close) and the very next
# trading day's close, so the move is attributable to that one event, not
# accumulated drift over a longer window.
HISTORICAL_EVENTS: tuple[HistoricalEvent, ...] = (
    HistoricalEvent(
        ticker="META", baseline_date="2022-02-02", reaction_date="2022-02-03",
        description=(
            "Meta Q4 2021 earnings reveal the first-ever decline in daily "
            "active users and weak Q1 guidance, reported after market "
            "close on Feb 2, 2022."
        ),
        baseline_close=319.94, reaction_close=235.50, actual_move_pct=-26.39,
    ),
    HistoricalEvent(
        ticker="NVDA", baseline_date="2023-05-24", reaction_date="2023-05-25",
        description=(
            "Nvidia Q1 FY24 earnings beat estimates with AI-driven "
            "guidance far above Street expectations, reported after "
            "market close on May 24, 2023."
        ),
        baseline_close=30.44, reaction_close=37.86, actual_move_pct=24.37,
    ),
    HistoricalEvent(
        ticker="BA", baseline_date="2024-01-05", reaction_date="2024-01-08",
        description=(
            "An Alaska Airlines Boeing 737 MAX 9 suffers a mid-flight "
            "door-plug blowout on Jan 5, 2024, prompting an FAA grounding "
            "order over the following weekend."
        ),
        baseline_close=249.00, reaction_close=229.00, actual_move_pct=-8.03,
    ),
    HistoricalEvent(
        ticker="TSLA", baseline_date="2020-08-11", reaction_date="2020-08-12",
        description="Tesla announces a 5-for-1 stock split, effective Aug 31, 2020.",
        baseline_close=91.63, reaction_close=103.65, actual_move_pct=13.12,
    ),
    HistoricalEvent(
        ticker="NFLX", baseline_date="2022-04-19", reaction_date="2022-04-20",
        description=(
            "Netflix reports its first subscriber loss in over a decade "
            "for Q1 2022, reported after market close on Apr 19, 2022."
        ),
        baseline_close=34.86, reaction_close=22.62, actual_move_pct=-35.12,
    ),
    HistoricalEvent(
        ticker="PTON", baseline_date="2021-11-04", reaction_date="2021-11-05",
        description=(
            "Peloton withdraws its full-year guidance and cites "
            "weakening demand, announced after market close on Nov 4, 2021."
        ),
        baseline_close=86.06, reaction_close=55.64, actual_move_pct=-35.35,
    ),
    HistoricalEvent(
        ticker="SNAP", baseline_date="2022-05-23", reaction_date="2022-05-24",
        description=(
            "Snap issues a surprise profit warning ahead of an investor "
            "conference, announced May 23, 2022."
        ),
        baseline_close=22.47, reaction_close=12.79, actual_move_pct=-43.08,
    ),
)


def load_events() -> list[HistoricalEvent]:
    """The seed dataset. Returns a fresh list each call (the tuple itself
    is immutable/frozen, same pattern as examples/targets/registry())."""
    return list(HISTORICAL_EVENTS)


class GroundingError(ValueError):
    """Raised when retrieval-grounding can't produce a usable result —
    same philosophy as target_builder.ScenarioProposalError: never
    silently degrade to an empty/misleading result."""


def _retrieval_prompt(scenario: str, events: list[HistoricalEvent], k: int) -> tuple[str, str]:
    event_lines = "\n".join(
        f"{i}. [{e.ticker}] {e.reaction_date}: {e.description} "
        f"(actual result: {e.actual_move_pct:+.1f}%, {e.actual_direction})"
        for i, e in enumerate(events)
    )
    system = (
        "You pick which historical market events are the most relevant "
        "real-world analogues for a new scenario, to ground a simulation's "
        "reasoning in actual precedent instead of speculation. Relevance "
        "means similar TRIGGER TYPE (e.g. earnings miss, safety incident, "
        "guidance change) or similar STAKES, not necessarily the same "
        "company or sector.\n\n"
        f"Historical events:\n{event_lines}\n\n"
        f"Respond with STRICT JSON, no markdown fencing: "
        f'{{"indices": [<up to {k} integers from the list above, most '
        'relevant first>]}}'
    )
    return system, f"New scenario: {scenario}"


async def _call_retrieval_llm(system: str, user: str, model: str) -> str:
    """Isolated from drift.failures.judge.OpenAIJudge for the same reason
    target_builder._call_proposal_llm is: that class's degrade-to-empty-
    on-failure behavior is right for coordination judging, wrong here —
    a silently empty retrieval would just look like a confusing mismatch
    downstream. Errors propagate instead."""
    import os

    from openai import AsyncOpenAI  # lazy import — openai is optional

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=200,
        )
    except Exception as e:
        raise GroundingError(
            f"LLM call failed ({type(e).__name__}): {e}. "
            "Check OPENAI_API_KEY, billing, and model name."
        ) from e
    return resp.choices[0].message.content or "{}"


async def retrieve_similar_events(
    scenario: str,
    events: list[HistoricalEvent] | None = None,
    *,
    k: int = 3,
    model: str = DEFAULT_MODEL,
) -> list[HistoricalEvent]:
    """Which of `events` (default: the seed dataset) are the most
    relevant real-world analogues for `scenario`.

    With only 7 events, a single LLM call describing all of them and
    asking for the top K is simpler and cheaper than building an
    embedding/vector-search pipeline for a dataset this size — revisit
    if the dataset grows into the hundreds.
    """
    scenario = scenario.strip()
    if not scenario:
        raise GroundingError("scenario description is required")
    pool = events if events is not None else load_events()
    if not pool:
        raise GroundingError("no historical events available to retrieve from")

    system, user = _retrieval_prompt(scenario, pool, k)
    raw = await _call_retrieval_llm(system, user, model)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise GroundingError(f"model did not return valid JSON: {e}") from e
    indices = data.get("indices") if isinstance(data, dict) else None
    if not isinstance(indices, list):
        raise GroundingError("model's JSON response had no usable 'indices' list")

    out: list[HistoricalEvent] = []
    for i in indices:
        if isinstance(i, int) and 0 <= i < len(pool) and pool[i] not in out:
            out.append(pool[i])
        if len(out) >= k:
            break
    if not out:
        raise GroundingError("model's proposal referenced no valid event indices")
    return out


def build_grounded_query(scenario: str, analogues: list[HistoricalEvent]) -> str:
    """Fold real historical analogues into a scenario's query text. This
    is the entire integration surface with target_builder's generated
    code — no changes to generated code needed, grounding is just a
    smarter query passed to initial_state(query)."""
    lines = [f"Scenario to analyze: {scenario}", ""]
    if analogues:
        lines.append(
            "Relevant historical precedents (real outcomes, for context — "
            "not a guarantee the same thing happens again):"
        )
        for e in analogues:
            lines.append(
                f"- {e.ticker} on {e.reaction_date}: {e.description} "
                f"Actual result: the stock moved {e.actual_move_pct:+.1f}% "
                f"({e.actual_direction})."
            )
        lines.append("")
    lines.append(
        "Using the scenario above and any relevant precedent, provide "
        "your analysis."
    )
    return "\n".join(lines)
