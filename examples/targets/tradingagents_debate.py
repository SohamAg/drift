"""Target — TradingAgents (TauricResearch/TradingAgents).

The largest and most structurally interesting target in the registry: ~20
nodes built from a real, actively-maintained open-source application, not a
vendor README snippet. Four analyst types (market / social / news /
fundamentals) each report in, then a Bull Researcher and Bear Researcher
argue opposing positions, a Research Manager rules on the debate, a Trader
proposes an action, and three risk debators (aggressive / neutral /
conservative) argue again before a Risk Judge issues the final call.

This is the adversarial-debate topology: unlike a supervisor routing to
specialists who don't interact, here multiple agents are DESIGNED to
disagree with each other on the same target (buy/sell/hold on one ticker),
which makes it the sharpest test yet of drift's `coordination_contradiction`
family — the debate structure means contradiction is expected mid-run, so
the interesting failure is a contradiction that survives INTO the final
decision, or a debate participant that ignores the other side's evidence.

Uses free `yfinance` data by default — no financial-data API key required,
just an LLM key (OPENAI_API_KEY). Costs stack fast: 4 analysts x tool
round-trips + 2-round bull/bear debate + 2-round risk debate is roughly
25-30 LLM calls per run at the default config below.

Model choice matters here in a way it doesn't for the other targets:
`tradingagents`' `build_chat_model` unconditionally sends a `reasoning_effort`
request param for the `"openai"` provider (see `tradingagents/llm.py`), which
the Chat Completions API rejects on non-reasoning models like `gpt-4o-mini`
(400 "Unrecognized request argument: reasoning_effort"). Default here is a
reasoning-capable model instead; this is a real constraint of the third-party
library, not something drift imposes.
"""
from __future__ import annotations

NAME = "tradingagents_debate"
DESCRIPTION = (
    "TradingAgents (TauricResearch) — 4 analysts -> bull/bear debate -> "
    "research manager -> trader -> risk debate -> risk judge"
)
TOPOLOGY = "hierarchical"  # debate-then-judge stages, each stage itself multi-agent
SOURCE = "TauricResearch/TradingAgents (official, third-party)"
REQUIRES = ["tradingagents", "langgraph", "langchain_openai"]
NEEDS_API_KEY = True
VISIBLE_TO_DRIFT = True
DEFAULT_QUERY = "NVDA"
DEFAULT_MODEL = "gpt-5-mini"  # must be reasoning-capable, see module docstring

# Trade date must be in the past relative to whenever this runs. Kept a few
# weeks behind "today" so it stays valid without needing to be bumped often.
DEFAULT_TRADE_DATE = "2026-07-15"


def build(model: str = "gpt-5-mini"):
    from tradingagents.config import TradingAgentsConfig
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    config = TradingAgentsConfig(
        llm_provider="openai",
        deep_think_llm=model,
        quick_think_llm=model,
        max_debate_rounds=1,
        max_risk_discuss_rounds=1,
        max_recur_limit=100,
    )
    trading_graph = TradingAgentsGraph(config=config)
    return trading_graph.graph  # CompiledStateGraph — cached_property, built once


def initial_state(query: str = DEFAULT_QUERY) -> dict:
    """``query`` is a ticker symbol (e.g. "NVDA"); date is fixed, see above."""
    from tradingagents.graph.propagation import Propagator

    propagator = Propagator(max_recur_limit=100)
    state = propagator.create_initial_state(query, DEFAULT_TRADE_DATE)
    return state.model_dump()
