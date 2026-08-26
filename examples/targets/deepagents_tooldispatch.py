"""Target 10 — deepagents (langchain-ai/deepagents). The known blind spot.

Included deliberately as a NEGATIVE control. deepagents dispatches sub-agents
through tool calls rather than graph nodes, so from LangGraph's `.stream()`
perspective — which is what drift reads — the trace is just
`model → tools → model`, with the same four node names whether zero or three
sub-agents are configured.

Drift runs against it without error and reports nothing. That is not a
detector-quality problem; it is drift's observation boundary. Keeping this
target in the registry means every future harness run re-confirms the
boundary rather than letting it be quietly forgotten, and any change that
makes findings appear here is a genuine capability change worth noticing.

See memory/empirical_validation_2026_07_05.md for the original finding.
"""
from __future__ import annotations

NAME = "deepagents_tooldispatch"
DESCRIPTION = "deepagents research agent — sub-agents hidden in tool calls (negative control)"
TOPOLOGY = "tool_dispatch"
SOURCE = "langchain-ai/deepagents (official, 25.7k stars)"
REQUIRES = ["deepagents", "langchain_openai", "langgraph"]
NEEDS_API_KEY = True
VISIBLE_TO_DRIFT = False  # <- the point of this target
DEFAULT_QUERY = (
    "Research the pros and cons of LangGraph vs CrewAI as multi-agent "
    "frameworks. Provide a short 3-paragraph summary."
)

SUBAGENTS = [
    {
        "name": "researcher",
        "description": "Researches a topic and reports findings.",
        "system_prompt": (
            "You are a researcher. When given a topic, produce a concise list "
            "of 3-4 key facts. Keep it under 100 words."
        ),
    },
    {
        "name": "critic",
        "description": "Critiques findings and flags any it disagrees with.",
        "system_prompt": (
            "You are a skeptical critic. Identify findings that seem "
            "overstated, wrong, or missing nuance. Keep it brief."
        ),
    },
    {
        "name": "summarizer",
        "description": "Summarizes findings into a final response.",
        "system_prompt": (
            "You are a summarizer. Given findings and any critique, produce a "
            "final 3-paragraph summary."
        ),
    },
]


def build(model: str = "gpt-4o-mini", with_subagents: bool = True):
    from deepagents import create_deep_agent

    return create_deep_agent(
        model=f"openai:{model}",
        tools=[],
        subagents=SUBAGENTS if with_subagents else None,
        system_prompt=(
            "You are a research supervisor. Delegate research to the researcher "
            "sub-agent, then have the critic review the findings, then have the "
            "summarizer produce a final summary. Keep the final answer under "
            "5 sentences."
        ),
    )


def initial_state(query: str = DEFAULT_QUERY) -> dict:
    return {"messages": [{"role": "user", "content": query}]}
