"""Target 10 — deepagents (langchain-ai/deepagents). The observation-boundary probe.

deepagents dispatches sub-agents through tool calls rather than graph nodes.
Until 2026-11-04 that made it drift's known blind spot: the trace was just
`model → tools → model`, the same four node names whether zero or three
sub-agents were configured, and this target was a NEGATIVE control.

Since the adapter streams real graphs with `subgraphs=True`, the sub-agent
graphs a `task` tool call starts DO show up — as `tools/model`,
`tools/TodoListMiddleware.after_model`, … steps under the `tools` namespace
(9 of 19 super-steps on a live gpt-4o-mini run, saved under
results/deepagents/). Since 2026-11-09 those steps are also NAMED: the
adapter matches each subgraph to the still-unanswered `task(...)` call in
the root conversation (by the task text the sub-agent was started with) and
stamps `subagent="researcher"` / `"critic"` / `"summarizer"` on the step —
15 of 15 sub-agent steps on a live run
(results/deepagents/deepagents_named_*.json).

Keep this target in the registry as the boundary probe for what is still
outside drift's view: a sub-agent that is a plain function call inside a
tool (no LangGraph graph), and dispatchers that never name the agent.

See memory/empirical_validation_2026_07_05.md for the original finding.
"""
from __future__ import annotations

NAME = "deepagents_tooldispatch"
DESCRIPTION = "deepagents research agent — sub-agents dispatched via tool calls (named since 2026-11-09)"
TOPOLOGY = "tool_dispatch"
SOURCE = "langchain-ai/deepagents (official, 25.7k stars)"
REQUIRES = ["deepagents", "langchain_openai", "langgraph"]
NEEDS_API_KEY = True
VISIBLE_TO_DRIFT = False  # agents are tool-dispatched, not graph nodes; visible AND named since 2026-11-09 (see docstring)
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
