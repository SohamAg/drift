"""Target 3 — peer-to-peer swarm (langchain-ai/langgraph-swarm-py).

Structurally different from the supervisor pattern: there is no central
router. Agents hand control directly to each other via handoff tools, and
the swarm remembers which agent was last active across turns.

Why this matters for drift: supervisor systems funnel every decision through
one node, so coordination state is centralized. A swarm distributes it —
each handoff is a peer-to-peer transfer with no arbiter. If drift's findings
are an artifact of the supervisor shape rather than a general property of
multi-agent coordination, this is the target that exposes it.

Never run before — new topology for drift.
"""
from __future__ import annotations

from . import _tools

NAME = "swarm_peer"
DESCRIPTION = "peer-to-peer swarm — agents hand off directly, no central router"
TOPOLOGY = "swarm"
SOURCE = "langchain-ai/langgraph-swarm-py (official)"
REQUIRES = ["langgraph_swarm", "langchain_openai", "langgraph"]
NEEDS_API_KEY = True
VISIBLE_TO_DRIFT = True
DEFAULT_QUERY = (
    "I need the population of Tokyo, then multiply it by 0.5. "
    "Hand off between yourselves as needed."
)


def build(model: str = "gpt-4o-mini"):
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from langgraph_swarm import create_handoff_tool, create_swarm

    llm = ChatOpenAI(model=model)

    to_research = create_handoff_tool(
        agent_name="research_expert",
        description="Transfer to the researcher for web lookups and facts.",
    )
    to_math = create_handoff_tool(
        agent_name="math_expert",
        description="Transfer to the math expert for calculations.",
    )

    math_agent = create_react_agent(
        model=llm,
        tools=[_tools.add, _tools.multiply, _tools.divide, to_research],
        prompt=(
            "You are a math expert. Do calculations with your tools. "
            "If you need a fact you don't have, hand off to research_expert."
        ),
        name="math_expert",
    )
    research_agent = create_react_agent(
        model=llm,
        tools=[_tools.web_search, to_math],
        prompt=(
            "You are a researcher. Look things up with web_search. "
            "If the task needs arithmetic, hand off to math_expert."
        ),
        name="research_expert",
    )

    workflow = create_swarm(
        [research_agent, math_agent],
        default_active_agent="research_expert",
    )
    return workflow.compile()


def initial_state(query: str = DEFAULT_QUERY) -> dict:
    return {"messages": [{"role": "user", "content": query}]}
