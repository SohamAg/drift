"""Target 1 — canonical 2-agent supervisor (langchain-ai/langgraph-supervisor-py).

The vendor's own README example: a supervisor routing between a math expert
and a research expert. Smallest honest multi-agent target — real framework
code, real LLM routing, but short traces (typically 3 super-steps).

Already validated: drift runs clean here, coordination detectors stay silent.
"""
from __future__ import annotations

from . import _tools

NAME = "supervisor_basic"
DESCRIPTION = "2-agent supervisor (math + research) — the vendor's README example"
TOPOLOGY = "supervisor"
SOURCE = "langchain-ai/langgraph-supervisor-py (official)"
REQUIRES = ["langgraph_supervisor", "langchain_openai", "langgraph"]
NEEDS_API_KEY = True
VISIBLE_TO_DRIFT = True
DEFAULT_QUERY = "What is 7 times 8? Then search the web for the capital of France."


def build(model: str = "gpt-4o-mini"):
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from langgraph_supervisor import create_supervisor

    llm = ChatOpenAI(model=model)

    math_agent = create_react_agent(
        model=llm,
        tools=[_tools.add, _tools.multiply, _tools.divide],
        prompt="You are a math expert. Always use one tool at a time.",
        name="math_expert",
    )
    research_agent = create_react_agent(
        model=llm,
        tools=[_tools.web_search],
        prompt="You are a world-class researcher. Do not do any math.",
        name="research_expert",
    )
    workflow = create_supervisor(
        [math_agent, research_agent],
        model=llm,
        prompt=(
            "You are a team supervisor managing a math expert and a researcher. "
            "For current events, use research_expert. "
            "For math problems, use math_expert."
        ),
    )
    return workflow.compile()


def initial_state(query: str = DEFAULT_QUERY) -> dict:
    return {"messages": [{"role": "user", "content": query}]}
