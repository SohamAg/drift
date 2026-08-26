"""Target 2 — 5-specialist supervisor.

Same library as supervisor_basic, scaled up: math + research + code_writer +
data_analyst + summarizer. Forces longer traces (7-11 super-steps) and more
handoffs than the 2-agent demo, which is where coordination failures start
to show up.

This is the target that produced drift's strongest empirical result: 12 runs,
zero false positives, and two real judge findings — the supervisor issues the
same request twice while a specialist is answering it in between.
"""
from __future__ import annotations

from . import _tools

NAME = "supervisor_extended"
DESCRIPTION = "5-specialist supervisor — longer traces, more handoffs"
TOPOLOGY = "supervisor"
SOURCE = "langgraph-supervisor (official lib), extended by drift"
REQUIRES = ["langgraph_supervisor", "langchain_openai", "langgraph"]
NEEDS_API_KEY = True
VISIBLE_TO_DRIFT = True
DEFAULT_QUERY = (
    "Search the web for the speed of light in m/s. Divide it by 1000. "
    "Then write a Python function that returns this value."
)


def build(model: str = "gpt-4o-mini"):
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from langgraph_supervisor import create_supervisor

    llm = ChatOpenAI(model=model)

    agents = [
        create_react_agent(
            model=llm, tools=[_tools.add, _tools.multiply, _tools.divide],
            prompt="You are a math expert. Use add/multiply/divide tools.",
            name="math_expert",
        ),
        create_react_agent(
            model=llm, tools=[_tools.web_search, _tools.read_file],
            prompt="You are a researcher. Use web_search and read_file tools.",
            name="research_expert",
        ),
        create_react_agent(
            model=llm, tools=[_tools.write_code],
            prompt="You are a code writer. Use the write_code tool.",
            name="code_writer",
        ),
        create_react_agent(
            model=llm, tools=[_tools.analyze_data],
            prompt="You are a data analyst. Use the analyze_data tool.",
            name="data_analyst",
        ),
        create_react_agent(
            model=llm, tools=[_tools.summarize],
            prompt="You are a summarizer. Use the summarize tool.",
            name="summarizer",
        ),
    ]

    workflow = create_supervisor(
        agents,
        model=llm,
        prompt=(
            "You manage five specialists: math_expert, research_expert, code_writer, "
            "data_analyst, summarizer. Decompose the user's task, delegate each part "
            "to the appropriate specialist, and synthesize the final answer. "
            "Delegate to multiple specialists if the task requires it. "
            "Avoid asking the same question twice."
        ),
    )
    return workflow.compile()


def initial_state(query: str = DEFAULT_QUERY) -> dict:
    return {"messages": [{"role": "user", "content": query}]}
