"""Target 4 — hierarchical agent teams (supervisor of supervisors).

Two specialist teams, each with its own supervisor, coordinated by a top-level
supervisor. Follows LangGraph's "hierarchical agent teams" tutorial shape.

    top_supervisor
      ├── research_team_supervisor → { search_agent, scrape_agent }
      └── authoring_team_supervisor → { writer_agent, chart_agent }

Why this is the richest target: state crosses THREE levels. The top supervisor
delegates to a team supervisor, which delegates to a worker, and results flow
back up. Every level is a place for information to be dropped, restated
incorrectly, or acted on stale. If coordination failures scale with nesting
depth, this is where drift should see it most clearly.
"""
from __future__ import annotations

from . import _tools

NAME = "hierarchical_teams"
DESCRIPTION = "supervisor of supervisors — 2 teams, 4 workers, 3 levels of delegation"
TOPOLOGY = "hierarchical"
SOURCE = "LangGraph hierarchical-agent-teams tutorial pattern"
REQUIRES = ["langgraph_supervisor", "langchain_openai", "langgraph"]
NEEDS_API_KEY = True
VISIBLE_TO_DRIFT = True
DEFAULT_QUERY = (
    "Research the growth of electric vehicle adoption, then write a short "
    "report about it with a chart."
)


def build(model: str = "gpt-4o-mini"):
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from langgraph_supervisor import create_supervisor

    llm = ChatOpenAI(model=model)

    # ---- research team ----
    search_agent = create_react_agent(
        model=llm, tools=[_tools.web_search],
        prompt="You search the web for facts. Report only what you found.",
        name="search_agent",
    )
    scrape_agent = create_react_agent(
        model=llm, tools=[_tools.fetch_url, _tools.read_file],
        prompt="You fetch and read documents. Report only their contents.",
        name="scrape_agent",
    )
    research_team = create_supervisor(
        [search_agent, scrape_agent],
        model=llm,
        prompt=(
            "You lead the research team: search_agent and scrape_agent. "
            "Gather the facts requested and report them back concisely."
        ),
        supervisor_name="research_team_lead",
    ).compile(name="research_team")

    # ---- authoring team ----
    writer_agent = create_react_agent(
        model=llm, tools=[_tools.draft_document, _tools.summarize],
        prompt="You write documents from supplied research. Do not invent facts.",
        name="writer_agent",
    )
    chart_agent = create_react_agent(
        model=llm, tools=[_tools.make_chart],
        prompt="You create charts from supplied data. Do not invent data.",
        name="chart_agent",
    )
    authoring_team = create_supervisor(
        [writer_agent, chart_agent],
        model=llm,
        prompt=(
            "You lead the authoring team: writer_agent and chart_agent. "
            "Produce the requested deliverable from the research provided."
        ),
        supervisor_name="authoring_team_lead",
    ).compile(name="authoring_team")

    # ---- top level ----
    top = create_supervisor(
        [research_team, authoring_team],
        model=llm,
        prompt=(
            "You coordinate two teams: research_team gathers facts, "
            "authoring_team produces documents and charts. "
            "Always research before authoring. Pass the research findings "
            "along to the authoring team explicitly."
        ),
        supervisor_name="top_supervisor",
    )
    return top.compile()


def initial_state(query: str = DEFAULT_QUERY) -> dict:
    return {"messages": [{"role": "user", "content": query}]}
