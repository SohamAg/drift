"""Target 5 — multi-agent collaboration (researcher + chart generator).

LangGraph's "multi-agent collaboration" tutorial shape, built as an explicit
StateGraph rather than via a prebuilt supervisor: two agents share one message
list and route to each other through a conditional edge until one of them
emits FINAL ANSWER.

The tutorial uses Tavily for search; this build substitutes the deterministic
mock so runs cost nothing extra and stay reproducible. The coordination shape
— which is what drift observes — is unchanged.

Why this target: unlike the supervisor targets, routing logic here lives in a
plain Python conditional edge rather than an LLM's tool choice. State is
shared wholesale (one message list, both agents append to it), which is the
classic setup for one agent overwriting context the other still needs.
"""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from . import _tools

NAME = "collaboration_pair"
DESCRIPTION = "researcher + chart-generator sharing one message list, conditional routing"
TOPOLOGY = "pipeline"
SOURCE = "LangGraph multi-agent-collaboration tutorial (Tavily stubbed)"
REQUIRES = ["langchain_openai", "langgraph"]
NEEDS_API_KEY = True
VISIBLE_TO_DRIFT = True
DEFAULT_QUERY = (
    "Get the UK's GDP over the past 3 years, then draw a line chart of it. "
    "Once you have the chart, finish."
)

_MAX_TURNS = 8


class _State(TypedDict):
    """Shared state. Both agents append to the same message list."""
    messages: Annotated[list, lambda a, b: (a or []) + (b or [])]
    sender: str
    turns: int


def build(model: str = "gpt-4o-mini"):
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph

    llm = ChatOpenAI(model=model)

    def _agent(name: str, system: str, tool):
        def node(state: _State) -> dict:
            convo = "\n".join(
                f"{getattr(m, 'name', None) or 'user'}: {m.content}"
                for m in state["messages"]
            )
            reply = llm.invoke(
                [
                    HumanMessage(content=f"{system}\n\nConversation so far:\n{convo}")
                ]
            )
            text = reply.content if isinstance(reply.content, str) else str(reply.content)
            # Deterministic tool layer: the agent's "work product" is mocked.
            work = tool(text[:120]) if tool else ""
            return {
                "messages": [AIMessage(content=f"{text}\n{work}", name=name)],
                "sender": name,
                "turns": state.get("turns", 0) + 1,
            }
        return node

    researcher = _agent(
        "researcher",
        "You are a researcher. Gather the facts needed. When you have them, "
        "say 'handing to chart_generator'. Do not draw charts yourself.",
        lambda t: _tools.web_search(t),
    )
    charter = _agent(
        "chart_generator",
        "You are a chart generator. Draw the chart from the researcher's data. "
        "When the chart is drawn, reply with 'FINAL ANSWER'.",
        lambda t: _tools.make_chart(t, "line"),
    )

    def route(state: _State) -> Literal["researcher", "chart_generator", "__end__"]:
        if state.get("turns", 0) >= _MAX_TURNS:
            return END
        last = state["messages"][-1]
        if "FINAL ANSWER" in str(getattr(last, "content", "")):
            return END
        return "chart_generator" if state["sender"] == "researcher" else "researcher"

    g = StateGraph(_State)
    g.add_node("researcher", researcher)
    g.add_node("chart_generator", charter)
    g.add_edge(START, "researcher")
    g.add_conditional_edges("researcher", route)
    g.add_conditional_edges("chart_generator", route)
    return g.compile()


def initial_state(query: str = DEFAULT_QUERY) -> dict:
    from langchain_core.messages import HumanMessage

    return {"messages": [HumanMessage(content=query)], "sender": "user", "turns": 0}
