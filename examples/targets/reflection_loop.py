"""Target 7 — reflection loop (generator ↔ critic).

LangGraph's reflection pattern: a generator drafts, a critic critiques, and
the generator revises, cycling a bounded number of times.

Why this target matters for drift: this is the shape that produces
`emergent_decay` — no single agent is wrong at any step, but the system can
converge on something worse over iterations, or ping-pong without improving.
It's also the smallest system where a *cycle* exists in the graph, which
makes it the natural test of whether drift's per-super-step trace handles
revisited nodes correctly.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from . import _tools

NAME = "reflection_loop"
DESCRIPTION = "generator ↔ critic cycle with bounded revisions"
TOPOLOGY = "loop"
SOURCE = "LangGraph reflection tutorial pattern"
REQUIRES = ["langchain_openai", "langgraph"]
NEEDS_API_KEY = True
VISIBLE_TO_DRIFT = True
DEFAULT_QUERY = "Write a three-sentence pitch for a multi-agent testing tool."

_MAX_REVISIONS = 3


class _State(TypedDict):
    messages: Annotated[list, lambda a, b: (a or []) + (b or [])]
    draft: str
    critique: str
    revision: int


def build(model: str = "gpt-4o-mini"):
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph

    llm = ChatOpenAI(model=model)

    def generator(state: _State) -> dict:
        prior = state.get("draft", "")
        crit = state.get("critique", "")
        prompt = (
            f"Task: {state['messages'][0].content}\n\n"
            + (f"Your previous draft:\n{prior}\n\nCritique to address:\n{crit}\n\n"
               "Write an improved draft." if prior else "Write the first draft.")
        )
        reply = llm.invoke([HumanMessage(content=prompt)])
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        return {
            "draft": text,
            "revision": state.get("revision", 0) + 1,
            "messages": [AIMessage(content=text, name="generator")],
        }

    def critic(state: _State) -> dict:
        reply = llm.invoke([HumanMessage(content=(
            "Critique this draft in one or two sentences. Be specific and "
            "actionable. If it is genuinely good, say 'APPROVED'.\n\n"
            f"{state.get('draft', '')}"
        ))])
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        _tools.critique(state.get("draft", ""))
        return {
            "critique": text,
            "messages": [AIMessage(content=text, name="critic")],
        }

    def should_continue(state: _State) -> str:
        if "APPROVED" in (state.get("critique") or ""):
            return END
        if state.get("revision", 0) >= _MAX_REVISIONS:
            return END
        return "generator"

    g = StateGraph(_State)
    g.add_node("generator", generator)
    g.add_node("critic", critic)
    g.add_edge(START, "generator")
    g.add_edge("generator", "critic")
    g.add_conditional_edges("critic", should_continue)
    return g.compile()


def initial_state(query: str = DEFAULT_QUERY) -> dict:
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=query)],
        "draft": "",
        "critique": "",
        "revision": 0,
    }
