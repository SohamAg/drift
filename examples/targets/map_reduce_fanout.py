"""Target 8 — map-reduce fan-out via LangGraph's Send API.

A dispatcher splits the task into N subtasks and spawns a worker per subtask
in parallel; a reducer merges their outputs.

Why this target: every other target is sequential — one agent acts, then the
next. This one has genuinely CONCURRENT writers to shared state. That is the
classic setting for lost updates and merge-order bugs, and it exercises a
different part of drift's trace capture (a single super-step containing many
parallel node executions rather than one).

Uses `Send` so the worker count is dynamic, decided at runtime from state —
which also means chaos that alters the subtask list changes the shape of the
graph execution itself, not just its values.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from . import _tools

NAME = "map_reduce_fanout"
DESCRIPTION = "dispatcher fans out N parallel workers via Send, reducer merges"
TOPOLOGY = "fanout"
SOURCE = "LangGraph Send API (map-reduce pattern)"
REQUIRES = ["langchain_openai", "langgraph"]
NEEDS_API_KEY = True
VISIBLE_TO_DRIFT = True
DEFAULT_QUERY = "Compare Python, Rust, and Go for building a web backend."


class _State(TypedDict):
    topic: str
    subtasks: list[str]
    findings: Annotated[list[str], operator.add]
    final: str


class _WorkerState(TypedDict):
    subtask: str


def build(model: str = "gpt-4o-mini"):
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    from langgraph.constants import Send
    from langgraph.graph import END, START, StateGraph

    llm = ChatOpenAI(model=model)

    def dispatcher(state: _State) -> dict:
        reply = llm.invoke([HumanMessage(content=(
            "Split this into exactly 3 short independent research subtasks, "
            "one per line, no numbering or commentary.\n\n"
            f"Topic: {state['topic']}"
        ))])
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        subs = [ln.strip() for ln in text.splitlines() if ln.strip()][:3]
        return {"subtasks": subs}

    def fan_out(state: _State):
        """Dynamic edge — one worker per subtask. Chaos here changes graph shape."""
        subs = state.get("subtasks") or []
        if not subs:
            return END
        return [Send("worker", {"subtask": s}) for s in subs]

    def worker(state: _WorkerState) -> dict:
        sub = state["subtask"]
        reply = llm.invoke([HumanMessage(content=(
            f"Answer this in one sentence: {sub}"
        ))])
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        _tools.web_search(sub)
        return {"findings": [f"{sub} -> {text[:180]}"]}

    def reducer(state: _State) -> dict:
        found = state.get("findings") or []
        return {"final": _tools.summarize(" | ".join(found))}

    g = StateGraph(_State)
    g.add_node("dispatcher", dispatcher)
    g.add_node("worker", worker)
    g.add_node("reducer", reducer)
    g.add_edge(START, "dispatcher")
    g.add_conditional_edges("dispatcher", fan_out, ["worker", END])
    g.add_edge("worker", "reducer")
    g.add_edge("reducer", END)
    return g.compile()


def initial_state(query: str = DEFAULT_QUERY) -> dict:
    return {"topic": query, "subtasks": [], "findings": [], "final": ""}
