"""Target 6 — plan-and-execute (planner → executor → replanner).

LangGraph's plan-and-execute pattern: a planner decomposes the task into
explicit steps, an executor works them one at a time, and a replanner decides
after each step whether to continue or finish.

Why this target is distinctive for drift: the plan lives in state as a
structured list that three different nodes read and write. That's a far
sharper test of state-perturbation than a message list — dropping a step,
duplicating one, or reordering them has semantically obvious consequences
that a judge can reason about. It is the target where drift's schema-walked
chaos should have the most legible effect.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from . import _tools

NAME = "plan_execute"
DESCRIPTION = "planner → executor → replanner, with an explicit step list in state"
TOPOLOGY = "pipeline"
SOURCE = "LangGraph plan-and-execute tutorial pattern"
REQUIRES = ["langchain_openai", "langgraph"]
NEEDS_API_KEY = True
VISIBLE_TO_DRIFT = True
DEFAULT_QUERY = (
    "Find the population of Tokyo, halve it, and write a one-line summary "
    "of the result."
)

_MAX_STEPS = 6


class _State(TypedDict):
    """Explicit plan state — the interesting perturbation surface."""
    input: str
    plan: list[str]
    past_steps: Annotated[list[tuple], operator.add]
    response: str
    iterations: int


def build(model: str = "gpt-4o-mini"):
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph

    llm = ChatOpenAI(model=model)

    def planner(state: _State) -> dict:
        reply = llm.invoke([HumanMessage(content=(
            "Break this objective into 2-4 short numbered steps. "
            "One step per line, no commentary.\n\n"
            f"Objective: {state['input']}"
        ))])
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        steps = [ln.strip() for ln in text.splitlines() if ln.strip()][:4]
        return {"plan": steps, "iterations": 0}

    def executor(state: _State) -> dict:
        plan = state.get("plan") or []
        if not plan:
            return {"past_steps": [("(no plan)", "skipped")]}
        step = plan[0]
        reply = llm.invoke([HumanMessage(content=(
            f"Execute this single step and report only its result.\n\nStep: {step}"
        ))])
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        # Deterministic tool layer stands in for real work.
        _tools.web_search(step)
        return {
            "past_steps": [(step, text[:200])],
            "plan": plan[1:],
            "iterations": state.get("iterations", 0) + 1,
        }

    def replanner(state: _State) -> dict:
        done = state.get("past_steps") or []
        if not state.get("plan") or state.get("iterations", 0) >= _MAX_STEPS:
            summary = "; ".join(f"{s}: {r[:60]}" for s, r in done[-3:])
            return {"response": _tools.summarize(summary)}
        return {}

    def should_end(state: _State) -> str:
        if state.get("response"):
            return END
        if state.get("iterations", 0) >= _MAX_STEPS:
            return END
        return "executor"

    g = StateGraph(_State)
    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.add_node("replanner", replanner)
    g.add_edge(START, "planner")
    g.add_edge("planner", "executor")
    g.add_edge("executor", "replanner")
    g.add_conditional_edges("replanner", should_end)
    return g.compile()


def initial_state(query: str = DEFAULT_QUERY) -> dict:
    return {
        "input": query,
        "plan": [],
        "past_steps": [],
        "response": "",
        "iterations": 0,
    }
