"""A deterministic two-node LangGraph app for exercising drift's CI path
without an API key: the GitHub Action's self-test runs against it.

triage → reply. `reply` trusts `priority` to be one of the known levels and
raises on anything else — the kind of unchecked handoff assumption chaos on
the `priority` field exposes. Exhaustive chaos therefore produces real
crashes; quick chaos with a fixed seed may or may not hit that field, which
is why the self-test that expects a failure uses the exhaustive intensity.

    drift run examples.ci.demo_app --intensity exhaustive --fail-on crash   # fails
    drift run examples.ci.demo_app --fail-on finding:injected_instruction_followed  # passes
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

SLA_HOURS = {"low": 72, "normal": 24, "high": 4}


class TicketState(TypedDict):
    ticket: str
    customer_tier: str
    priority: str
    reply: str
    escalated: bool


def triage(state: TicketState) -> dict:
    text = state["ticket"].lower()
    if state["customer_tier"] == "enterprise" or "outage" in text:
        return {"priority": "high"}
    return {"priority": state["priority"] or "normal"}


def reply(state: TicketState) -> dict:
    hours = SLA_HOURS[state["priority"]]  # KeyError on any priority triage didn't produce
    return {
        "reply": f"Thanks — we'll respond within {hours}h.",
        "escalated": state["priority"] == "high",
    }


def build(model: str | None = None):
    sg = StateGraph(TicketState)
    sg.add_node("triage", triage)
    sg.add_node("reply", reply)
    sg.add_edge(START, "triage")
    sg.add_edge("triage", "reply")
    sg.add_edge("reply", END)
    return sg.compile()


def initial_state(query: str | None = None) -> TicketState:
    return {
        "ticket": query or "Login page is slow since this morning",
        "customer_tier": "pro",
        "priority": "",
        "reply": "",
        "escalated": False,
    }
