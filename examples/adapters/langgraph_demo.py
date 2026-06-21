"""LangGraph adapter demo — run drift's auto-chaos against a tiny graph.

What this shows: drift takes your compiled LangGraph (or any object with
.invoke / .ainvoke), enumerates schema-driven chaos perturbations from
your initial state, and reports which perturbations crashed your graph
vs silently changed its output.

The graph below is a 3-node ticket triage:
    classify -> route -> respond
classify reads `text` and sets `priority`. route decides whether to
escalate based on priority + the user's `is_premium` flag. respond writes
a final `reply`. Both branches expect `open_tickets[ticket_id]` to exist —
chaos that removes the ticket from the dict mid-state should crash respond,
which is exactly the kind of subtle production failure drift exists to
surface.

Run from project root with either:

    PYTHONPATH=src python examples/adapters/langgraph_demo.py

If langgraph is installed, drift drives the real compiled graph. If not,
the demo falls back to a hand-rolled equivalent with the same shape — the
adapter contract is "anything with .invoke(state) -> dict" so the demo
runs either way.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from drift.adapters.langgraph import drift_test  # noqa: E402


# ---------------------------------------------------------------------------
# Build the graph. Real langgraph if available; otherwise an equivalent.
# ---------------------------------------------------------------------------

def _classify(state: dict) -> dict:
    text = state.get("text", "")
    if "urgent" in text.lower() or "down" in text.lower():
        priority = "high"
    elif "question" in text.lower():
        priority = "low"
    else:
        priority = "normal"
    return {"priority": priority}


def _route(state: dict) -> str:
    if state.get("priority") == "high" or state.get("is_premium"):
        return "escalate"
    return "respond"


def _escalate(state: dict) -> dict:
    ticket_id = state["ticket_id"]
    ticket = state["open_tickets"][ticket_id]  # KeyError if removed by chaos
    return {
        "reply": f"escalated ticket {ticket_id} ({ticket['issue']}) to on-call",
        "escalated": True,
    }


def _respond(state: dict) -> dict:
    ticket_id = state["ticket_id"]
    ticket = state["open_tickets"][ticket_id]  # KeyError if removed by chaos
    return {
        "reply": f"resolved ticket {ticket_id}: {ticket['issue']}",
        "escalated": False,
    }


def _build_graph() -> Any:
    """Compile via langgraph if available; fall back to a hand-rolled equivalent."""
    try:
        from langgraph.graph import END, START, StateGraph  # type: ignore
    except ImportError:
        # Hand-rolled equivalent — identical behavior, identical interface
        # (.invoke(state) -> dict). Lets the demo run without langgraph
        # installed so users can see the adapter in action immediately.
        class _ManualGraph:
            def invoke(self, state: dict) -> dict:
                merged = dict(state)
                merged.update(_classify(merged))
                if _route(merged) == "escalate":
                    merged.update(_escalate(merged))
                else:
                    merged.update(_respond(merged))
                return merged

        return _ManualGraph()

    # Real langgraph build. State schema is untyped (uses dict) so chaos
    # can mutate any field at runtime.
    graph = StateGraph(dict)
    graph.add_node("classify", _classify)
    graph.add_node("escalate", _escalate)
    graph.add_node("respond", _respond)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", _route, {
        "escalate": "escalate",
        "respond": "respond",
    })
    graph.add_edge("escalate", END)
    graph.add_edge("respond", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Run drift_test against the graph.
# ---------------------------------------------------------------------------

def main() -> None:
    graph = _build_graph()
    using_langgraph = "ManualGraph" not in type(graph).__name__

    initial_state = {
        "ticket_id": "TKT-42",
        "text": "site is down can someone help",
        "is_premium": True,
        "open_tickets": {
            "TKT-42": {"issue": "checkout 500s", "customer": "acme"},
        },
        "reply": "",
        "escalated": False,
        "priority": "",
    }

    backend = "langgraph (real)" if using_langgraph else "hand-rolled fallback (langgraph not installed)"
    print(f"graph backend : {backend}")
    print(f"initial state : {initial_state}")
    print()

    result = drift_test(
        graph=graph,
        initial_state=initial_state,
        intensity="aggressive",
        seed=7,
    )

    for line in result.summary_lines():
        print(line)
    print()

    # Baseline.
    if result.baseline.crashed:
        print(f"baseline: CRASHED {result.baseline.error_type}: {result.baseline.error}")
    else:
        print("baseline OK:")
        for k, v in result.baseline.final_state.items():
            print(f"  {k}: {v!r}")
    print()

    if not result.perturbations:
        print("(no perturbations scheduled — try intensity='aggressive' or a richer state schema)")
        return

    # Sort: crashes first (highest signal for a developer), then divergences,
    # then unchanged.
    def _sort_key(p):
        return (0 if p.crashed else 1 if p.diverged else 2, p.event_name)

    print(f"per-perturbation results (sorted by severity):")
    for p in sorted(result.perturbations, key=_sort_key):
        if p.crashed:
            tag = f"CRASH   {p.error_type}"
            detail = p.error
        elif p.diverged:
            tag = "DIVERGE"
            detail = p.divergence_summary
        else:
            tag = "UNCHANGED"
            detail = "(graph absorbed the perturbation without observable change)"

        print(f"  [{tag:>9s}] {p.event_name}")
        print(f"             {p.event_summary}")
        print(f"             {detail}")
        print()


if __name__ == "__main__":
    main()
