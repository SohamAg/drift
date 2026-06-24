"""LangGraph adapter demo — run drift's auto-chaos against a tiny graph.

What this shows: drift takes your compiled LangGraph (or any object with
.stream / .invoke), enumerates schema-driven chaos perturbations from
your initial state, and reports which perturbations crashed your graph,
silently diverged its output, or were absorbed. With `--judge openai`,
also runs drift's 6-family LLM coordination judge over each perturbation's
per-super-step trace.

The graph below is a 3-node ticket triage:
    classify -> route -> respond | escalate
classify reads `text` and sets `priority`. route decides whether to
escalate based on priority + the user's `is_premium` flag. Both terminal
nodes look up `open_tickets[ticket_id]` without a defensive check —
exactly the kind of subtle production failure drift exists to surface.

Run from project root:

    # default — schema-chaos + crash/diverge detection only
    PYTHONPATH=src python examples/adapters/langgraph_demo.py

    # add the LLM judge over per-perturbation traces (needs OPENAI_API_KEY)
    PYTHONPATH=src python examples/adapters/langgraph_demo.py --judge openai

If langgraph is installed, drift drives the real compiled graph (with
streaming, so traces populate). If not, the demo falls back to a hand-rolled
equivalent that also exposes `.stream()` so trace + judge still work.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from drift.adapters.langgraph import drift_test  # noqa: E402
from drift.failures.judge import build_judge  # noqa: E402


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
    """Compile via langgraph if available; fall back to a hand-rolled equivalent
    that also exposes `.stream()` so per-super-step traces populate."""
    try:
        from langgraph.graph import END, START, StateGraph  # type: ignore
    except ImportError:
        # Hand-rolled equivalent — exposes both .stream() (yields one chunk
        # per node, langgraph-shaped) and .invoke() for back-compat. Lets the
        # demo run without langgraph installed AND still produce traces the
        # judge can read.
        class _ManualGraph:
            def stream(self, state: dict):
                merged = dict(state)
                u1 = _classify(merged)
                merged.update(u1)
                yield {"classify": u1}
                if _route(merged) == "escalate":
                    u2 = _escalate(merged)
                    merged.update(u2)
                    yield {"escalate": u2}
                else:
                    u2 = _respond(merged)
                    merged.update(u2)
                    yield {"respond": u2}

            def invoke(self, state: dict) -> dict:
                out = dict(state)
                for chunk in self.stream(state):
                    for upd in chunk.values():
                        out.update(upd)
                return out

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intensity", default="aggressive",
                        choices=["off", "light", "moderate", "aggressive"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--judge", default="off",
                        choices=["off", "mock", "openai"],
                        help="enable LLM judge over per-perturbation traces. "
                             "'openai' costs a few cents and needs OPENAI_API_KEY.")
    parser.add_argument("--judge-model", default=None,
                        help="override judge model (default gpt-4o-mini for openai)")
    parser.add_argument("--guideline", action="append", default=[],
                        help="plain-English pattern to additionally watch for "
                             "(repeatable). Only effective with --judge.")
    args = parser.parse_args()

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

    judge_llm = None
    if args.judge != "off":
        if args.judge == "openai" and not os.environ.get("OPENAI_API_KEY"):
            sys.exit("--judge openai requires OPENAI_API_KEY; export it or add to .env")
        judge_llm = build_judge(args.judge, model=args.judge_model)

    backend = "langgraph (real)" if using_langgraph else "hand-rolled fallback (langgraph not installed)"
    print(f"graph backend : {backend}")
    print(f"intensity     : {args.intensity}  seed: {args.seed}")
    print(f"judge         : {args.judge}" + (f"  ({len(args.guideline)} guideline(s))" if args.guideline else ""))
    print(f"initial state : {initial_state}")
    print()

    result = drift_test(
        graph=graph,
        initial_state=initial_state,
        intensity=args.intensity,
        seed=args.seed,
        judge_llm=judge_llm,
        user_guidelines=args.guideline or None,
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
    if result.baseline.judge_findings:
        print("  judge findings on baseline:")
        for f in result.baseline.judge_findings:
            print(f"    [{f['failure_type']}] {f['summary']}")
    print()

    if not result.perturbations:
        print("(no perturbations scheduled — try a higher intensity or a richer state schema)")
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
        for f in p.judge_findings:
            print(f"             JUDGE [{f['failure_type']}] {f['summary']}")
        print()


if __name__ == "__main__":
    main()
