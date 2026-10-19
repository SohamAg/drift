"""Regression tracking demo — catch a newly-introduced coordination failure
between two "versions" of the same graph, the way drift.regression is meant
to be used across real prompt edits.

v1's worker cites the ticket id that's actually in state (grounded). v2 is
byte-identical except the worker's rationale cites a different ticket id
that was never written anywhere — simulating what a careless prompt edit
looks like: same graph_signature, same shape, new hallucination. Both runs
are tracked under that one signature, so the second compares against the
first instead of starting fresh.

Not a test (no assertions/pytest harness). A runnable demo of the loop:
run → track → (simulated edit) → run again → track → see what moved.

Run:
    PYTHONPATH=src python examples/adapters/regression_demo.py
"""
from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*langgraph.*")

from drift.adapters.langgraph import drift_test  # noqa: E402
from drift import regression  # noqa: E402

GRAPH_SIGNATURE = "regression_demo_ticket_worker"
HISTORY_DIR = REPO_ROOT / "results" / "regression_demo" / "_history"


def _build_worker(hallucinate: bool):
    """intake writes ticket_id=TICKET-7; worker either cites it (grounded)
    or cites TICKET-42, which is never in state (hallucinated)."""
    from langgraph.graph import END, START, StateGraph

    def intake(state: dict) -> dict:
        return {"stage": "intake_done", "ticket_id": "TICKET-7"}

    def worker(state: dict) -> dict:
        cited = "TICKET-42" if hallucinate else state.get("ticket_id", "")
        return {"stage": "worker_done", "rationale": f"closed {cited} as a duplicate"}

    g = StateGraph(dict)
    g.add_node("intake", intake)
    g.add_node("worker", worker)
    g.add_edge(START, "intake")
    g.add_edge("intake", "worker")
    g.add_edge("worker", END)
    return g.compile()


def _summarize_findings(findings: list[dict]) -> str:
    if not findings:
        return "(none)"
    return ", ".join(f"[{f['failure_type']}]" for f in findings)


def main() -> None:
    # Fresh history each run so the demo is repeatable, not cumulative noise
    # across invocations.
    shutil.rmtree(HISTORY_DIR, ignore_errors=True)

    print("=" * 74)
    print("regression tracking demo — hallucinated_reference introduced between runs")
    print("=" * 74)

    print("\n[1] v1 — worker cites the real ticket_id (grounded)")
    graph_v1 = _build_worker(hallucinate=False)
    result1 = drift_test(graph=graph_v1, initial_state={}, intensity="off", seed=1)
    print(f"    coord findings: {_summarize_findings(result1.baseline.coordination_findings)}")
    report1 = regression.track(result1, GRAPH_SIGNATURE, history_dir=HISTORY_DIR)
    print(f"    regression: has_previous={report1.has_previous} "
          f"(expected: False — first tracked run)")

    print("\n[2] v2 — same graph_signature, worker now cites TICKET-42 (never in state)")
    graph_v2 = _build_worker(hallucinate=True)
    result2 = drift_test(graph=graph_v2, initial_state={}, intensity="off", seed=1)
    print(f"    coord findings: {_summarize_findings(result2.baseline.coordination_findings)}")
    report2 = regression.track(result2, GRAPH_SIGNATURE, history_dir=HISTORY_DIR)
    print(f"    regression: has_previous={report2.has_previous}, "
          f"is_regression={report2.is_regression}")
    for flag in report2.flags:
        print(f"      [{flag.severity}] {flag.summary}")

    print()
    if report2.is_regression and "hallucinated_reference" in report2.new_finding_types:
        print("PASS — regression tracking caught the newly-introduced hallucination")
    else:
        print("UNEXPECTED — regression tracking did not flag the new finding type "
              "(hallucinated_reference detector or regression.compare() may have changed)")

    print(f"\nhistory file: {HISTORY_DIR / (GRAPH_SIGNATURE + '.jsonl')}")
    print(f"terminal trend: PYTHONPATH=src python -m drift history {GRAPH_SIGNATURE} "
          f"--history-dir {HISTORY_DIR}")


if __name__ == "__main__":
    main()
