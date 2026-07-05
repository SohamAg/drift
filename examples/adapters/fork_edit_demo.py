"""Fork-edit-replay demo — validate that editing state at a fork point
"closes" a coordination finding on the downstream branch.

Uses the same HALLUCINATED-REFERENCE adversarial MAS as
`run_drift_on_adversarial_mas.py`. Baseline fires `hallucinated_reference`
because the worker's rationale mentions TICKET-42 with no such id anywhere
in state. If we fork at step 1 and add `ticket_id: "TICKET-42"` to the
state before the worker runs, the detector should stay silent on the
forked branch — the id is now grounded.

Not a test (no assertions or pytest harness). A runnable demo of the
diagnostic loop: run → see finding → fork with a hypothesized fix →
verify finding goes away.

Run:
    PYTHONPATH=src python examples/adapters/fork_edit_demo.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*langgraph.*")

from drift.adapters.langgraph import drift_test, drift_test_fork  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_drift_on_adversarial_mas import _build_hallucination_mas  # noqa: E402


def _summarize_findings(findings: list[dict]) -> str:
    if not findings:
        return "(none)"
    return ", ".join(f"[{f['failure_type']}]" for f in findings)


def main() -> None:
    graph = _build_hallucination_mas()

    print("=" * 74)
    print("fork-edit-replay demo — hallucinated_reference MAS")
    print("=" * 74)

    # 1) Baseline run.
    print("\n[1] baseline drift_test — should fire hallucinated_reference")
    parent = drift_test(
        graph=graph,
        initial_state={"task": "clear the queue"},
        intensity="off",
        seed=1,
    )
    print(f"    baseline trace: {len(parent.baseline.trace)} step(s)")
    print(f"    baseline coord findings: "
          f"{_summarize_findings(parent.baseline.coordination_findings)}")

    if not any(f["failure_type"] == "hallucinated_reference"
               for f in parent.baseline.coordination_findings):
        print("    UNEXPECTED: baseline didn't fire the detector; demo assumptions broken")
        sys.exit(1)

    # 2) Fork at step 1 (after intake), inject ticket_id into state.
    #    Diagnostic hypothesis: "if the intake had populated ticket_id, the
    #    worker's mention would be grounded and the detector wouldn't fire."
    print("\n[2] fork at step 1, inject ticket_id='TICKET-42' into state")
    fork = drift_test_fork(
        graph=graph,
        parent_result=parent,
        fork_step=1,
        edits={"ticket_id": "TICKET-42"},
        also_apply_at_initial=True,   # also demonstrate top-vs-bottom compare
    )
    print(f"    fork branch trace: {len(fork.fork_branch.trace)} step(s)")
    print(f"    fork branch coord findings: "
          f"{_summarize_findings(fork.fork_branch.coordination_findings)}")
    print(f"    top-edited branch coord findings: "
          f"{_summarize_findings(fork.top_edited_branch.coordination_findings)}")

    # 3) Verdict.
    fork_still_fires = any(
        f["failure_type"] == "hallucinated_reference"
        for f in fork.fork_branch.coordination_findings
    )
    top_still_fires = any(
        f["failure_type"] == "hallucinated_reference"
        for f in fork.top_edited_branch.coordination_findings
    )
    print("\n[3] verdict:")
    print(f"    fork-edit at step 1  : {'STILL FIRES' if fork_still_fires else 'CLOSED'}")
    print(f"    initial-state edit    : {'STILL FIRES' if top_still_fires else 'CLOSED'}")
    if not fork_still_fires and not top_still_fires:
        print("    -> both branches close the finding — the fix is legitimate at")
        print("       both the design (initial state) and the local (fork) level.")
    elif not fork_still_fires and top_still_fires:
        print("    -> fork closes it but initial-state edit doesn't. Path-dependence:")
        print("       the fix works locally but wouldn't survive a redesign at initial")
        print("       state (something upstream re-hallucinates).")
    else:
        print("    -> the fix didn't close the finding; hypothesis was wrong.")

    print(f"\n    fork duration: {fork.duration_s:.3f}s (both branches combined)")


if __name__ == "__main__":
    main()
