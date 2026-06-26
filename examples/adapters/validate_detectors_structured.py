"""Structured-trace validation for the coordination-detector library.

This is the apples-to-apples validation for the production path: each detector
runs on hand-labelled langgraph-shaped traces (the same shape
`drift.adapters.langgraph._stream_or_invoke` emits during a real
`drift_test`). Computes per-detector precision/recall/F1.

Why not just use MAST? MAST traces are unstructured framework-diverse text
(see run_detectors_on_mast.py for the empirical recall ceiling on raw text).
The detector library's structured `detect()` path needs a list of
{step, node, update, state_after} records, which MAST doesn't ship. So we
hand-built a small labelled corpus that covers:

  - True positives for each detector (the failure DOES occur)
  - Hard negatives (structurally similar trace where the failure does NOT
    occur — designed to make a naive detector misfire)
  - Cross-checks (a positive for detector A must not trigger detector B)

Every fixture's source is cited in its `notes` field — adapted from MAST mode
descriptions, Anthropic's postmortem, or Cognition's named open problems.

Usage:
    PYTHONPATH=src python examples/adapters/validate_detectors_structured.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from drift.failures.library import ALL_DETECTORS  # noqa: E402
from drift.failures.library.base import from_adapter_trace  # noqa: E402


@dataclass
class Fixture:
    """One labelled trace fixture."""
    name: str
    trace: list[dict]
    expected_detectors: set[str]   # which library detectors should fire
    initial_state: dict | None = None
    roles_by_agent: dict[str, str] = field(default_factory=dict)
    notes: str = ""                # source citation / why this is positive/negative


def _trace(steps: list[tuple[str, dict]], start_state: dict | None = None) -> list[dict]:
    """Build {step, node, update, state_after} from (node, update) pairs."""
    running = dict(start_state or {})
    out = []
    for i, (node, update) in enumerate(steps, start=1):
        running = {**running, **update}
        out.append({
            "step": i,
            "node": node,
            "update": dict(update),
            "state_after": dict(running),
        })
    return out


# ===========================================================================
# Fixtures. Each labelled with `expected_detectors` = the set of detectors
# that MUST fire. Hard negatives have empty expected sets — they're traces
# that LOOK structurally similar to a positive but lack the failure signature.
# ===========================================================================


FIXTURES: list[Fixture] = [
    # ---- verifier_always_approves ----------------------------------------

    Fixture(
        name="vap_pos_structured_verdict",
        trace=_trace([
            ("planner",  {"task": "ticket-101"}),
            ("verifier", {"verdict": "approve"}),
            ("planner",  {"task": "ticket-102"}),
            ("verifier", {"verdict": "approve"}),
            ("planner",  {"task": "ticket-103"}),
            ("verifier", {"verdict": "approve"}),
            ("planner",  {"task": "ticket-104"}),
            ("verifier", {"verdict": "approve"}),
        ]),
        expected_detectors={"verifier_always_approves"},
        notes="MAST 3.3 (Incorrect Verification): verifier approves 100% with zero rejections.",
    ),

    Fixture(
        name="vap_pos_freetext_rationale",
        trace=_trace([
            ("planner",       {"task": "x"}),
            ("code_reviewer", {"rationale": "lgtm, ship it"}),
            ("planner",       {"task": "y"}),
            ("code_reviewer", {"rationale": "approved, no issues"}),
            ("planner",       {"task": "z"}),
            ("code_reviewer", {"rationale": "ok pass"}),
        ]),
        expected_detectors={"verifier_always_approves"},
        notes="MAST 4.2 (Lack of result verification): rationale-only approvals, no structured verdict.",
    ),

    Fixture(
        name="vap_neg_has_rejections",
        trace=_trace([
            ("verifier", {"verdict": "approve"}),
            ("verifier", {"verdict": "approve"}),
            ("verifier", {"verdict": "reject"}),
            ("verifier", {"verdict": "approve"}),
            ("verifier", {"verdict": "approve"}),
            ("verifier", {"verdict": "reject"}),
        ]),
        expected_detectors=set(),
        notes="Hard negative: verifier rejects some of the time — actually verifying.",
    ),

    Fixture(
        name="vap_neg_below_decision_count",
        trace=_trace([
            ("verifier", {"verdict": "approve"}),
            ("verifier", {"verdict": "approve"}),
        ]),
        expected_detectors=set(),
        notes="Hard negative: too few decisions to call it 'always.'",
    ),

    Fixture(
        name="vap_neg_planner_approves",
        trace=_trace([
            ("planner", {"verdict": "approve"}),
            ("planner", {"verdict": "approve"}),
            ("planner", {"verdict": "approve"}),
            ("planner", {"verdict": "approve"}),
            ("planner", {"verdict": "approve"}),
        ]),
        expected_detectors=set(),
        notes="Hard negative: agent name doesn't match verifier role — planner approving its own plan is normal.",
    ),

    # ---- infinite_handoff ------------------------------------------------

    Fixture(
        name="ih_pos_no_progress",
        trace=_trace([
            ("agent_a", {"thinking": "..."}),
            ("agent_b", {"thinking": "..."}),
            ("agent_a", {"thinking": "..."}),
            ("agent_b", {"thinking": "..."}),
            ("agent_a", {"thinking": "..."}),
        ], start_state={"task": "fix it", "result": ""}),
        expected_detectors={"infinite_handoff"},
        notes="MAST 1.3 + Cognition #2: A↔B bouncing 5 times, no fields populated.",
    ),

    Fixture(
        name="ih_neg_has_progress",
        trace=_trace([
            ("a", {"step1": True}),
            ("b", {"step2": True}),
            ("a", {"step3": True}),
            ("b", {"step4": True}),
            ("a", {"step5": True}),
        ], start_state={"task": "fix it"}),
        expected_detectors=set(),
        notes="Hard negative: each handoff adds a new keyed field — actual progress.",
    ),

    Fixture(
        name="ih_neg_three_agents",
        trace=_trace([
            ("a", {"x": "..."}),
            ("b", {"x": "..."}),
            ("c", {"x": "..."}),
            ("a", {"x": "..."}),
            ("b", {"x": "..."}),
            ("c", {"x": "..."}),
        ], start_state={"task": "x", "x": "..."}),
        expected_detectors=set(),
        notes="Hard negative: three-agent rotation, not a two-agent loop.",
    ),

    # ---- subagent_fanout_excess ------------------------------------------

    Fixture(
        name="sfe_pos_hard_count",
        trace=_trace(
            [("orchestrator", {"plan": "split"})] +
            [(f"worker_{i}", {f"out_{i}": f"r_{i}"}) for i in range(10)]
        ),
        expected_detectors={"subagent_fanout_excess"},
        notes="Anthropic 50-subagent incident scaled down: 10 distinct subagents > threshold 8.",
    ),

    Fixture(
        name="sfe_neg_small_mapreduce",
        trace=_trace([
            ("orchestrator", {"plan": "split"}),
            ("w1", {"r1": "ok"}),
            ("w2", {"r2": "ok"}),
            ("w3", {"r3": "ok"}),
            ("orchestrator", {"final": "merged"}),
        ]),
        expected_detectors=set(),
        notes="Hard negative: standard 3-worker map-reduce. Cognition's working pattern.",
    ),

    # ---- combined / overlap fixture --------------------------------------

    Fixture(
        name="combined_vap_plus_ih",
        trace=_trace([
            ("planner",  {"rationale": "(thinking)"}),
            ("verifier", {"verdict": "approve"}),
            ("planner",  {"rationale": "(thinking)"}),
            ("verifier", {"verdict": "approve"}),
            ("planner",  {"rationale": "(thinking)"}),
            ("verifier", {"verdict": "approve"}),
            ("planner",  {"rationale": "(thinking)"}),
            ("verifier", {"verdict": "approve"}),
        ], start_state={"task": "review x", "rationale": "(thinking)", "verdict": "approve"}),
        expected_detectors={"verifier_always_approves", "infinite_handoff"},
        notes="Both detectors fire: rubber-stamping verifier AND alternating-pair loop with no progress.",
    ),
]


# ===========================================================================
# Evaluation
# ===========================================================================


def _evaluate(fixtures: list[Fixture]) -> dict:
    per_detector: dict[str, dict] = {}
    for det in ALL_DETECTORS:
        per_detector[det.NAME] = {
            "name": det.NAME, "mast_modes": list(det.MAST_MODES), "source": det.SOURCE,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "failures": [],
        }

    per_fixture: list[dict] = []

    for fix in fixtures:
        ctx = from_adapter_trace(
            fix.trace,
            initial_state=fix.initial_state,
            roles_by_agent=fix.roles_by_agent,
        )
        fixture_row = {
            "fixture": fix.name,
            "expected": sorted(fix.expected_detectors),
            "got": [],
            "notes": fix.notes,
        }
        fired_names: set[str] = set()
        for det in ALL_DETECTORS:
            findings = det.detect(ctx)
            predicted = bool(findings)
            actual = det.NAME in fix.expected_detectors
            outcome = (
                "TP" if predicted and actual else
                "FP" if predicted and not actual else
                "FN" if not predicted and actual else
                "TN"
            )
            per_detector[det.NAME][outcome.lower()] += 1
            if outcome in ("FP", "FN"):
                per_detector[det.NAME]["failures"].append({
                    "fixture": fix.name, "outcome": outcome,
                })
            if predicted:
                fired_names.add(det.NAME)
                fixture_row["got"].append({
                    "detector": det.NAME,
                    "summary": findings[0].summary[:140],
                })
        per_fixture.append(fixture_row)

    for d in per_detector.values():
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        d["precision"] = tp / (tp + fp) if (tp + fp) else None
        d["recall"] = tp / (tp + fn) if (tp + fn) else None
        p, r = d["precision"], d["recall"]
        d["f1"] = (2 * p * r / (p + r)) if (p and r) else None

    agg_tp = sum(d["tp"] for d in per_detector.values())
    agg_fp = sum(d["fp"] for d in per_detector.values())
    agg_fn = sum(d["fn"] for d in per_detector.values())
    agg_tn = sum(d["tn"] for d in per_detector.values())
    agg_p = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) else None
    agg_r = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) else None
    agg_f1 = (2 * agg_p * agg_r / (agg_p + agg_r)) if (agg_p and agg_r) else None

    return {
        "n_fixtures": len(fixtures),
        "n_detectors": len(ALL_DETECTORS),
        "per_detector": per_detector,
        "per_fixture": per_fixture,
        "aggregate": {
            "tp": agg_tp, "fp": agg_fp, "fn": agg_fn, "tn": agg_tn,
            "precision": agg_p, "recall": agg_r, "f1": agg_f1,
        },
    }


def _fmt(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "  - "


def _print_report(summary: dict) -> None:
    print()
    print("=== drift coordination-detector library × structured-trace fixtures ===")
    print(f"  fixtures   : {summary['n_fixtures']}")
    print(f"  detectors  : {summary['n_detectors']}")
    print()
    agg = summary["aggregate"]
    print(f"  Aggregate confusion (detector × fixture pairs):")
    print(f"    TP={agg['tp']}  FP={agg['fp']}  FN={agg['fn']}  TN={agg['tn']}")
    print(f"    precision = {_fmt(agg['precision'])}")
    print(f"    recall    = {_fmt(agg['recall'])}")
    print(f"    F1        = {_fmt(agg['f1'])}")
    print()
    print("  Per-detector:")
    for name, d in summary["per_detector"].items():
        gt = d["tp"] + d["fn"]
        print(
            f"    [{name}]  TP={d['tp']} FP={d['fp']} FN={d['fn']} TN={d['tn']}  "
            f"P={_fmt(d['precision'])} R={_fmt(d['recall'])} F1={_fmt(d['f1'])}  (gt+={gt})"
        )
        for fail in d["failures"]:
            print(f"      {fail['outcome']} on fixture {fail['fixture']!r}")
    print()
    print("  Per-fixture verdict:")
    for fx in summary["per_fixture"]:
        expected = ",".join(fx["expected"]) or "(none - hard negative)"
        got = ",".join(g["detector"] for g in fx["got"]) or "(none)"
        ok = "OK" if set(fx["expected"]) == {g["detector"] for g in fx["got"]} else "FAIL"
        print(f"    [{ok:>4s}]  {fx['fixture']:<32s} expected={expected}  got={got}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true", help="emit full JSON instead of pretty report")
    args = p.parse_args()
    summary = _evaluate(FIXTURES)
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
        return
    _print_report(summary)


if __name__ == "__main__":
    main()
