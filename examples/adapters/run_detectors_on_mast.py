"""Run drift's coordination-detector library against the MAST human-labelled dataset.

Computes per-detector precision/recall on the 19-trace MAD subset, using each
detector's `detect_from_text` variant (the structured `detect()` path can't be
used because MAST traces are unstructured text transcripts).

Ground truth alignment: each detector declares `MAST_MODES` — the list of MAST
mode IDs it targets. For a given trace, the detector's ground truth label is
the OR of the majority-vote (>=2/3 annotators) annotation across those modes.

Caveat (printed at the end too): MAST modes are coarser than our named
detectors. `verifier_always_approves` maps to MAST 3.3/4.2/4.3 (incorrect /
lack of verification), but MAST can fire those modes for reasons that aren't
"approves everything." Treat the F1 numbers as a directional anchor, NOT as
absolute proof of detector quality. Synthetic positive/negative pairs in
tests/test_library_detectors.py are the cleaner specificity test; this run
shows whether the text-variant catches anything on real, unstructured traces.

No LLM cost — runs entirely locally over the bundled dataset. Compared with
the F1 = 0.16 generic LLM-judge baseline (CASE_STUDY_MAST.md), this is the
free-detector floor.

Usage:
    PYTHONPATH=src python examples/adapters/run_detectors_on_mast.py
    PYTHONPATH=src python examples/adapters/run_detectors_on_mast.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from drift.failures.library import ALL_DETECTORS  # noqa: E402
from drift.failures.mast_eval import majority_vote, parse_mode_id  # noqa: E402

MAST_DATASET = REPO_ROOT / "data" / "external" / "mast" / "MAD_human_labelled_dataset.json"


def _ground_truth_for_detector(record: dict, mast_modes: list[str]) -> bool:
    """OR of majority-vote labels across the detector's targeted MAST modes."""
    targeted = set(mast_modes)
    for ann in record.get("annotations", []):
        mid = parse_mode_id(ann["failure mode"])
        if mid in targeted and majority_vote(ann):
            return True
    return False


def _evaluate(records: list[dict]) -> dict:
    """Run every library detector over every trace; compute per-detector confusion."""
    per_detector: dict[str, dict] = {}
    per_trace_detail: list[dict] = []

    for det in ALL_DETECTORS:
        per_detector[det.NAME] = {
            "name": det.NAME,
            "mast_modes": list(det.MAST_MODES),
            "source": det.SOURCE,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "fired_on_traces": [],
            "missed_on_traces": [],
        }

    for record in records:
        trace_text = record.get("trace", "") or ""
        trace_detail = {
            "trace_id": record["trace_id"],
            "mas_name": record["mas_name"],
            "detectors": [],
        }
        for det in ALL_DETECTORS:
            findings = det.detect_from_text(trace_text)
            predicted = bool(findings)
            actual = _ground_truth_for_detector(record, det.MAST_MODES)
            outcome = (
                "TP" if predicted and actual else
                "FP" if predicted and not actual else
                "FN" if not predicted and actual else
                "TN"
            )
            per_detector[det.NAME][outcome.lower()] += 1
            if predicted:
                per_detector[det.NAME]["fired_on_traces"].append({
                    "trace_id": record["trace_id"],
                    "mas_name": record["mas_name"],
                    "actual": actual,
                    "evidence": findings[0].summary[:160],
                })
            elif actual:
                per_detector[det.NAME]["missed_on_traces"].append({
                    "trace_id": record["trace_id"],
                    "mas_name": record["mas_name"],
                })
            trace_detail["detectors"].append({
                "detector": det.NAME,
                "predicted": predicted,
                "actual": actual,
                "outcome": outcome,
            })
        per_trace_detail.append(trace_detail)

    # Per-detector precision/recall/F1.
    for d in per_detector.values():
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        d["precision"] = tp / (tp + fp) if (tp + fp) else None
        d["recall"] = tp / (tp + fn) if (tp + fn) else None
        p, r = d["precision"], d["recall"]
        d["f1"] = (2 * p * r / (p + r)) if (p and r) else None

    # Aggregate (sum confusion across detectors — treats each detector-trace pair
    # as one classification, comparable shape to the judge's per-mode aggregate).
    agg_tp = sum(d["tp"] for d in per_detector.values())
    agg_fp = sum(d["fp"] for d in per_detector.values())
    agg_fn = sum(d["fn"] for d in per_detector.values())
    agg_tn = sum(d["tn"] for d in per_detector.values())
    agg_p = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) else None
    agg_r = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) else None
    agg_f1 = (2 * agg_p * agg_r / (agg_p + agg_r)) if (agg_p and agg_r) else None

    return {
        "n_traces": len(records),
        "n_detectors": len(ALL_DETECTORS),
        "per_detector": per_detector,
        "per_trace": per_trace_detail,
        "aggregate": {
            "tp": agg_tp, "fp": agg_fp, "fn": agg_fn, "tn": agg_tn,
            "precision": agg_p, "recall": agg_r, "f1": agg_f1,
        },
    }


def _print_report(summary: dict) -> None:
    print()
    print("=== drift coordination-detector library × MAST empirical results ===")
    print(f"  traces            : {summary['n_traces']}")
    print(f"  detectors         : {summary['n_detectors']} ({', '.join(d for d in summary['per_detector'])})")
    print(f"  mode              : detect_from_text (raw-text variant; structured")
    print(f"                      detect() needs adapter trace shape MAST doesn't ship)")
    print()
    agg = summary["aggregate"]
    print(f"  Aggregate confusion (across all detector × trace pairs):")
    print(f"    TP = {agg['tp']}    FP = {agg['fp']}")
    print(f"    FN = {agg['fn']}    TN = {agg['tn']}")
    print()
    def _fmt(x: float | None) -> str:
        return f"{x:.3f}" if x is not None else "  - "
    print(f"  aggregate precision = {_fmt(agg['precision'])}")
    print(f"  aggregate recall    = {_fmt(agg['recall'])}")
    print(f"  aggregate F1        = {_fmt(agg['f1'])}")
    print()
    print("  Per-detector breakdown:")
    for name, d in summary["per_detector"].items():
        modes = ",".join(d["mast_modes"])
        gt_pos = d["tp"] + d["fn"]
        print(
            f"    [{name}]  TP={d['tp']} FP={d['fp']} FN={d['fn']} TN={d['tn']}  "
            f"P={_fmt(d['precision'])} R={_fmt(d['recall'])} F1={_fmt(d['f1'])}  "
            f"(mast={modes}, gt+={gt_pos})"
        )
        if d["fired_on_traces"]:
            print(f"      fired on:")
            for f in d["fired_on_traces"][:4]:
                ok = "✓" if f["actual"] else "✗"
                print(f"        {ok} trace_{f['trace_id']} ({f['mas_name']}): {f['evidence']}")
            if len(d["fired_on_traces"]) > 4:
                print(f"        ... and {len(d['fired_on_traces']) - 4} more")
    print()
    print("  Comparison anchor: generic LLM judge on same MAST subset = F1 0.16")
    print("                     (per CASE_STUDY_MAST.md; gpt-4o-mini, ~$0.07/run)")
    print()
    print("  Honest read of these numbers:")
    print("    The text-variant detectors score near-zero recall on MAST. This is NOT")
    print("    a bug — it's the expected ceiling of text-pattern heuristics on this")
    print("    dataset, and surfaces a real diagnosis:")
    print("      - MAST traces are framework-diverse (ChatDev/MetaGPT/AppWorld/...)")
    print("        and predominantly code-execution + system messages, not the")
    print("        chat-style 'Reviewer: approved' dialogue our text patterns assume.")
    print("      - MAST mode 3.3 ('Incorrect Verification') in an AppWorld trace")
    print("        means an agent didn't sanity-check its own code output — there")
    print("        is no verifier-role agent at all.")
    print("      - MAST mode 1.3 ('Step Repetition') is broader than two-agent")
    print("        bouncing; often it's one agent repeating itself.")
    print("    Implication: text-variants are useful as a quick sanity layer but")
    print("    cannot beat the LLM judge on framework-diverse transcripts. The")
    print("    structured detect() path running over adapter traces is the actual")
    print("    production target; synthetic structured traces (next: build a small")
    print("    labelled set of langgraph fixtures) are the right validation surface.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true", help="emit full per-trace JSON to stdout instead of pretty report")
    p.add_argument("--sample", type=int, default=None, help="run on the first N traces only")
    args = p.parse_args()

    if not MAST_DATASET.exists():
        sys.exit(
            f"MAST dataset not found at {MAST_DATASET}.\n"
            "Download MAD_human_labelled_dataset.json from "
            "https://huggingface.co/datasets/mcemri/MAST-Data into data/external/mast/."
        )

    records = json.loads(MAST_DATASET.read_text(encoding="utf-8"))
    if args.sample is not None:
        records = records[: args.sample]

    summary = _evaluate(records)

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
        return
    _print_report(summary)


if __name__ == "__main__":
    main()
