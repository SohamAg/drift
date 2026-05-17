"""Run drift's detectors across every converted MAESTRO trace and tabulate.

Walks data/external/maestro/drift_traces/, runs `drift.analyze.analyze_trace`
on each file under each of drift's three shipped topologies (support,
code_review, ops), and prints a summary of which detectors fired, on which
MAESTRO example types, how often.

This is the empirical-validation step: are drift's detectors finding real
coordination failures in published multi-agent traces, or only in drift's
own synthetic scenarios?
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Make src/ importable when run from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from drift.analyze import analyze_trace
from drift.topologies import list_topologies


TRACES_DIR = REPO_ROOT / "data" / "external" / "maestro" / "drift_traces"
TOPOLOGIES = ["support", "code_review", "ops"]


def main() -> None:
    if not TRACES_DIR.exists():
        sys.exit(f"missing {TRACES_DIR}; run maestro_to_drift.py first")

    trace_files = sorted(TRACES_DIR.glob("*.jsonl"))
    print(f"found {len(trace_files):,} converted MAESTRO traces")
    print(f"running drift's detectors under each of: {TOPOLOGIES}")
    print()

    # Per-topology: total failures, failure-type counts, per-MAESTRO-example counts
    overall: dict[str, dict] = {
        t: {
            "total_failures": 0,
            "by_type": Counter(),
            "by_example": defaultdict(Counter),
            "files_with_any_failure": 0,
        }
        for t in TOPOLOGIES
    }
    errors = 0
    for idx, path in enumerate(trace_files, start=1):
        if idx % 100 == 0:
            print(f"  processed {idx} / {len(trace_files)}")

        # Recover the MAESTRO example name from the first snapshot record.
        with path.open(encoding="utf-8") as fh:
            first = json.loads(fh.readline())
        example_name = "unknown"
        case = next(iter(first.get("open_cases", {}).values()), None)
        if case:
            example_name = case.get("issue", "unknown")

        for topology in TOPOLOGIES:
            try:
                failures, _ = analyze_trace(path, topology)
            except Exception as e:
                errors += 1
                continue
            if failures:
                overall[topology]["files_with_any_failure"] += 1
            for f in failures:
                overall[topology]["total_failures"] += 1
                overall[topology]["by_type"][f.failure_type] += 1
                overall[topology]["by_example"][example_name][f.failure_type] += 1

    print(f"\n=== drift x MAESTRO empirical results ===")
    print(f"traces analyzed: {len(trace_files):,}")
    print(f"errors during analyze: {errors}")
    print()

    for topology in TOPOLOGIES:
        s = overall[topology]
        print(f"--- topology: {topology} ---")
        print(f"  files with >=1 failure: {s['files_with_any_failure']:,} / {len(trace_files):,}")
        print(f"  total failures detected: {s['total_failures']:,}")
        if s["by_type"]:
            print(f"  by detector:")
            for ftype, n in s["by_type"].most_common():
                print(f"    {n:6d}  {ftype}")
        else:
            print(f"  no detectors fired on any trace.")
        print()

    # Cross-cut: which MAESTRO examples produced the most failures?
    print("=== failures by MAESTRO example (any topology) ===")
    by_example_total: Counter = Counter()
    for topology in TOPOLOGIES:
        for ex, ctr in overall[topology]["by_example"].items():
            by_example_total[ex] += sum(ctr.values())
    for ex, n in by_example_total.most_common():
        print(f"  {n:6d}  {ex}")


if __name__ == "__main__":
    main()
