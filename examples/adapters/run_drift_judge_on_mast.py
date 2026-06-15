"""Run drift's LLM judge across the MAST human-labelled dataset.

MAST (Cemri et al., Berkeley, arXiv:2503.13657) publishes 19 multi-agent
traces with binary failure-mode annotations from 3 human raters per mode.
This is exactly the validation ground truth the MAESTRO experiment lacked.

Differences from the MAESTRO runner:
  - MAST traces are unstructured text (per-MAS formats differ wildly), so
    we skip drift's structured analyze pipeline. We send the raw trajectory
    directly to the OpenAIJudge with a MAST-shaped system prompt.
  - The judge taxonomy is set per-trace from that trace's own annotations,
    so we get clean apples-to-apples comparisons with the human labels.
  - Ground truth is majority vote (≥2 of 3 annotators).
  - Long traces are truncated to MAX_TRACE_CHARS; we keep the head of the
    trajectory because failures usually surface as the run progresses but
    we'd rather miss late failures than blow the context window.

Cost at gpt-4o-mini rates: ~$0.07 for the full 19-trace run.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from drift.failures.judge import OpenAIJudge  # noqa: E402
from drift.failures.mast_eval import judge_one_trace as _judge_one  # noqa: E402

MAST_DATASET = REPO_ROOT / "data" / "external" / "mast" / "MAD_human_labelled_dataset.json"
RESULTS_DIR = REPO_ROOT / "results" / "mast_judge"


async def _run_all(
    records: list[dict],
    judge_model: str,
    concurrency: int,
    user_guidelines: list[str] | None = None,
) -> list[dict]:
    judge = OpenAIJudge(model=judge_model)
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def _bounded(r: dict) -> dict:
        nonlocal done
        async with sem:
            res = await _judge_one(r, judge, user_guidelines=user_guidelines)
        done += 1
        print(f"  {done}/{len(records)} judged  (trace_id={r['trace_id']} {r['mas_name']})", file=sys.stderr)
        return res

    return await asyncio.gather(*(_bounded(r) for r in records))


def _tabulate(results: list[dict]) -> dict:
    errored = sum(1 for r in results if "error" in r)
    tp = fp = fn = tn = 0
    by_mode: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        if "error" in r:
            continue
        for m in r["per_mode"]:
            tp += (m["outcome"] == "TP")
            fp += (m["outcome"] == "FP")
            fn += (m["outcome"] == "FN")
            tn += (m["outcome"] == "TN")
            by_mode[f'{m["mode_id"]} {m["name"]}'][m["outcome"]] += 1

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None

    # Per-mode breakdown (only modes that actually have ground-truth positives
    # are interesting for recall; modes with all-zero ground truth tell you
    # nothing about whether the judge is sensitive — only whether it's quiet).
    per_mode_report = []
    for mode_label, c in sorted(by_mode.items()):
        gt_pos = c["TP"] + c["FN"]
        pred_pos = c["TP"] + c["FP"]
        prec = c["TP"] / pred_pos if pred_pos else None
        rec = c["TP"] / gt_pos if gt_pos else None
        per_mode_report.append({
            "mode": mode_label,
            "tp": c["TP"], "fp": c["FP"], "fn": c["FN"], "tn": c["TN"],
            "ground_truth_positives": gt_pos,
            "precision": prec,
            "recall": rec,
        })

    return {
        "n_traces": len(results),
        "n_errored": errored,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_mode": per_mode_report,
    }


def _load_guidelines(path: str | None) -> list[str]:
    """Load guidelines from a file: one plain-English pattern per non-blank line.
    Lines starting with `#` are treated as comments."""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        sys.exit(f"guidelines file not found: {path}")
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=int, default=None, help="run on the first N traces only (smoke test)")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--output-name", default=None)
    p.add_argument(
        "--guidelines-file", default=None,
        help="path to a file with user guidelines (one per non-blank line, # for comments). "
             "Appended to the judge's system prompt for every trace; lets you measure F1 delta "
             "vs the no-guideline baseline (CASE_STUDY_MAST.md = F1 0.16).",
    )
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set; expected in .env or environment")
    if not MAST_DATASET.exists():
        sys.exit(f"missing {MAST_DATASET}; download MAD_human_labelled_dataset.json first")

    records = json.loads(MAST_DATASET.read_text(encoding="utf-8"))
    if args.sample is not None:
        records = records[: args.sample]

    user_guidelines = _load_guidelines(args.guidelines_file)

    label = args.output_name or time.strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"running drift's LLM judge over {len(records)} MAST human-labelled traces", file=sys.stderr)
    print(f"  judge model      : {args.judge_model}", file=sys.stderr)
    print(f"  concurrency      : {args.concurrency}", file=sys.stderr)
    print(f"  output dir       : {out_dir.relative_to(REPO_ROOT)}", file=sys.stderr)
    if user_guidelines:
        print(f"  guidelines       : {len(user_guidelines)} from {args.guidelines_file}", file=sys.stderr)
    print(file=sys.stderr)

    t0 = time.perf_counter()
    results = asyncio.run(_run_all(
        records, args.judge_model, args.concurrency,
        user_guidelines=user_guidelines or None,
    ))
    elapsed = time.perf_counter() - t0

    for r in results:
        if "error" in r:
            continue
        (out_dir / f"trace_{r['trace_id']:03d}.json").write_text(
            json.dumps(r, indent=2, default=str), encoding="utf-8"
        )

    summary = _tabulate(results)
    summary["elapsed_seconds"] = round(elapsed, 1)
    summary["judge_model"] = args.judge_model
    summary["n_records"] = len(records)
    summary["user_guidelines"] = user_guidelines
    summary["n_user_guidelines"] = len(user_guidelines)
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"=== drift LLM-judge x MAST empirical results ===")
    print(f"  traces judged : {summary['n_traces']}  ({summary['n_errored']} errored)")
    print(f"  elapsed       : {summary['elapsed_seconds']}s")
    print(f"  judge model   : {summary['judge_model']}")
    print()
    print(f"  Confusion matrix (across all mode-trace pairs):")
    print(f"    TP = {summary['tp']}    FP = {summary['fp']}")
    print(f"    FN = {summary['fn']}    TN = {summary['tn']}")
    print()
    if summary["precision"] is not None:
        print(f"  precision = {summary['precision']:.3f}")
    if summary["recall"] is not None:
        print(f"  recall    = {summary['recall']:.3f}")
    if summary["f1"] is not None:
        print(f"  F1        = {summary['f1']:.3f}")
    print()
    print(f"  Per-mode breakdown (sorted by ground-truth positives, modes with positives only):")
    for m in sorted(summary["per_mode"], key=lambda x: -x["ground_truth_positives"]):
        if m["ground_truth_positives"] == 0:
            continue
        prec = f"{m['precision']:.2f}" if m["precision"] is not None else "  - "
        rec = f"{m['recall']:.2f}" if m["recall"] is not None else "  - "
        print(f"    [gt+={m['ground_truth_positives']:2d}] TP={m['tp']} FP={m['fp']} FN={m['fn']}  "
              f"P={prec} R={rec}  {m['mode']}")
    print()
    print(f"per-trace details: {out_dir.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
