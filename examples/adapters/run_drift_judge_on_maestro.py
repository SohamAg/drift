"""Run drift's LLM judge across every converted MAESTRO trace and tabulate.

Companion to run_drift_on_maestro.py — that script tested whether drift's
*deterministic* detectors fired on MAESTRO traces (answer: zero, because
deterministic rules are domain-specific to drift's shipped topologies).
This script tests whether the *LLM-judged* detector (added after the
zero-fires result motivated us to build hybrid detection) fires.

What it does:
  1. Loads OPENAI_API_KEY from .env (via python-dotenv).
  2. Walks data/external/maestro/drift_traces/.
  3. For each trace, runs analyze_records with the topology's deterministic
     detectors PLUS an LLMJudgeDetector(every=1, window=10000) layered on top.
     window=10000 effectively means "the whole trace" for MAESTRO's short traces.
  4. Saves per-trace raw judge responses to results/maestro_judge/<trace>.json
     for later inspection.
  5. Tabulates: how many traces had ≥1 LLM-judged failure, by family,
     by MAESTRO example name.

Flags:
  --sample N      : run on the first N traces only (default 5 — smoke test).
                    Use --full to run on all 1,056.
  --topology X    : which topology's deterministic detectors to layer with
                    (default support; the judge is topology-agnostic so this
                    only affects deterministic fires).
  --judge-model M : OpenAI model name for the judge (default gpt-4o-mini).
  --concurrency N : how many traces to judge in parallel (default 8).

Output goes to stdout (summary table) and results/maestro_judge/ (per-trace
JSON + a `_summary.json`).

Cost: at gpt-4o-mini rates and the per-trace token estimates measured on
the 50-trace sample, full run (~1,056 traces × ~2 judge calls) is ~$0.50-1.00.
The smoke-test default keeps spend under a cent.
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

# Make src/ importable when run from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Load env BEFORE importing anything that constructs OpenAI clients.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from drift.analyze import analyze_records_async, load_trace  # noqa: E402
from drift.failures.judge import (  # noqa: E402
    JUDGE_PREFIX,
    LLMJudgeDetector,
    OpenAIJudge,
)

TRACES_DIR = REPO_ROOT / "data" / "external" / "maestro" / "drift_traces"
RESULTS_DIR = REPO_ROOT / "results" / "maestro_judge"


class _RecordingJudge:
    """Wraps OpenAIJudge so we can persist each raw response alongside the
    summary — useful for later inspection without re-spending tokens."""

    def __init__(self, inner: OpenAIJudge) -> None:
        self.inner = inner
        self.calls: list[dict] = []  # {system, user, raw, latency_s}

    async def judge(self, *, system: str, user: str) -> str:
        t0 = time.perf_counter()
        raw = await self.inner.judge(system=system, user=user)
        self.calls.append({
            "user_chars": len(user),
            "raw": raw,
            "latency_s": round(time.perf_counter() - t0, 3),
        })
        return raw


async def _judge_one_trace(
    path: Path,
    topology: str,
    model: str,
) -> dict:
    """Run analyze_records_async on one trace with a judge layered on top.

    Returns a dict with per-trace stats — det/llm failure counts, judge
    raw responses, MAESTRO example name, errors if any.
    """
    snap_raw, act_raw, evt_raw = load_trace(path)
    if not snap_raw:
        return {"path": path.name, "error": "no snapshots"}

    example_name = "unknown"
    first_case = next(iter(snap_raw[0].get("open_cases", {}).values()), None)
    if first_case:
        example_name = first_case.get("issue", "unknown")

    recorder = _RecordingJudge(OpenAIJudge(model=model))
    detector = LLMJudgeDetector(recorder, every=1, window=10_000)

    try:
        failures, summary = await analyze_records_async(
            snap_raw, act_raw, evt_raw, topology,
            extra_detectors=[detector],
        )
    except Exception as e:
        return {"path": path.name, "error": f"{type(e).__name__}: {e}"}

    det = [f for f in failures if not f.failure_type.startswith(JUDGE_PREFIX)]
    llm = [f for f in failures if f.failure_type.startswith(JUDGE_PREFIX)]
    return {
        "path": path.name,
        "example": example_name,
        "n_actions": summary["n_actions"],
        "n_snapshots": summary["n_snapshots"],
        "n_failures_det": len(det),
        "n_failures_llm": len(llm),
        "failures_by_type": {
            "deterministic": dict(Counter(f.failure_type for f in det)),
            "llm": dict(Counter(f.failure_type for f in llm)),
        },
        "llm_failures": [
            {
                "failure_type": f.failure_type,
                "summary": f.summary,
                "evidence_action_ids": f.evidence_action_ids,
                "agents_involved": f.agents_involved,
                "timestep": f.timestep,
            }
            for f in llm
        ],
        "judge_calls": recorder.calls,
    }


async def _run(
    paths: list[Path],
    topology: str,
    model: str,
    concurrency: int,
) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    done = 0
    total = len(paths)

    async def _bounded(p: Path) -> dict:
        nonlocal done
        async with sem:
            r = await _judge_one_trace(p, topology, model)
        done += 1
        if done % 25 == 0 or done == total:
            print(f"  {done}/{total} traces judged", file=sys.stderr)
        return r

    return await asyncio.gather(*(_bounded(p) for p in paths))


def _tabulate(results: list[dict]) -> dict:
    """Roll per-trace results into the summary that gets printed + saved."""
    traces_with_llm = sum(1 for r in results if r.get("n_failures_llm", 0) > 0)
    traces_with_det = sum(1 for r in results if r.get("n_failures_det", 0) > 0)
    errored = sum(1 for r in results if "error" in r)

    llm_by_family: Counter = Counter()
    det_by_type: Counter = Counter()
    for r in results:
        for ftype, n in r.get("failures_by_type", {}).get("llm", {}).items():
            llm_by_family[ftype] += n
        for ftype, n in r.get("failures_by_type", {}).get("deterministic", {}).items():
            det_by_type[ftype] += n

    by_example_llm: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        if r.get("n_failures_llm", 0) > 0:
            ex = r.get("example", "unknown")
            for ftype, n in r.get("failures_by_type", {}).get("llm", {}).items():
                by_example_llm[ex][ftype] += n

    return {
        "n_traces": len(results),
        "n_errored": errored,
        "traces_with_llm_failure": traces_with_llm,
        "traces_with_det_failure": traces_with_det,
        "llm_failures_by_family": dict(llm_by_family),
        "det_failures_by_type": dict(det_by_type),
        "llm_failures_by_example": {k: dict(v) for k, v in by_example_llm.items()},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=int, default=5, help="run on the first N traces (smoke test). Use --full for all.")
    p.add_argument("--full", action="store_true", help="run on all available traces (overrides --sample).")
    p.add_argument("--topology", default="support", choices=["support", "code_review", "ops"])
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--output-name", default=None, help="subdir name under results/maestro_judge/ (default: timestamp)")
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set; expected to be loaded from .env or environment.")

    if not TRACES_DIR.exists():
        sys.exit(f"missing {TRACES_DIR}; run maestro_to_drift.py first")

    all_traces = sorted(TRACES_DIR.glob("*.jsonl"))
    if not all_traces:
        sys.exit(f"no traces in {TRACES_DIR}")

    paths = all_traces if args.full else all_traces[: args.sample]
    label = args.output_name or time.strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"running drift's LLM judge over {len(paths)} / {len(all_traces)} MAESTRO traces", file=sys.stderr)
    print(f"  topology         : {args.topology}", file=sys.stderr)
    print(f"  judge model      : {args.judge_model}", file=sys.stderr)
    print(f"  concurrency      : {args.concurrency}", file=sys.stderr)
    print(f"  output dir       : {out_dir.relative_to(REPO_ROOT)}", file=sys.stderr)
    print(file=sys.stderr)

    t0 = time.perf_counter()
    results = asyncio.run(_run(paths, args.topology, args.judge_model, args.concurrency))
    elapsed = time.perf_counter() - t0

    # Persist per-trace details (the raw judge responses are the most
    # interesting thing to inspect later — saves having to re-spend tokens).
    for r in results:
        if "error" in r:
            continue
        (out_dir / f"{Path(r['path']).stem}.json").write_text(
            json.dumps(r, indent=2, default=str), encoding="utf-8"
        )

    summary = _tabulate(results)
    summary["elapsed_seconds"] = round(elapsed, 1)
    summary["topology"] = args.topology
    summary["judge_model"] = args.judge_model
    summary["n_traces_total_in_dir"] = len(all_traces)
    summary["n_traces_judged"] = len(paths)
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Stdout: human-readable summary
    print()
    print(f"=== drift LLM-judge x MAESTRO empirical results ===")
    print(f"  traces judged       : {summary['n_traces_judged']:,} / {summary['n_traces_total_in_dir']:,}")
    print(f"  traces errored      : {summary['n_errored']}")
    print(f"  elapsed             : {summary['elapsed_seconds']}s")
    print(f"  topology (det side) : {summary['topology']}")
    print(f"  judge model         : {summary['judge_model']}")
    print()
    print(f"  traces with >=1 deterministic failure : {summary['traces_with_det_failure']:,}")
    print(f"  traces with >=1 llm-judged failure    : {summary['traces_with_llm_failure']:,}")
    print()
    if summary["llm_failures_by_family"]:
        print(f"  LLM-judged failures by family:")
        for fam, n in sorted(summary["llm_failures_by_family"].items(), key=lambda kv: -kv[1]):
            print(f"    {n:6d}  {fam}")
    else:
        print(f"  no LLM-judged failures fired on any trace.")
    print()
    if summary["det_failures_by_type"]:
        print(f"  deterministic failures (for comparison):")
        for ft, n in sorted(summary["det_failures_by_type"].items(), key=lambda kv: -kv[1]):
            print(f"    {n:6d}  {ft}")
    print()
    print(f"per-trace details + raw judge responses: {out_dir.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
