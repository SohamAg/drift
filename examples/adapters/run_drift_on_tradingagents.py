"""Run drift against TradingAgents (TauricResearch) — the adversarial-debate MAS.

What this validates: whether the supervisor-repetition-style coordination bug
(see run_drift_on_langgraph_supervisor.py) is a one-off quirk of that one
demo, or a pattern that recurs on a structurally different, much larger
third-party MAS. TradingAgents is the sharpest test available for
`coordination_contradiction`: Bull and Bear researchers are DESIGNED to
disagree, so the interesting failure isn't "they disagreed" (expected) — it's
a contradiction that survives past the Research Manager / Risk Judge into
the final recommendation, or a debate participant that ignores the other
side's evidence outright.

Uses the registry target `tradingagents_debate` (examples/targets/) so build
logic stays in one place.

Usage:
    PYTHONPATH=src python examples/adapters/run_drift_on_tradingagents.py
    PYTHONPATH=src python examples/adapters/run_drift_on_tradingagents.py --judge openai --max-perturbations 4

Cost warning: this graph is ~21 nodes (4 analysts w/ tool round-trips, 2-round
bull/bear debate, research manager, trader, 3-way risk debate, risk judge) —
a SINGLE invocation is ~20-30 LLM calls even at max_debate_rounds=1. Start
with --max-perturbations 1 --judge off to smoke-test before spending on a
full sweep.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from drift.adapters.langgraph import drift_test  # noqa: E402
from drift.failures.judge import build_judge  # noqa: E402
from targets import tradingagents_debate as target  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "tradingagents_debate"


def _summarize_perturbation(p) -> dict:
    tags = []
    if p.crashed:
        tags.append(f"CRASH[{p.error_type}]")
    if p.diverged:
        tags.append("DIVERGE")
    if not p.crashed and not p.diverged:
        tags.append("UNCHANGED")
    if p.judge_findings:
        tags.append(f"JUDGE×{len(p.judge_findings)}")
    if p.coordination_findings:
        tags.append(f"COORD×{len(p.coordination_findings)}")
    return {
        "event": p.event_name,
        "field": p.perturbed_field,
        "pattern": p.pattern_type,
        "tags": tags,
        "error": p.error[:200] if p.crashed else "",
        "divergence": p.divergence_summary[:200] if p.diverged else "",
        "trace_steps": len(p.trace),
        "judge_findings": [
            {"type": f["failure_type"], "summary": f["summary"][:200]}
            for f in p.judge_findings
        ],
        "coordination_findings": [
            {"type": f["failure_type"], "summary": f["summary"][:200], "agents": f.get("agents_involved", [])}
            for f in p.coordination_findings
        ],
    }


def _print_report(result, ticker: str, elapsed: float) -> None:
    print()
    print("=" * 78)
    print("drift × TradingAgents (TauricResearch) empirical run")
    print("=" * 78)
    print(f"ticker                  : {ticker}")
    print(f"wall clock              : {elapsed:.1f}s")
    print(f"intensity               : {result.intensity}")
    print(f"divergence mode         : {result.divergence_mode}")
    print(f"perturbations attempted : {len(result.perturbations)}")
    print(f"  crashed               : {result.n_crashed}")
    print(f"  diverged              : {result.n_diverged}")
    print(f"  unchanged             : {result.n_unchanged}")
    print(f"judge findings (total)  : {result.n_judge_findings}")
    print(f"coord findings (total)  : {result.n_coordination_findings}")
    if result.judge_calls_used:
        print(f"judge calls (tier 3)    : {result.judge_calls_used} / {result.judge_calls_budget}")
    print()

    print("--- BASELINE (unperturbed) --------------------------------------------")
    if result.baseline.crashed:
        print(f"  BASELINE CRASHED: {result.baseline.error_type}: {result.baseline.error[:300]}")
    else:
        print(f"  baseline ran OK; trace length {len(result.baseline.trace)} super-steps")
    for f in result.baseline.judge_findings:
        print(f"  JUDGE : [{f['failure_type']}] {f['summary'][:200]}")
    for f in result.baseline.coordination_findings:
        print(f"  COORD : [{f['failure_type']}] {f['summary'][:200]}")
    print()

    def _sort_key(p):
        return (0 if p.crashed else 1 if p.diverged else 2, p.event_name)

    print("--- PER-PERTURBATION (sorted by severity) ------------------------------")
    for p in sorted(result.perturbations, key=_sort_key):
        summ = _summarize_perturbation(p)
        print(f"  [{' '.join(summ['tags'])}] {p.event_name}")
        if summ["error"]:
            print(f"    error      : {summ['error']}")
        if summ["divergence"]:
            print(f"    divergence : {summ['divergence']}")
        for f in summ["judge_findings"]:
            print(f"    JUDGE      : [{f['type']}] {f['summary']}")
        for f in summ["coordination_findings"]:
            print(f"    COORD      : [{f['type']}] {f['summary']}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticker", default=target.DEFAULT_QUERY)
    p.add_argument("--intensity", default="light",
                   choices=["off", "light", "moderate", "aggressive"])
    p.add_argument("--max-perturbations", type=int, default=1,
                   help="each perturbation = one full 21-node graph invocation, ~20-30 LLM calls.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--model", default=target.DEFAULT_MODEL,
                   help="must be reasoning-capable — tradingagents' build_chat_model "
                        "always sends reasoning_effort for the openai provider.")
    p.add_argument("--judge", default="off", choices=["off", "mock", "openai"])
    p.add_argument("--divergence-mode", default="exact", choices=["exact", "tiered", "off"])
    p.add_argument("--baseline-rollouts", type=int, default=1)
    p.add_argument("--max-judge-calls", type=int, default=10)
    p.add_argument("--save-json", action="store_true")
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set; export it or put it in .env")

    print(f"building tradingagents_debate (model={args.model})...", file=sys.stderr)
    app = target.build(model=args.model)
    init = target.initial_state(args.ticker)

    judge_llm = build_judge(args.judge) if args.judge != "off" else None

    print(f"running drift_test against TradingAgents ({args.ticker})...", file=sys.stderr)
    print(f"  intensity    : {args.intensity}", file=sys.stderr)
    print(f"  max perturb. : {args.max_perturbations}", file=sys.stderr)
    print(f"  judge        : {args.judge}", file=sys.stderr)
    print(f"  divergence   : {args.divergence_mode}", file=sys.stderr)
    print(file=sys.stderr)

    t0 = time.time()
    result = drift_test(
        graph=app,
        initial_state=init,
        intensity=args.intensity,
        max_perturbations=args.max_perturbations,
        seed=args.seed,
        judge_llm=judge_llm,
        divergence_mode=args.divergence_mode,
        baseline_rollouts=args.baseline_rollouts,
        max_judge_calls=args.max_judge_calls,
    )
    elapsed = time.time() - t0

    _print_report(result, args.ticker, elapsed)

    if args.save_json:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = {
            "ticker": args.ticker,
            "elapsed_seconds": elapsed,
            "intensity": result.intensity,
            "divergence_mode": result.divergence_mode,
            "n_crashed": result.n_crashed,
            "n_diverged": result.n_diverged,
            "n_unchanged": result.n_unchanged,
            "n_judge_findings": result.n_judge_findings,
            "n_coordination_findings": result.n_coordination_findings,
            "judge_calls_used": result.judge_calls_used,
            "baseline": {
                "crashed": result.baseline.crashed,
                "error": result.baseline.error,
                "trace_steps": len(result.baseline.trace),
                "judge_findings": result.baseline.judge_findings,
                "coordination_findings": result.baseline.coordination_findings,
            },
            "perturbations": [_summarize_perturbation(pr) for pr in result.perturbations],
        }
        fname = f"run_{args.intensity}_{args.seed}.json"
        path = RESULTS_DIR / fname
        path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"\nresult written to {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
