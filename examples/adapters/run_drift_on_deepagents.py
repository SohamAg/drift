"""Target 3 validation — run drift against `langchain-ai/deepagents`.

Real 3rd-party OSS multi-agent framework (25.7k stars, official LangChain).
Uses LangGraph under the hood, dispatches to sub-agents, filesystem-mediated
state. Exactly the surface area drift's detectors target.

Goal: run drift's baseline + a small chaos perturbation sweep against a
deepagent, see whether ANY of the 6 structured coord detectors or the LLM
judge fire on an unfamiliar 3rd-party MAS. This is the strongest empirical
validation drift has had — no cherry-picking, no adversarial construction,
no known-buggy target. Whatever surfaces surfaces.

Cost budget: 1 query, 2 perturbations, judge on. Estimate ~$0.30-0.80 at
gpt-4o-mini rates depending on how much sub-agent recursion the deepagent
does.

Run:
    PYTHONPATH=src python examples/adapters/run_drift_on_deepagents.py
    PYTHONPATH=src python examples/adapters/run_drift_on_deepagents.py --save-json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*langgraph.*")

from drift.adapters.langgraph import drift_test_async  # noqa: E402
from drift.failures.judge import build_judge  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "deepagents"


def _build_deepagent(model: str = "gpt-4o-mini"):
    """Build a minimal deepagent — no external tools, just its built-in
    planning + sub-agent + filesystem tools. Simplest legitimate config.
    """
    from deepagents import create_deep_agent

    return create_deep_agent(
        model=f"openai:{model}",
        tools=[],   # rely on built-in fs / planning / subagent tools
        system_prompt=(
            "You are a research assistant. Break the task down into steps "
            "and delegate subtasks to sub-agents when appropriate. Keep "
            "your final answer under 5 sentences."
        ),
    )


def _initial_state(question: str) -> dict:
    """LangGraph MessagesState shape — deepagents wraps its state internally
    but accepts this input on invoke."""
    return {"messages": [{"role": "user", "content": question}]}


def _summary_row(question: str, result) -> dict:
    return {
        "question": question,
        "baseline_crashed": result.baseline.crashed,
        "baseline_error": result.baseline.error[:200] if result.baseline.crashed else "",
        "baseline_trace_steps": len(result.baseline.trace),
        "baseline_unique_agents": sorted({s.get("node", "") for s in result.baseline.trace}),
        "baseline_judge": [
            {"type": f["failure_type"], "summary": f["summary"][:200]}
            for f in result.baseline.judge_findings
        ],
        "baseline_coord": [
            {"type": f["failure_type"], "summary": f["summary"][:200],
             "agents": f.get("agents_involved", [])}
            for f in result.baseline.coordination_findings
        ],
        "n_perturbations": len(result.perturbations),
        "n_crashed": result.n_crashed,
        "n_diverged": result.n_diverged,
        "n_unchanged": result.n_unchanged,
        "n_judge": result.n_judge_findings,
        "n_coord": result.n_coordination_findings,
        "perturbations": [
            {"event": p.event_name, "crashed": p.crashed, "diverged": p.diverged,
             "judge_findings": [
                 {"type": f["failure_type"], "summary": f["summary"][:160]}
                 for f in p.judge_findings
             ],
             "coord_findings": [
                 {"type": f["failure_type"], "summary": f["summary"][:160],
                  "agents": f.get("agents_involved", [])}
                 for f in p.coordination_findings
             ]}
            for p in result.perturbations
        ],
    }


def _print_report(row: dict) -> None:
    print()
    print("=" * 74)
    print("drift x deepagents — real 3rd-party MAS validation")
    print("=" * 74)
    print(f"question       : {row['question'][:120]!r}")
    print(f"baseline       : "
          f"crashed={row['baseline_crashed']}, trace_steps={row['baseline_trace_steps']}, "
          f"unique_agents={row['baseline_unique_agents']}")
    if row['baseline_error']:
        print(f"  error        : {row['baseline_error']}")
    print()
    print(f"baseline judge findings : {len(row['baseline_judge'])}")
    for f in row['baseline_judge']:
        print(f"    [{f['type']}] {f['summary']}")
    print(f"baseline coord findings : {len(row['baseline_coord'])}")
    for f in row['baseline_coord']:
        print(f"    [{f['type']}] agents={f['agents']}")
        print(f"      {f['summary']}")
    print()
    print(f"perturbations : {row['n_perturbations']} "
          f"(crashed={row['n_crashed']}, diverged={row['n_diverged']}, unchanged={row['n_unchanged']})")
    print(f"  total judge : {row['n_judge']}")
    print(f"  total coord : {row['n_coord']}")
    for p in row['perturbations']:
        tags = []
        if p['crashed']:
            tags.append('CRASH')
        if p['diverged']:
            tags.append('DIVERGE')
        if p['coord_findings']:
            tags.append(f"COORD*{len(p['coord_findings'])}")
        if p['judge_findings']:
            tags.append(f"JUDGE*{len(p['judge_findings'])}")
        print(f"    [{' '.join(tags) or 'no findings'}] {p['event']}")
        for f in p['coord_findings']:
            print(f"      COORD  [{f['type']}] agents={f['agents']} {f['summary']}")
        for f in p['judge_findings']:
            print(f"      JUDGE  [{f['type']}] {f['summary']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--question", default=(
        "Research the pros and cons of LangGraph vs CrewAI as multi-agent "
        "frameworks. Provide a short 3-paragraph summary."
    ))
    p.add_argument("--max-perturbations", type=int, default=2)
    p.add_argument("--intensity", default="moderate")
    p.add_argument("--baseline-rollouts", type=int, default=1)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--save-json", action="store_true")
    p.add_argument("--divergence-mode", default="tiered")
    p.add_argument("--max-judge-calls", type=int, default=6)
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")

    print(f"building deepagent (model={args.model})...", file=sys.stderr)
    try:
        agent = _build_deepagent(model=args.model)
    except ImportError as e:
        sys.exit(f"deepagents not installed: {e}")
    initial = _initial_state(args.question)

    print(f"running drift_test on deepagent (intensity={args.intensity}, "
          f"perturb_cap={args.max_perturbations})...", file=sys.stderr)

    judge_llm = None if args.no_judge else build_judge("openai")

    async def _run():
        return await drift_test_async(
            graph=agent,
            initial_state=initial,
            intensity=args.intensity,
            max_perturbations=args.max_perturbations,
            seed=7,
            judge_llm=judge_llm,
            divergence_mode=args.divergence_mode,
            baseline_rollouts=args.baseline_rollouts,
            max_judge_calls=args.max_judge_calls,
        )

    t0 = time.perf_counter()
    result = asyncio.run(_run())
    elapsed = time.perf_counter() - t0

    row = _summary_row(args.question, result)
    row["_elapsed_s"] = round(elapsed, 1)
    _print_report(row)

    if args.save_json:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"deepagents_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
        print(f"\nresults written to {path.relative_to(REPO_ROOT)}", file=sys.stderr)

    print(f"\ntotal elapsed: {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
