"""Extended-MAS drift run — 5-specialist supervisor with richer state.

Where the canonical 2-agent demo (`run_drift_on_langgraph_supervisor.py`)
produces 3-step traces below the coordination detectors' thresholds, this
one builds a 5-specialist supervisor: math + research + code_writer +
data_analyst + summarizer. Forces longer traces, more handoffs, and
crosses into the regime where `infinite_handoff` (≥4 alternations) and
`subagent_fanout_excess` (≥8 distinct subagents OR fanout/output ratio >3)
have a chance to actually fire on real LLM behavior.

The questions are deliberately compound — designed to require multiple
specialists working together, with judgment calls about who does what.

Usage:
    PYTHONPATH=src python examples/adapters/run_drift_on_supervisor_extended.py
    PYTHONPATH=src python examples/adapters/run_drift_on_supervisor_extended.py --save-json
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_drift_on_langgraph_supervisor import _summarize_perturbation  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "langgraph_supervisor_extended"


# ---------------------------------------------------------------------------
# Five-specialist supervisor.
# ---------------------------------------------------------------------------


def _add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def _multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def _divide(a: float, b: float) -> float:
    """Divide a by b."""
    return a / b if b else float("nan")


def _web_search(query: str) -> str:
    """Mocked web search. Deterministic on the tool layer."""
    return f"(mock web result for query: {query!r})"


def _read_file(filename: str) -> str:
    """Mocked file-reader tool."""
    return f"(mock file content for: {filename})"


def _write_code(language: str, intent: str) -> str:
    """Mocked code-writer that returns a 1-line solution stub."""
    return f"# {language} stub for: {intent}\nresult = None"


def _analyze_data(table_csv: str, instruction: str) -> str:
    """Mocked data-analyst tool — returns a one-line analysis."""
    return f"(mock analysis of {len(table_csv)} chars: {instruction})"


def _summarize(text: str) -> str:
    """Mocked summarizer tool."""
    return f"(summary of {len(text)} chars)"


def _build_extended_mas(model_name: str = "gpt-4o-mini"):
    """Five specialists + one supervisor. Returns a compiled StateGraph."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from langgraph_supervisor import create_supervisor

    model = ChatOpenAI(model=model_name)

    math_agent = create_react_agent(
        model=model, tools=[_add, _multiply, _divide],
        prompt="You are a math expert. Use add/multiply/divide tools.",
        name="math_expert",
    )
    research_agent = create_react_agent(
        model=model, tools=[_web_search, _read_file],
        prompt="You are a researcher. Use web_search and read_file tools.",
        name="research_expert",
    )
    code_agent = create_react_agent(
        model=model, tools=[_write_code],
        prompt="You are a code writer. Use the write_code tool.",
        name="code_writer",
    )
    data_agent = create_react_agent(
        model=model, tools=[_analyze_data],
        prompt="You are a data analyst. Use the analyze_data tool.",
        name="data_analyst",
    )
    summary_agent = create_react_agent(
        model=model, tools=[_summarize],
        prompt="You are a summarizer. Use the summarize tool.",
        name="summarizer",
    )

    workflow = create_supervisor(
        [math_agent, research_agent, code_agent, data_agent, summary_agent],
        model=model,
        prompt=(
            "You manage five specialists: math_expert, research_expert, code_writer, "
            "data_analyst, summarizer. Decompose the user's task, delegate each part "
            "to the appropriate specialist, and synthesize the final answer. "
            "Delegate to multiple specialists if the task requires it. "
            "Avoid asking the same question twice."
        ),
    )
    return workflow.compile()


# Questions designed to force multi-specialist coordination + longer traces.
QUESTIONS = [
    # ---- multi-step compound questions ----
    {
        "q": "Search the web for the speed of light in m/s. Divide it by 1000. "
             "Then write a Python function that returns this value.",
        "category": "multi_3step",
    },
    {
        "q": "Find the population of Tokyo via web search, multiply it by 0.5, "
             "then write me a one-line summary of the result.",
        "category": "multi_3step_summary",
    },
    {
        "q": "Read the file 'data.csv', analyze it for trends, then summarize "
             "the analysis in one paragraph.",
        "category": "data_pipeline",
    },
    # ---- forced fanout ----
    {
        "q": "I need four separate things done in parallel: "
             "(1) search for the boiling point of water, "
             "(2) compute 7 times 8, "
             "(3) write a Python function to reverse a string, "
             "(4) analyze the data 'a,b,c\\n1,2,3'. "
             "Use all four specialists.",
        "category": "forced_fanout_4",
    },
    {
        "q": "Do all five of these in parallel: search for the GDP of Germany, "
             "compute 100 divided by 3, write a Python class for a circle, "
             "analyze 'x,y\\n1,2\\n3,4', and summarize 'hello world'. "
             "I need every specialist to contribute.",
        "category": "forced_fanout_5",
    },
    # ---- ambiguous-which-specialist ----
    {
        "q": "Tell me about the formula E=mc^2",
        "category": "ambiguous_specialist_choice",
    },
    # ---- iterative / could loop ----
    {
        "q": "Research Mars's atmosphere, then use the data to compute its "
             "average pressure in atmospheres, then research how that compares "
             "to Earth, then write code that prints both. Re-verify each step.",
        "category": "iterative_re_verify",
    },
    # ---- contradictory instruction (might cause looping) ----
    {
        "q": "Have the math_expert do nothing but research, and have the "
             "research_expert do nothing but math. Then return the result of 2+2.",
        "category": "contradictory_role",
    },
]


def _findings_summary(result) -> dict:
    return {
        "n_perturbations": len(result.perturbations),
        "n_crashed": result.n_crashed,
        "n_diverged": result.n_diverged,
        "n_unchanged": result.n_unchanged,
        "n_judge_findings": result.n_judge_findings,
        "n_coord_findings": result.n_coordination_findings,
        "judge_calls_used": result.judge_calls_used,
        "baseline_crashed": result.baseline.crashed,
        "baseline_trace_steps": len(result.baseline.trace),
        "baseline_judge": [
            {"type": f["failure_type"], "summary": f["summary"][:160]}
            for f in result.baseline.judge_findings
        ],
        "baseline_coord": [
            {"type": f["failure_type"], "summary": f["summary"][:160],
             "agents": f.get("agents_involved", [])}
            for f in result.baseline.coordination_findings
        ],
        "baseline_unique_agents": sorted({s["node"] for s in result.baseline.trace}),
        "perturbations": [_summarize_perturbation(p) for p in result.perturbations],
    }


def _print_report(per_q: list[dict]) -> None:
    print()
    print("=" * 76)
    print("drift × langgraph-supervisor (5-specialist) — EXTENDED MAS RUN")
    print("=" * 76)

    totals = {
        "perturbations": sum(q["n_perturbations"] for q in per_q),
        "crashed": sum(q["n_crashed"] for q in per_q),
        "diverged": sum(q["n_diverged"] for q in per_q),
        "unchanged": sum(q["n_unchanged"] for q in per_q),
        "judge": sum(q["n_judge_findings"] for q in per_q),
        "coord": sum(q["n_coord_findings"] for q in per_q),
        "judge_calls": sum(q["judge_calls_used"] for q in per_q),
    }
    for k, v in totals.items():
        print(f"  total {k:<14s}: {v}")
    print()

    print("--- BASELINE-LEVEL FINDINGS -------------------------------------------")
    for q in per_q:
        if q["baseline_judge"] or q["baseline_coord"]:
            print(f"  [{q['category']}] {q['question'][:60]!r}")
            print(f"    trace_steps={q['baseline_trace_steps']} agents={q['baseline_unique_agents']}")
            for f in q["baseline_judge"]:
                print(f"    JUDGE: [{f['type']}] {f['summary']}")
            for f in q["baseline_coord"]:
                print(f"    COORD: [{f['type']}] {f['summary']}  agents={f.get('agents', [])}")
    print()

    print("--- PER-QUESTION DETAIL -----------------------------------------------")
    for q in per_q:
        agents = q["baseline_unique_agents"]
        print(f"  [{q['category']:<28s}] {q['question'][:60]!r}")
        print(f"    baseline: {q['baseline_trace_steps']} steps, agents={agents}")
        print(f"    perturb : pert={q['n_perturbations']} crashed={q['n_crashed']} "
              f"diverged={q['n_diverged']} unchanged={q['n_unchanged']} "
              f"judge={q['n_judge_findings']} coord={q['n_coord_findings']}")
        for p in q["perturbations"]:
            tags = " ".join(p["tags"])
            if "COORD" in tags or "JUDGE" in tags or "CRASH" in tags:
                print(f"      [{tags}] {p['event']}")
                for f in p.get("coordination_findings", []):
                    print(f"        COORD: [{f['type']}] {f['summary']}")
                for f in p.get("judge_findings", []):
                    print(f"        JUDGE: [{f['type']}] {f['summary']}")
    print()


async def _run_one(question: dict, *, model: str, max_perturbations: int,
                   intensity: str, baseline_rollouts: int, use_judge: bool) -> dict:
    app = _build_extended_mas(model_name=model)
    init = {"messages": [{"role": "user", "content": question["q"]}]}
    judge_llm = build_judge("openai") if use_judge else None

    result = await drift_test_async(
        graph=app, initial_state=init,
        intensity=intensity, max_perturbations=max_perturbations,
        seed=7, judge_llm=judge_llm,
        divergence_mode="tiered",
        baseline_rollouts=baseline_rollouts,
        max_judge_calls=15,
    )
    summary = _findings_summary(result)
    summary["question"] = question["q"]
    summary["category"] = question["category"]
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true",
                   help="3 representative questions only (faster, cheaper)")
    p.add_argument("--max-perturbations", type=int, default=3,
                   help="cap per question (kept low because 5-agent traces are longer)")
    p.add_argument("--intensity", default="aggressive")
    p.add_argument("--baseline-rollouts", type=int, default=2,
                   help="kept lower than question sweep because each rollout is more expensive on the 5-agent MAS")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--save-json", action="store_true")
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")

    qs = QUESTIONS
    if args.quick:
        qs = [QUESTIONS[0], QUESTIONS[3], QUESTIONS[6]]  # 3-step, fanout-4, iterative

    print(f"running drift × 5-specialist supervisor on {len(qs)} compound question(s)", file=sys.stderr)
    print(f"  model         : {args.model}", file=sys.stderr)
    print(f"  intensity     : {args.intensity}", file=sys.stderr)
    print(f"  perturb cap   : {args.max_perturbations}", file=sys.stderr)
    print(f"  rollouts      : {args.baseline_rollouts}", file=sys.stderr)
    print(f"  judge         : {'off' if args.no_judge else 'openai'}", file=sys.stderr)
    print(file=sys.stderr)

    async def _run_all():
        out = []
        for i, q in enumerate(qs, start=1):
            print(f"  [{i}/{len(qs)}] {q['category']:<28s} {q['q'][:80]!r}", file=sys.stderr)
            t0 = time.perf_counter()
            try:
                row = await _run_one(
                    q, model=args.model,
                    max_perturbations=args.max_perturbations,
                    intensity=args.intensity,
                    baseline_rollouts=args.baseline_rollouts,
                    use_judge=not args.no_judge,
                )
                out.append(row)
                print(f"        done in {time.perf_counter()-t0:.1f}s "
                      f"(baseline_steps={row['baseline_trace_steps']}, "
                      f"unique_agents={len(row['baseline_unique_agents'])})", file=sys.stderr)
            except Exception as e:
                print(f"        FAILED: {type(e).__name__}: {e}", file=sys.stderr)
                out.append({"question": q["q"], "category": q["category"],
                            "error": f"{type(e).__name__}: {e}",
                            "n_perturbations": 0, "n_crashed": 0, "n_diverged": 0,
                            "n_unchanged": 0, "n_judge_findings": 0, "n_coord_findings": 0,
                            "judge_calls_used": 0, "baseline_crashed": False,
                            "baseline_trace_steps": 0, "baseline_judge": [],
                            "baseline_coord": [], "baseline_unique_agents": [],
                            "perturbations": []})
        return out

    t0 = time.perf_counter()
    per_q = asyncio.run(_run_all())
    elapsed = time.perf_counter() - t0

    _print_report(per_q)

    if args.save_json:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        prefix = time.strftime("%Y%m%d_%H%M%S")
        path = RESULTS_DIR / f"extended_{prefix}.json"
        path.write_text(json.dumps({
            "config": {
                "model": args.model, "intensity": args.intensity,
                "max_perturbations": args.max_perturbations,
                "baseline_rollouts": args.baseline_rollouts,
                "use_judge": not args.no_judge,
            },
            "elapsed_seconds": round(elapsed, 1),
            "per_question": per_q,
        }, indent=2, default=str), encoding="utf-8")
        print(f"\nresults written to {path.relative_to(REPO_ROOT)}")
    print(f"total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
