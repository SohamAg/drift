"""Run drift against a real langgraph-supervisor multi-agent system.

What this validates: drift's coordination-detector library + chaos perturbation
catch real-shaped failures on a real, popular OSS langgraph MAS — not just on
hand-labelled fixtures. Target is `langchain-ai/langgraph-supervisor-py` (1.6k
stars, official LangChain ref), running the canonical math + research example
from its README.

The demo: a supervisor delegates to two specialist agents (math expert with
add/multiply tools, research expert with a mocked web_search tool). The
supervisor decides who handles each request and synthesizes the final answer.

What drift does to it:
  1. Chaos perturbs the initial `messages` state (empty list, malformed
     content, swapped roles, etc.) to surface schema-fragility bugs.
  2. The coordination library scans each per-perturbation trace for
     `verifier_always_approves`, `infinite_handoff`, `subagent_fanout_excess`.
  3. Tiered divergence cascade compares baseline vs perturbed final states.

Honest framing of what this proves:
  - If drift fires on coordination patterns in the supervisor demo, those are
    REAL findings on a REAL system — not synthetic test data.
  - If drift fires on nothing, the supervisor demo is well-behaved (which it
    might be — it's a 3-node toy example). That's information too.
  - This is NOT a benchmark. We're documenting "what does drift find on a
    representative langgraph MAS." Comparing-to-otherwise: without drift, a
    user running this demo sees the final answer and nothing else; with drift,
    they see what happens under input chaos and what coordination patterns
    fire across the trace.

Usage:
    pip install drift[validation]   # or just: pip install langgraph-supervisor langchain-openai
    PYTHONPATH=src python examples/adapters/run_drift_on_langgraph_supervisor.py
    PYTHONPATH=src python examples/adapters/run_drift_on_langgraph_supervisor.py --intensity aggressive --judge openai

Cost (default config): ~5-15 OpenAI calls per perturbation × ~10 perturbations
= ~$0.05-0.15 total at gpt-4o-mini rates.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

# Silence the deprecation noise from langgraph-prebuilt's create_react_agent —
# the library author hasn't migrated yet, not our problem.
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*langgraph.*")

from drift.adapters.langgraph import drift_test  # noqa: E402
from drift.failures.judge import build_judge  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "langgraph_supervisor"


# ---------------------------------------------------------------------------
# Build the target MAS — canonical math + research supervisor from the README.
# ---------------------------------------------------------------------------


def _add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def _multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def _web_search(query: str) -> str:
    """Mocked web search — returns a fixed string so the demo is deterministic
    on the tool layer; the only randomness comes from the LLM itself."""
    return f"(mock web result for query: {query!r})"


def _build_supervisor_mas(model_name: str = "gpt-4o-mini"):
    """Construct the canonical langgraph-supervisor README example.

    Returns a compiled StateGraph (has .invoke / .stream / .astream).
    """
    # Imports here so the script doesn't crash if langgraph-supervisor isn't
    # installed — the friendly error happens in main() instead.
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from langgraph_supervisor import create_supervisor

    model = ChatOpenAI(model=model_name)

    math_agent = create_react_agent(
        model=model,
        tools=[_add, _multiply],
        prompt="You are a math expert. Use the add and multiply tools to compute results.",
        name="math_expert",
    )
    research_agent = create_react_agent(
        model=model,
        tools=[_web_search],
        prompt="You are a web researcher. Use the web_search tool to find facts.",
        name="research_expert",
    )

    workflow = create_supervisor(
        [math_agent, research_agent],
        model=model,
        prompt=(
            "You are a team supervisor managing a math expert and a research expert. "
            "Delegate tasks to them and provide the final answer."
        ),
    )
    return workflow.compile()


def _initial_state(question: str) -> dict:
    """Initial state for the demo: a single-message conversation. Drift will
    perturb this dict — flipping bools (none here), reversing lists, clearing
    keys, corrupting strings, etc."""
    return {"messages": [{"role": "user", "content": question}]}


# ---------------------------------------------------------------------------
# Report helpers — compress drift's findings into something readable.
# ---------------------------------------------------------------------------


def _summarize_perturbation(p) -> dict:
    """Compress one PerturbationResult into a JSON-serializable summary row."""
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
        "error": p.error[:120] if p.crashed else "",
        "divergence": p.divergence_summary[:160] if p.diverged else "",
        "trace_steps": len(p.trace),
        "judge_findings": [
            {"type": f["failure_type"], "summary": f["summary"][:160]}
            for f in p.judge_findings
        ],
        "coordination_findings": [
            {"type": f["failure_type"], "summary": f["summary"][:160], "agents": f.get("agents_involved", [])}
            for f in p.coordination_findings
        ],
    }


def _print_report(result, question: str) -> None:
    print()
    print("=" * 74)
    print("drift × langgraph-supervisor empirical run")
    print("=" * 74)
    print(f"question                : {question!r}")
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

    # Baseline.
    print("--- BASELINE (unperturbed) -------------------------------------------")
    if result.baseline.crashed:
        print(f"  BASELINE CRASHED: {result.baseline.error_type}: {result.baseline.error}")
    else:
        print(f"  baseline ran OK; final state had {len(result.baseline.final_state or {})} keys, "
              f"trace length {len(result.baseline.trace)} super-steps")
    if result.baseline.judge_findings:
        print(f"  judge findings on baseline:")
        for f in result.baseline.judge_findings:
            print(f"    [{f['failure_type']}] {f['summary'][:140]}")
    if result.baseline.coordination_findings:
        print(f"  coordination findings on baseline:")
        for f in result.baseline.coordination_findings:
            print(f"    [{f['failure_type']}] {f['summary'][:140]}")
    print()

    # Per-perturbation, sorted by severity.
    def _sort_key(p):
        return (0 if p.crashed else 1 if p.diverged else 2, p.event_name)

    print("--- PER-PERTURBATION (sorted by severity) ----------------------------")
    for p in sorted(result.perturbations, key=_sort_key):
        summ = _summarize_perturbation(p)
        tag_str = " ".join(summ["tags"])
        print(f"  [{tag_str}] {p.event_name}")
        if summ["error"]:
            print(f"    error      : {summ['error']}")
        if summ["divergence"]:
            print(f"    divergence : {summ['divergence']}")
        for f in summ["judge_findings"]:
            print(f"    JUDGE      : [{f['type']}] {f['summary']}")
        for f in summ["coordination_findings"]:
            print(f"    COORD      : [{f['type']}] {f['summary']}")
    print()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--question", default="What is 7 times 8?",
                   help="initial user message. Try math / research / mixed prompts to see different agent topologies.")
    p.add_argument("--intensity", default="moderate",
                   choices=["off", "light", "moderate", "aggressive"])
    p.add_argument("--max-perturbations", type=int, default=8,
                   help="cap (each perturbation = one full graph invocation = 5-15 OpenAI calls).")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--model", default="gpt-4o-mini",
                   help="LLM to use for the MAS itself. drift's judge is separate.")
    p.add_argument("--judge", default="off", choices=["off", "mock", "openai"],
                   help="enable drift's LLM coordination judge over per-perturbation traces.")
    p.add_argument("--divergence-mode", default="exact",
                   choices=["exact", "tiered", "off"])
    p.add_argument("--baseline-rollouts", type=int, default=1,
                   help="N baseline runs to measure LLM noise floor (tiered mode only). "
                        "Each rollout = one full graph invocation. Recommend 3-5 for LLM graphs.")
    p.add_argument("--max-judge-calls", type=int, default=10,
                   help="hard ceiling on tier-3 LLM divergence-equivalence calls.")
    p.add_argument("--save-json", action="store_true",
                   help="write the full result + per-perturbation detail to results/langgraph_supervisor/")
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set; export it or put it in .env")

    print(f"building langgraph-supervisor demo with model={args.model}...", file=sys.stderr)
    try:
        app = _build_supervisor_mas(model_name=args.model)
    except ImportError as e:
        sys.exit(
            f"could not build langgraph-supervisor demo: {e}\n"
            "Install with: pip install drift[validation]\n"
            "or: pip install langgraph-supervisor langchain-openai"
        )

    judge_llm = None
    if args.judge != "off":
        judge_llm = build_judge(args.judge)

    init = _initial_state(args.question)
    print(f"running drift_test against the supervisor MAS...", file=sys.stderr)
    print(f"  question     : {args.question!r}", file=sys.stderr)
    print(f"  intensity    : {args.intensity}", file=sys.stderr)
    print(f"  max perturb. : {args.max_perturbations}", file=sys.stderr)
    print(f"  judge        : {args.judge}", file=sys.stderr)
    print(f"  divergence   : {args.divergence_mode}", file=sys.stderr)
    print(file=sys.stderr)

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

    _print_report(result, args.question)

    if args.save_json:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = {
            "question": args.question,
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
            "perturbations": [_summarize_perturbation(p) for p in result.perturbations],
        }
        fname = f"run_{args.intensity}_{args.seed}.json"
        path = RESULTS_DIR / fname
        path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"\nresult written to {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
