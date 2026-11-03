"""Live validation — message-aware chaos + detectors on a REAL LangGraph MAS.

Runs drift's exhaustive schema walk against the vendor's own
langgraph-supervisor example (examples/targets/supervisor_basic.py: a
supervisor routing between a math expert and a research expert, each a
create_react_agent subgraph) with a live gpt-4o-mini. The point is to see
the new machinery on real model behaviour rather than scripted stubs:

  - subgraph visibility: the trace should show `math_expert/agent`,
    `math_expert/tools`, ... not just the three top-level nodes;
  - message normalisation: two identical runs should NOT diverge on
    `messages` (per-run ids are stripped), so the noise band is meaningful;
  - the six message patterns: what does a real supervisor do when the
    user's instruction is dropped / truncated / garbled, when the last tool
    call failed, when a tool result carries an injected instruction, when
    history claims the task is already done;
  - the three message-aware detectors on top of all that.

Cost: one question, exhaustive on `{"messages": [...]}` = clear_list + 6
message patterns = 7 perturbations + 2 baseline rollouts ≈ 9 graph runs of
3–8 LLM calls each. A few cents on gpt-4o-mini. No judge by default.

Usage:
    PYTHONPATH=src python examples/adapters/run_drift_on_supervisor_message_chaos.py
    PYTHONPATH=src python examples/adapters/run_drift_on_supervisor_message_chaos.py --extended --save-json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*langgraph.*")

from drift.adapters.langgraph import AdapterResult, drift_test_async  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "supervisor_message_chaos"

QUESTIONS = {
    "basic": "What is 7 times 8? Then search the web for the capital of France.",
    "extended": (
        "Search the web for the speed of light in m/s. Divide it by 1000. "
        "Then write a Python function that returns this value."
    ),
}


def _build(extended: bool, model: str):
    if extended:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from run_drift_on_supervisor_extended import _build_extended_mas
        return _build_extended_mas(model_name=model), "supervisor_extended_5"
    from targets import supervisor_basic
    return supervisor_basic.build(model=model), supervisor_basic.NAME


def _result_to_json(result: AdapterResult) -> dict:
    """Full dump (traces included) — the server's shape, built from dataclasses."""
    d = asdict(result)
    d["n_crashed"] = result.n_crashed
    d["n_diverged"] = result.n_diverged
    d["n_unchanged"] = result.n_unchanged
    d["n_recursion_limit"] = result.n_recursion_limit
    d["n_coordination_findings"] = result.n_coordination_findings
    return d


def _print(result: AdapterResult, label: str) -> None:
    print()
    print("=" * 76)
    print(f"drift × {label} — message-aware chaos, live model")
    print("=" * 76)
    for line in result.summary_lines():
        print(line)
    b = result.baseline
    nodes = [s["node"] for s in b.trace]
    print(f"\nbaseline: {len(b.trace)} super-steps, {len(set(nodes))} distinct nodes")
    print("  nodes in order:", " -> ".join(nodes))
    sub = sorted({s["node"] for s in b.trace if s.get("namespace")})
    print(f"  subgraph steps visible: {len([s for s in b.trace if s.get('namespace')])}  ({', '.join(sub) or 'none'})")
    if b.final_state:
        msgs = b.final_state.get("messages", [])
        print(f"  final conversation: {len(msgs)} messages; last AI: "
              f"{next((m['content'] for m in reversed(msgs) if m.get('type') == 'ai' and m.get('content')), '')[:140]!r}")
    for f in b.coordination_findings:
        print(f"  BASELINE COORD [{f['failure_type']}] {f['summary'][:200]}")
    if result.noise_band:
        nb = result.noise_band.get("messages")
        if nb:
            print(f"  noise band on messages: {nb.sample_count} samples, {len(nb.distinct_values)} distinct value(s)")

    print("\nperturbations:")
    for p in sorted(result.perturbations, key=lambda x: (x.fault_class, x.pattern_type)):
        verdict = (
            f"CRASH[{p.crash_class or p.error_type}]" if p.crashed
            else "DIVERGE" if p.diverged else "UNCHANGED"
        )
        print(f"  [{p.fault_class:<8s}] {p.pattern_type:<32s} {verdict:<24s} "
              f"{len(p.trace)} steps  {p.duration_s:.1f}s")
        print(f"             {p.event_summary[:150]}")
        if p.crashed:
            print(f"             error: {p.error[:150]}")
        elif p.diverged:
            print(f"             diff: {p.divergence_summary[:200]}")
        if p.final_state and not p.crashed:
            msgs = p.final_state.get("messages", [])
            last_ai = next((m["content"] for m in reversed(msgs) if m.get("type") == "ai" and m.get("content")), "")
            print(f"             last AI: {str(last_ai)[:160]!r}")
        for f in p.coordination_findings:
            print(f"             COORD [{f['failure_type']}] {f['summary'][:220]}")
        for d in p.filtered_divergences:
            print(f"             filtered t{d.tier}: {d.name} ({'noise band' if d.within_noise_band else d.judge_reasoning})")


async def _main(args) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    graph, label = _build(args.extended, args.model)
    question = QUESTIONS["extended" if args.extended else "basic"]
    init = {"messages": [{"role": "user", "content": question}]}
    print(f"target: {label}   model: {args.model}", file=sys.stderr)
    print(f"question: {question!r}", file=sys.stderr)
    t0 = time.perf_counter()
    result = await drift_test_async(
        graph=graph,
        initial_state=init,
        intensity="exhaustive",
        seed=11,
        divergence_mode="tiered",
        baseline_rollouts=args.baseline_rollouts,
        max_judge_calls=0,
    )
    print(f"wall clock: {time.perf_counter() - t0:.0f}s", file=sys.stderr)
    _print(result, label)
    if args.save_json:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        payload = _result_to_json(result)
        payload["_meta"] = {"target": label, "model": args.model, "question": question,
                            "wall_clock_s": round(time.perf_counter() - t0, 1)}
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nsaved: {out.relative_to(REPO_ROOT)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extended", action="store_true", help="5-specialist supervisor instead of the 2-agent README example")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--baseline-rollouts", type=int, default=2)
    p.add_argument("--save-json", action="store_true")
    asyncio.run(_main(p.parse_args()))


if __name__ == "__main__":
    main()
