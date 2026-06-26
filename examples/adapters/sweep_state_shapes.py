"""State-shape sweep — same supervisor MAS, varying initial-state shapes.

Real langgraph apps usually carry more than just `messages`. Production state
schemas often include session_id, user_id, metadata, conversation_history,
context dicts, feature flags. Each new field gives drift's schema-driven
chaos engine a new attack surface — different perturbation patterns fire
on bool vs dict vs str vs numeric fields.

This sweep validates drift on the canonical math+research supervisor with
SEVERAL different state shapes — minimal, with-metadata, with-history,
with-flags, kitchen-sink — so we can document which chaos patterns surface
which failure modes.

Each shape runs with tiered cascade + 2 baseline rollouts + LLM judge.

Usage:
    PYTHONPATH=src python examples/adapters/sweep_state_shapes.py
    PYTHONPATH=src python examples/adapters/sweep_state_shapes.py --quick
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*langgraph.*")

from drift.adapters.langgraph import drift_test_async  # noqa: E402
from drift.failures.judge import build_judge  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_drift_on_langgraph_supervisor import (  # noqa: E402
    _build_supervisor_mas,
    _summarize_perturbation,
)

RESULTS_DIR = REPO_ROOT / "results" / "state_shape_sweep"


QUESTION = "What is 7 times 8?"


# State shapes a real production langgraph app would actually plug in.
# Drift's schema-driven chaos picks patterns based on runtime types — bool
# fields get flip_bool, dict fields get clear_dict / remove_dict_key /
# inject_dict_key, str gets corrupt_string, list gets clear_list /
# duplicate_list_entry / reverse_list, numeric gets boundary etc.
SHAPES = {
    "minimal": lambda q: {
        "messages": [{"role": "user", "content": q}],
    },
    "with_metadata": lambda q: {
        "messages": [{"role": "user", "content": q}],
        "session_id": "sess_abc123",
        "user_id": "user_42",
        "trace_id": "trc_xyz",
    },
    "with_history": lambda q: {
        "messages": [{"role": "user", "content": q}],
        "conversation_history": [
            {"role": "user", "content": "Previous question 1"},
            {"role": "assistant", "content": "Previous answer 1"},
            {"role": "user", "content": "Previous question 2"},
            {"role": "assistant", "content": "Previous answer 2"},
        ],
        "turn_number": 3,
    },
    "with_flags": lambda q: {
        "messages": [{"role": "user", "content": q}],
        "is_premium": True,
        "is_admin": False,
        "verbose_mode": True,
        "max_tokens_remaining": 5000,
    },
    "with_context_dict": lambda q: {
        "messages": [{"role": "user", "content": q}],
        "context": {
            "locale": "en-US",
            "timezone": "America/New_York",
            "user_tier": "pro",
            "feature_set": ["math", "research", "vision"],
        },
    },
    "kitchen_sink": lambda q: {
        "messages": [{"role": "user", "content": q}],
        "session_id": "sess_xyz",
        "user_id": "user_99",
        "is_premium": True,
        "is_admin": False,
        "turn_number": 1,
        "max_tokens_remaining": 8000,
        "context": {"locale": "en-US", "feature_set": ["math"]},
        "conversation_history": [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "ok"},
        ],
        "open_tickets": {"TKT-1": {"status": "open"}},
    },
}


def _findings_summary(result) -> dict:
    # Inspect what chaos patterns fired — that's the key signal for this experiment.
    pattern_counts: Counter[str] = Counter()
    pattern_outcomes: dict[str, Counter] = defaultdict(Counter)
    field_outcomes: dict[str, Counter] = defaultdict(Counter)
    for p in result.perturbations:
        pat = p.pattern_type
        field = p.perturbed_field
        pattern_counts[pat] += 1
        if p.crashed:
            outcome = "CRASH"
        elif p.diverged:
            outcome = "DIVERGE"
        else:
            outcome = "UNCHANGED"
        pattern_outcomes[pat][outcome] += 1
        field_outcomes[field][outcome] += 1
    return {
        "n_perturbations": len(result.perturbations),
        "n_crashed": result.n_crashed,
        "n_diverged": result.n_diverged,
        "n_unchanged": result.n_unchanged,
        "n_judge_findings": result.n_judge_findings,
        "n_coord_findings": result.n_coordination_findings,
        "judge_calls_used": result.judge_calls_used,
        "baseline_crashed": result.baseline.crashed,
        "baseline_error": result.baseline.error,
        "baseline_trace_steps": len(result.baseline.trace),
        "baseline_judge": [
            {"type": f["failure_type"], "summary": f["summary"][:140]}
            for f in result.baseline.judge_findings
        ],
        "patterns_used": dict(pattern_counts),
        "pattern_outcomes": {pat: dict(c) for pat, c in pattern_outcomes.items()},
        "field_outcomes": {f: dict(c) for f, c in field_outcomes.items()},
        "perturbations": [_summarize_perturbation(p) for p in result.perturbations],
    }


async def _run_one(shape_name: str, shape_fn, *, model: str,
                   max_perturbations: int, intensity: str,
                   baseline_rollouts: int, use_judge: bool) -> dict:
    app = _build_supervisor_mas(model_name=model)
    init = shape_fn(QUESTION)
    judge_llm = build_judge("openai") if use_judge else None
    result = await drift_test_async(
        graph=app, initial_state=init,
        intensity=intensity, max_perturbations=max_perturbations,
        seed=7, judge_llm=judge_llm,
        divergence_mode="tiered",
        baseline_rollouts=baseline_rollouts,
        max_judge_calls=10,
    )
    s = _findings_summary(result)
    s["shape"] = shape_name
    s["initial_state_keys"] = sorted(init.keys())
    return s


def _print_report(per_shape: list[dict]) -> None:
    print()
    print("=" * 76)
    print("drift × langgraph-supervisor — STATE-SHAPE SWEEP")
    print("=" * 76)
    print(f"  one fixed question across {len(per_shape)} state shapes")
    print(f"  shape varies = chaos catalog varies = surface area varies")
    print()

    totals = {
        "perturbations": sum(s["n_perturbations"] for s in per_shape),
        "crashed": sum(s["n_crashed"] for s in per_shape),
        "diverged": sum(s["n_diverged"] for s in per_shape),
        "unchanged": sum(s["n_unchanged"] for s in per_shape),
        "judge": sum(s["n_judge_findings"] for s in per_shape),
        "coord": sum(s["n_coord_findings"] for s in per_shape),
    }
    for k, v in totals.items():
        print(f"  total {k:<14s}: {v}")
    print()

    for s in per_shape:
        print(f"--- SHAPE: {s['shape']} ---")
        print(f"  initial keys           : {s['initial_state_keys']}")
        print(f"  baseline               : {'CRASH' if s['baseline_crashed'] else 'ok'}, "
              f"trace={s['baseline_trace_steps']} steps")
        print(f"  perturbations          : n={s['n_perturbations']} "
              f"crashed={s['n_crashed']} diverged={s['n_diverged']} unchanged={s['n_unchanged']}")
        print(f"  judge findings         : {s['n_judge_findings']}  "
              f"coord findings: {s['n_coord_findings']}")
        if s["patterns_used"]:
            print(f"  chaos patterns fired   :")
            for pat, n in s["patterns_used"].items():
                outcomes = s["pattern_outcomes"].get(pat, {})
                outc_str = " ".join(f"{k}x{v}" for k, v in outcomes.items())
                print(f"     {n:>2d}x {pat:<35s} -> {outc_str}")
        if s["field_outcomes"]:
            print(f"  by perturbed field     :")
            for fld, outc in s["field_outcomes"].items():
                outc_str = " ".join(f"{k}x{v}" for k, v in outc.items())
                print(f"     {fld:<25s} -> {outc_str}")
        if s["baseline_judge"]:
            print(f"  baseline judge findings:")
            for f in s["baseline_judge"]:
                print(f"    [{f['type']}] {f['summary']}")
        print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true",
                   help="3 representative shapes only")
    p.add_argument("--max-perturbations", type=int, default=5)
    p.add_argument("--intensity", default="aggressive")
    p.add_argument("--baseline-rollouts", type=int, default=2)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--save-json", action="store_true")
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")

    if args.quick:
        shapes = {k: SHAPES[k] for k in ("minimal", "with_flags", "kitchen_sink")}
    else:
        shapes = SHAPES

    print(f"running drift on {len(shapes)} state shapes × 1 question", file=sys.stderr)
    print(f"  question      : {QUESTION!r}", file=sys.stderr)
    print(f"  intensity     : {args.intensity}", file=sys.stderr)
    print(f"  perturb cap   : {args.max_perturbations}", file=sys.stderr)
    print(f"  rollouts      : {args.baseline_rollouts}", file=sys.stderr)
    print(f"  judge         : {'off' if args.no_judge else 'openai'}", file=sys.stderr)
    print(file=sys.stderr)

    async def _run_all():
        out = []
        for i, (name, fn) in enumerate(shapes.items(), start=1):
            print(f"  [{i}/{len(shapes)}] shape={name}", file=sys.stderr)
            t0 = time.perf_counter()
            try:
                row = await _run_one(
                    name, fn, model=args.model,
                    max_perturbations=args.max_perturbations,
                    intensity=args.intensity,
                    baseline_rollouts=args.baseline_rollouts,
                    use_judge=not args.no_judge,
                )
                out.append(row)
                print(f"        done in {time.perf_counter()-t0:.1f}s "
                      f"(patterns_fired={len(row['patterns_used'])})", file=sys.stderr)
            except Exception as e:
                print(f"        FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return out

    t0 = time.perf_counter()
    rows = asyncio.run(_run_all())
    elapsed = time.perf_counter() - t0

    # Save BEFORE printing — a console-encoding crash during print shouldn't
    # destroy the experiment data.
    if args.save_json:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"shapes_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps({
            "config": {
                "question": QUESTION, "model": args.model,
                "intensity": args.intensity,
                "max_perturbations": args.max_perturbations,
                "baseline_rollouts": args.baseline_rollouts,
                "use_judge": not args.no_judge,
            },
            "elapsed_seconds": round(elapsed, 1),
            "per_shape": rows,
        }, indent=2, default=str), encoding="utf-8")
        print(f"results written to {path.relative_to(REPO_ROOT)}", file=sys.stderr)

    _print_report(rows)
    print(f"total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
