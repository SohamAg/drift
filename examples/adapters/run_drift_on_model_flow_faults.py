"""Model-layer and control-flow chaos, live (drift.chaos.checkpoint).

Part `model`: LangGraph's create_react_agent + the mocked web_search tool
with gpt-4o-mini. Each model fault is written onto the baseline checkpoint
where the agent's web_search call is pending, and the run resumes N times:

  model_bad_tool_args        the call loses its arguments → does the agent
                             read the validation error and retry correctly?
  model_unknown_tool         the call names a tool that doesn't exist →
                             does it recover, or answer from nothing?
  model_duplicate_tool_call  the tool executes twice → does anything notice?
  model_dropped_tool_call    the model "answers" with nothing → the run ends
                             with an empty answer (no recovery is possible:
                             there is no next model turn)

Part `flow`: the 2-agent langgraph-supervisor system (supervisor →
research_expert / math_expert). Every skippable root node is skipped at
the point the baseline ran it next: does the supervisor notice that the
expert it handed off to produced nothing?

Saves the UI-shaped JSON under results/model_flow_faults/ before printing.

    PYTHONPATH=src python examples/adapters/run_drift_on_model_flow_faults.py --part model --rollouts 3 --save-json
    PYTHONPATH=src python examples/adapters/run_drift_on_model_flow_faults.py --part flow --rollouts 2 --save-json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples" / "adapters"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from drift.adapters.langgraph import drift_test_async  # noqa: E402
from drift.adapters.serialize import shape_adapter_result  # noqa: E402
from drift.chaos.checkpoint import MODEL_FAULTS, skippable_nodes  # noqa: E402
from drift.guidelines import GuidelineSpec  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "model_flow_faults"
REACT_Q = "Search the web for the capital of France and tell me the answer in one sentence."
SUPER_Q = "Look up the population of Tokyo and the population of Paris, then compute their sum."


def _react(model: str):
    from langchain_openai import ChatOpenAI
    from run_drift_on_langgraph_supervisor import _web_search  # type: ignore

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(
            ChatOpenAI(model=model, temperature=0), tools=[_web_search],
            prompt="You are a helpful assistant. Use web_search when you need facts.",
        )


def _supervisor(model: str):
    from run_drift_on_langgraph_supervisor import _build_supervisor_mas, _initial_state  # type: ignore

    return _build_supervisor_mas(model_name=model), _initial_state(SUPER_Q)


async def _main(args) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    if args.part == "model":
        graph = _react(args.model)
        state = {"messages": [{"role": "user", "content": REACT_Q}]}
        chaos = [{"tool": "_web_search", "fault": f} for f in (args.faults or MODEL_FAULTS)]
        name, desc = "react_model_faults", "create_react_agent + mocked web_search, model-layer faults"
    else:
        graph, state = _supervisor(args.model)
        nodes = args.nodes or skippable_nodes(graph)
        chaos = [{"skip_node": n} for n in nodes]
        name, desc = "supervisor_flow_faults", "langgraph-supervisor 2-agent, skip_node faults"
    t0 = time.perf_counter()
    res = await drift_test_async(
        graph=graph, initial_state=state, intensity="off", seed=11, rules=GuidelineSpec(chaos=chaos),
        divergence_mode="exact", perturbation_rollouts=args.rollouts,
    )
    wall = time.perf_counter() - t0
    if args.save_json:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        shaped = shape_adapter_result(
            res, graph_meta={"name": name, "description": desc, "agents": []},
            seed=11, judge="off", judge_model=None, user_guidelines=None,
        )
        path = RESULTS_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        path.write_text(json.dumps(shaped, indent=1, default=str), encoding="utf-8")
        print(f"saved {path}")

    print(f"\n{args.model} · part={args.part} · {args.rollouts} rollouts/fault · "
          f"baseline {'CRASHED' if res.baseline.crashed else 'ok'} · {wall:.0f}s")
    print(f"baseline answer: {_final(res.baseline.final_state)!r}")
    for p in res.perturbations:
        n = p.n_rollouts
        fired = sum(1 for r in p.all_rollouts() if r.fault_fired)
        c = p.outcome_counts
        print(f"\n{p.event_name:44} fired {fired}/{n}  crashed {c['crashed']}/{n}  "
              f"diverged {c['diverged']}/{n}  unchanged {c['unchanged']}/{n}")
        print(f"    {p.event_summary[:220]}")
        for ftype, k in sorted(p.finding_rollout_counts.items()):
            print(f"    {ftype:34} {k}/{n}")
        for r in p.all_rollouts():
            tail = f"{r.error_type}: {r.error[:90]}" if r.crashed else repr(_final(r.final_state))
            print(f"    r{r.rollout_index}: {tail}"[:170])


def _final(state) -> str:
    msgs = (state or {}).get("messages") or []
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("type") == "ai" and m.get("content") and not m.get("tool_calls"):
            return str(m["content"])[:110]
    return ""


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("model", "flow"), default="model")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--faults", nargs="*", default=None, help=f"subset of: {', '.join(MODEL_FAULTS)}")
    ap.add_argument("--nodes", nargs="*", default=None, help="flow part: nodes to skip (default: all skippable)")
    ap.add_argument("--save-json", action="store_true")
    asyncio.run(_main(ap.parse_args()))
