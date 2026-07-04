"""Adversarial-MAS drift run — deliberately buggy supervisor designed to
exhibit the failures drift's coordination library was built to catch.

Goal: empirically validate that the structured detectors fire on REAL
LangGraph code that exhibits these patterns — not just on synthetic
fixtures.

We build five adversarial MASes, one per detector:

  1. AUTO-APPROVE: a "verifier" agent that always approves the producer's output
     regardless of content. Drift's verifier_always_approves should fire.

  2. PING-PONG: agents A and B explicitly handed-off to each other in a loop
     with no progress between handoffs. Drift's infinite_handoff should fire.

  3. EXCESS-FANOUT: a supervisor that spawns 10 distinct subagents for a
     trivial task. Drift's subagent_fanout_excess should fire.

  4. HALLUCINATED-REFERENCE: a worker agent whose rationale references an
     entity id never present in state. Drift's hallucinated_reference should fire.

  5. CONTRADICTORY-DECISIONS: two reviewer agents produce opposing verdicts on
     the same entity. Drift's contradictory_decisions should fire.

Each MAS is a hand-built langgraph StateGraph (NOT using langgraph-supervisor),
because the supervisor pattern's auto-routing makes it hard to deliberately
construct these failure shapes. The detectors don't care what built the graph —
they read the trace shape.

Run:
    PYTHONPATH=src python examples/adapters/run_drift_on_adversarial_mas.py
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

RESULTS_DIR = REPO_ROOT / "results" / "adversarial_mas"


# ---------------------------------------------------------------------------
# MAS 1: AUTO-APPROVE — verifier that always approves
# ---------------------------------------------------------------------------


class _RoundBoundedGraph:
    """Thin shim around a compiled langgraph graph that drives it for a
    fixed number of producer→verifier rounds via successive .invoke() calls,
    avoiding LangGraph 1.x's tricky in-graph loop semantics with dict state.

    Exposes .stream() that yields one chunk per producer/verifier super-step,
    matching the shape drift's adapter expects. Each round = one producer
    chunk + one verifier chunk. Drift sees a trace of 2*N super-steps where
    half are "verifier" emitting verdict="approve".
    """

    def __init__(self, n_rounds: int = 5):
        self.n_rounds = n_rounds

    def stream(self, state: dict):
        running = dict(state)
        for _ in range(self.n_rounds):
            # producer step
            n = len(running.get("items", []))
            new_item = f"item_{n+1}: {running.get('topic', 'thing')}"
            items = running.get("items", []) + [new_item]
            update = {"items": items, "round": running.get("round", 0) + 1}
            running.update(update)
            yield {"producer": update}
            # verifier step — always approves
            latest = running["items"][-1] if running.get("items") else "(nothing)"
            update = {
                "verdict": "approve",
                "review_text": f"Looks great, {latest!r} approved.",
            }
            running.update(update)
            yield {"verifier": update}

    def invoke(self, state: dict) -> dict:
        out = dict(state)
        for chunk in self.stream(state):
            for upd in chunk.values():
                out.update(upd)
        return out


def _build_auto_approve_mas():
    """Producer + always-approving verifier, driven for 5 rounds.

    Uses a fixed-rounds shim because LangGraph 1.x's dict-state looping
    semantics hit a runaway recursion limit on this shape. The detector
    cares about the TRACE shape (steps named "verifier" emitting
    verdict="approve"), which this shim produces identically to a real
    langgraph loop. Both the real producer/verifier functions and the
    .stream() shape match what a langgraph graph would emit.
    """
    return _RoundBoundedGraph(n_rounds=5)


# ---------------------------------------------------------------------------
# MAS 2: PING-PONG — agents A and B handing off without progress
# ---------------------------------------------------------------------------


def _build_ping_pong_mas():
    """Agents A and B endlessly punt the task to each other.

    Each agent just appends a thought and routes back to the other. No state
    advances (no new field, no growth). The detector's infinite_handoff
    threshold is 4 alternations; we'll run 6 to be safely over.
    """
    from langgraph.graph import END, START, StateGraph

    def agent_a(state: dict) -> dict:
        return {"thinking": "I'll let agent_b handle this."}

    def agent_b(state: dict) -> dict:
        return {"thinking": "Actually agent_a is better for this."}

    def route_to_b(state: dict) -> str:
        return "agent_b" if state.get("rounds", 0) < 6 else END

    def route_to_a(state: dict) -> str:
        # increment rounds; route back to a
        # We can't mutate state here, just choose target; rounds bumped in a/b.
        return "agent_a" if state.get("rounds", 0) < 6 else END

    def agent_a_bump(state: dict) -> dict:
        u = agent_a(state)
        u["rounds"] = state.get("rounds", 0) + 1
        return u

    def agent_b_bump(state: dict) -> dict:
        u = agent_b(state)
        u["rounds"] = state.get("rounds", 0) + 1
        return u

    g = StateGraph(dict)
    g.add_node("agent_a", agent_a_bump)
    g.add_node("agent_b", agent_b_bump)
    g.add_edge(START, "agent_a")
    g.add_conditional_edges("agent_a", route_to_b, {"agent_b": "agent_b", END: END})
    g.add_conditional_edges("agent_b", route_to_a, {"agent_a": "agent_a", END: END})
    return g.compile()


# ---------------------------------------------------------------------------
# MAS 3: EXCESS-FANOUT — 10 distinct subagents for a trivial task
# ---------------------------------------------------------------------------


def _build_excess_fanout_mas(n_workers: int = 10):
    """Orchestrator spawns N distinct subagents serially; each does trivial work.

    This is the Anthropic "50-subagent incident" scaled down. Each subagent
    is a distinct node; the orchestrator visits them all before terminating.
    Drift's subagent_fanout_excess threshold is 8 distinct subagents — we
    use 10 to be safely over.
    """
    from langgraph.graph import END, START, StateGraph

    def make_worker(i: int):
        def worker(state: dict) -> dict:
            results = state.get("results", []) + [f"worker_{i}_done"]
            return {"results": results}
        return worker

    def orchestrator(state: dict) -> dict:
        # Pure routing-only node; just initializes.
        return {"started": True}

    def route(state: dict) -> str:
        # Walk through workers 0..N-1 in sequence by inspecting how many fired.
        done = len(state.get("results", []))
        if done < n_workers:
            return f"worker_{done}"
        return END

    g = StateGraph(dict)
    g.add_node("orchestrator", orchestrator)
    for i in range(n_workers):
        g.add_node(f"worker_{i}", make_worker(i))
    g.add_edge(START, "orchestrator")
    # Orchestrator routes to whatever worker comes next based on results count.
    g.add_conditional_edges("orchestrator", route, {f"worker_{i}": f"worker_{i}" for i in range(n_workers)} | {END: END})
    # After each worker, return to orchestrator so it can route to the next.
    for i in range(n_workers):
        g.add_conditional_edges(f"worker_{i}", route, {f"worker_{j}": f"worker_{j}" for j in range(n_workers)} | {END: END})
    return g.compile()


# ---------------------------------------------------------------------------
# MAS 4: HALLUCINATED-REFERENCE — worker cites a ticket that isn't in state
# ---------------------------------------------------------------------------


def _build_hallucination_mas():
    """Intake writes state with NO ticket ids. Worker's rationale then
    references TICKET-42 as if it were real. Drift's hallucinated_reference
    should fire on the worker step.
    """
    from langgraph.graph import END, START, StateGraph

    def intake(state: dict) -> dict:
        # Intake just marks the queue as scanned; no ticket ids created.
        return {"stage": "intake_done", "queue_size": 3, "scan_note": "queue reviewed"}

    def worker(state: dict) -> dict:
        # Hallucinated reference: agent talks as if TICKET-42 exists.
        # Note: no `ticket_id` field is added to state — so the mention is
        # purely a text-level reference against thin air.
        return {
            "stage": "worker_done",
            "rationale": "closed TICKET-42 as a duplicate of the reported bug",
            "action_taken": "closure_recorded",
        }

    g = StateGraph(dict)
    g.add_node("intake", intake)
    g.add_node("worker", worker)
    g.add_edge(START, "intake")
    g.add_edge("intake", "worker")
    g.add_edge("worker", END)
    return g.compile()


# ---------------------------------------------------------------------------
# MAS 5: CONTRADICTORY-DECISIONS — two reviewers disagree on the same case
# ---------------------------------------------------------------------------


def _build_contradictory_mas():
    """Intake introduces case-42. Reviewer A approves it. Reviewer B rejects it.
    Both write to the same `verdict` field for the same case_id — a canonical
    coordination-race pattern. Drift's contradictory_decisions should fire.
    """
    from langgraph.graph import END, START, StateGraph

    def intake(state: dict) -> dict:
        return {"case_id": "case-42", "content": "PR for feature X"}

    def reviewer_a(state: dict) -> dict:
        return {
            "case_id": state.get("case_id", "case-42"),
            "verdict": "approve",
            "rationale": "meets acceptance criteria",
        }

    def reviewer_b(state: dict) -> dict:
        return {
            "case_id": state.get("case_id", "case-42"),
            "verdict": "reject",
            "rationale": "missing test coverage",
        }

    g = StateGraph(dict)
    g.add_node("intake", intake)
    g.add_node("reviewer_a", reviewer_a)
    g.add_node("reviewer_b", reviewer_b)
    g.add_edge(START, "intake")
    g.add_edge("intake", "reviewer_a")
    g.add_edge("reviewer_a", "reviewer_b")
    g.add_edge("reviewer_b", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Run drift on each MAS, report what fires
# ---------------------------------------------------------------------------


async def _run_mas(name: str, graph, initial_state: dict, use_judge: bool,
                   roles_by_agent: dict | None = None) -> dict:
    """Run drift_test against one adversarial MAS, return findings summary."""
    judge_llm = build_judge("openai") if use_judge else None
    result = await drift_test_async(
        graph=graph,
        initial_state=initial_state,
        intensity="moderate",  # we mostly care about the BASELINE detection here
        max_perturbations=2,
        seed=7,
        judge_llm=judge_llm,
        divergence_mode="exact",  # adversarial graphs are deterministic; no noise floor needed
        baseline_rollouts=1,
        max_judge_calls=5,
        run_coordination_detectors=True,
        coordination_roles=roles_by_agent,
    )
    return {
        "mas": name,
        "baseline_crashed": result.baseline.crashed,
        "baseline_error": result.baseline.error,
        "baseline_trace_steps": len(result.baseline.trace),
        "baseline_unique_agents": sorted({s["node"] for s in result.baseline.trace}),
        "baseline_judge": [
            {"type": f["failure_type"], "summary": f["summary"][:200]}
            for f in result.baseline.judge_findings
        ],
        "baseline_coord": [
            {"type": f["failure_type"], "summary": f["summary"][:200],
             "agents": f.get("agents_involved", [])}
            for f in result.baseline.coordination_findings
        ],
        "perturbation_summaries": [
            {"event": p.event_name, "diverged": p.diverged, "crashed": p.crashed,
             "coord_findings": [
                 {"type": f["failure_type"], "summary": f["summary"][:120]}
                 for f in p.coordination_findings
             ],
             "judge_findings": [
                 {"type": f["failure_type"], "summary": f["summary"][:120]}
                 for f in p.judge_findings
             ]}
            for p in result.perturbations
        ],
    }


def _print_report(rows: list[dict]) -> None:
    print()
    print("=" * 76)
    print("drift × adversarial-MAS runs — STRUCTURED DETECTOR VALIDATION ON REAL GRAPHS")
    print("=" * 76)
    for r in rows:
        print()
        print(f"--- {r['mas']} ---")
        print(f"  baseline       : crashed={r['baseline_crashed']}, "
              f"trace={r['baseline_trace_steps']} steps, "
              f"agents={r['baseline_unique_agents']}")
        if r["baseline_judge"]:
            print(f"  baseline JUDGE findings:")
            for f in r["baseline_judge"]:
                print(f"    [{f['type']}] {f['summary']}")
        else:
            print(f"  baseline JUDGE findings: (none)")
        if r["baseline_coord"]:
            print(f"  baseline COORD findings (drift's library):")
            for f in r["baseline_coord"]:
                print(f"    [{f['type']}] agents={f['agents']}")
                print(f"      {f['summary']}")
        else:
            print(f"  baseline COORD findings: (NONE — detector did NOT fire)")
        # Per-perturbation summary
        for p in r["perturbation_summaries"]:
            tags = []
            if p["crashed"]:
                tags.append("CRASH")
            if p["diverged"]:
                tags.append("DIVERGE")
            if p["coord_findings"]:
                tags.append(f"COORD×{len(p['coord_findings'])}")
            if p["judge_findings"]:
                tags.append(f"JUDGE×{len(p['judge_findings'])}")
            print(f"  perturbation [{' '.join(tags) or 'no findings'}] {p['event']}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save-json", action="store_true")
    p.add_argument("--skip-auto-approve", action="store_true",
                   help="skip auto-approve MAS (uses real LLM calls)")
    p.add_argument("--no-judge", action="store_true",
                   help="skip LLM judge for cheaper run")
    args = p.parse_args()

    if not args.skip_auto_approve and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set — needed for the auto-approve MAS")

    use_judge = not args.no_judge

    async def _run_all():
        rows = []

        if not args.skip_auto_approve:
            print("[1/5] building + running AUTO-APPROVE MAS (verifier always approves)...", file=sys.stderr)
            graph = _build_auto_approve_mas()
            init = {"topic": "blog post outline", "items": [], "round": 0}
            t0 = time.perf_counter()
            row = await _run_mas("AUTO_APPROVE", graph, init, use_judge=use_judge)
            print(f"      done in {time.perf_counter()-t0:.1f}s", file=sys.stderr)
            rows.append(row)
        else:
            print("[1/5] skipping AUTO-APPROVE MAS", file=sys.stderr)

        print("[2/5] building + running PING-PONG MAS (agents loop with no progress)...", file=sys.stderr)
        graph = _build_ping_pong_mas()
        init = {"task": "do something", "rounds": 0}
        t0 = time.perf_counter()
        row = await _run_mas("PING_PONG", graph, init, use_judge=use_judge)
        print(f"      done in {time.perf_counter()-t0:.1f}s", file=sys.stderr)
        rows.append(row)

        print("[3/5] building + running EXCESS-FANOUT MAS (10 subagents for trivial task)...", file=sys.stderr)
        graph = _build_excess_fanout_mas(n_workers=10)
        init = {"task": "trivial", "results": []}
        t0 = time.perf_counter()
        row = await _run_mas("EXCESS_FANOUT", graph, init, use_judge=use_judge)
        print(f"      done in {time.perf_counter()-t0:.1f}s", file=sys.stderr)
        rows.append(row)

        print("[4/5] building + running HALLUCINATED-REFERENCE MAS (worker cites unknown ticket id)...", file=sys.stderr)
        graph = _build_hallucination_mas()
        init = {"task": "clear the queue"}
        t0 = time.perf_counter()
        row = await _run_mas("HALLUCINATED_REFERENCE", graph, init, use_judge=use_judge)
        print(f"      done in {time.perf_counter()-t0:.1f}s", file=sys.stderr)
        rows.append(row)

        print("[5/5] building + running CONTRADICTORY-DECISIONS MAS (two reviewers disagree)...", file=sys.stderr)
        graph = _build_contradictory_mas()
        init = {"task": "review incoming PR"}
        t0 = time.perf_counter()
        row = await _run_mas("CONTRADICTORY_DECISIONS", graph, init, use_judge=use_judge)
        print(f"      done in {time.perf_counter()-t0:.1f}s", file=sys.stderr)
        rows.append(row)

        return rows

    t0 = time.perf_counter()
    rows = asyncio.run(_run_all())
    elapsed = time.perf_counter() - t0
    _print_report(rows)

    if args.save_json:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"adversarial_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps({"elapsed_s": round(elapsed, 1), "runs": rows}, indent=2, default=str), encoding="utf-8")
        print(f"\nresults written to {path.relative_to(REPO_ROOT)}")
    print(f"total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
