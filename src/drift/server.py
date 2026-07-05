"""drift web server — FastAPI backend for the local UI.

After the 2026-06-29 cleanup all native-simulator endpoints (/api/runs,
/api/compare, /api/byoa, /api/topologies, /api/scenarios) were removed
along with the simulator. The server now exposes only adapter-shaped
endpoints:

  GET  /api/adapter-graphs           — list bundled graphs the Adapter tab can run
  POST /api/adapter-demo             — run drift_test against a chosen graph
  GET  /api/results                  — list saved experiment JSON under results/
  GET  /api/results/{relpath:path}   — fetch one saved result by path
  GET  /api/mast-demos               — list curated MAST demo traces
  POST /api/mast-analyze             — judge one MAST trace (cached or live)

The HTML/CSS/JS frontend is served from /web/.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# Resolved at import so endpoints find paths regardless of CWD.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
WEB_DIR: Path = PROJECT_ROOT / "web"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
MAST_DATASET: Path = PROJECT_ROOT / "data" / "external" / "mast" / "MAD_human_labelled_dataset.json"
MAST_CACHED_RESULTS_DIR: Path = PROJECT_ROOT / "results" / "mast_judge" / "full"


# Curated MAST traces for the "MAST" demo. Each entry pairs one MAST
# trace_id with a short human-readable narrative so the demo can show
# "real published multi-agent run X had these human-flagged failures —
# here is what drift's judge sees." Picked to span frameworks + outcomes
# (clean win, mixed, hard miss) so the demo is honest about scope.
MAST_DEMO_TRACES: list[dict[str, Any]] = [
    {
        "id": 2,
        "title": "AG2 — Christmas ribbon math",
        "mas_name": "AG2",
        "task_brief": "Math word problem with insufficient data. Agents debate how much ribbon to use per gift bow.",
        "story": "WIN",
        "story_blurb": "Drift caught both failure modes the human annotators flagged, with zero false alarms.",
    },
    {
        "id": 11,
        "title": "AppWorld — Bucket-list manager",
        "mas_name": "AppWorld",
        "task_brief": "Supervisor + Spotify/Notes/etc. agents coordinate to mark items off a bucket list.",
        "story": "MIXED",
        "story_blurb": "Drift caught 2 of 4 real failures (task drift and constraint violation) but missed both verification-related ones.",
    },
    {
        "id": 9,
        "title": "MetaGPT — Budget tracker app",
        "mas_name": "MetaGPT",
        "task_brief": "Coder + Tester agents build a budget tracker. Tests don't actually match the code's logic.",
        "story": "MIXED",
        "story_blurb": "Drift correctly flagged the reasoning-action mismatch but missed the backtracking failure.",
    },
    {
        "id": 3,
        "title": "ChatDev — Sudoku project (hard case)",
        "mas_name": "ChatDev",
        "task_brief": "Software dev pipeline (CEO/CTO/coder/tester) builds a Sudoku app over hundreds of agent turns.",
        "story": "MISS",
        "story_blurb": "Long-trace honest limit: 9 human-flagged failures, drift's current generic prompt caught none. Shows the prompt-iteration headroom.",
    },
]


# ---- request models -------------------------------------------------------

class MastAnalyzeRequest(BaseModel):
    """Judge one curated MAST trace, either via cached result or a live call."""
    trace_id: int
    mode: str = "cached"  # cached | live
    judge_model: str = "gpt-4o-mini"
    user_guidelines: list[str] = Field(default_factory=list)


class AdapterDemoRequest(BaseModel):
    """Run drift's langgraph adapter against a bundled graph.

    Backs the Adapter tab. Server-side graph build keeps user code out of
    the request surface — safe to expose. For arbitrary graphs the user
    installs drift locally and calls drift_test from a notebook.
    """
    # Which graph to run.
    # "ticket_triage" — bundled 3-node demo, always available.
    # "langgraph_supervisor" — requires langgraph-supervisor + OPENAI_API_KEY.
    graph_name: str = "ticket_triage"
    # Per-graph overrides; currently used to pass the user's question.
    state_overrides: dict[str, Any] = Field(default_factory=dict)

    # Chaos knobs.
    # off | light | moderate | aggressive | exhaustive.
    intensity: str = "aggressive"
    seed: int = 7
    max_perturbations: int = Field(default=25, ge=1, le=500)
    auto_chaos_exclude: list[str] = Field(default_factory=list)

    # Judge config (optional LLM over traces).
    judge: str = "off"
    judge_model: str | None = None
    user_guidelines: list[str] = Field(default_factory=list)

    # Divergence cascade.
    divergence_mode: str = "exact"
    baseline_rollouts: int = Field(default=1, ge=1, le=10)
    max_judge_calls: int = Field(default=10, ge=0, le=100)
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)


class AdapterForkRequest(BaseModel):
    """Fork-edit-replay a completed adapter run at a specific step.

    Design spec: docs/design/fork_edit_replay_v1.md. Deferred features live
    in memory/feature_ideas.md; ship v1 (state edit + top-vs-bottom compare)
    only.

    The client passes back the parent's baseline trace + initial state (data
    it already received from /api/adapter-demo). The server rebuilds the
    graph (using the same graph_name + state_overrides) and calls the fork
    API. This avoids a server-side graph cache — trade a redundant graph
    build for statelessness. Cheap.
    """
    # Same fields as AdapterDemoRequest for graph rebuild.
    graph_name: str = "ticket_triage"
    state_overrides: dict[str, Any] = Field(default_factory=dict)

    # Parent baseline the client received from /api/adapter-demo.
    parent_baseline_trace: list[dict[str, Any]] = Field(default_factory=list)
    parent_initial_state: dict[str, Any] = Field(default_factory=dict)

    # Fork params.
    fork_step: int = Field(..., ge=1, le=500)
    edits: dict[str, Any] = Field(default_factory=dict)
    also_apply_at_initial: bool = False

    # Optional coordination role overrides for the coord detectors on the
    # forked trace.
    coordination_roles: dict[str, str] = Field(default_factory=dict)


# ---- adapter graph registry -----------------------------------------------

def _list_adapter_graphs() -> list[dict[str, Any]]:
    """Return user-facing metadata for every graph the Adapter tab can run.

    Includes an `available` flag so the UI can grey out options whose
    deps / env vars aren't satisfied.
    """
    import os

    sup_pkg_ok = True
    try:
        import langgraph_supervisor  # noqa: F401
        import langchain_openai  # noqa: F401
    except ImportError:
        sup_pkg_ok = False
    sup_key_ok = bool(os.environ.get("OPENAI_API_KEY"))

    return [
        {
            "name": "ticket_triage",
            "label": "Ticket Triage (3-node demo)",
            "description": (
                "Bundled 3-node demo: classify -> (escalate | respond). "
                "Both terminal nodes look up open_tickets[ticket_id] without "
                "a defensive check. Free, deterministic, always available."
            ),
            "agents": ["classify", "escalate", "respond"],
            "available": True,
            "unavailable_reason": "",
            "supports_query_override": True,
            "query_field_label": "Ticket text",
            "query_default": "site is down can someone help",
            "needs_openai": False,
        },
        {
            "name": "langgraph_supervisor",
            "label": "LangGraph Supervisor (math + research)",
            "description": (
                "Canonical math+research supervisor from langchain-ai/"
                "langgraph-supervisor-py's README. Supervisor delegates to "
                "math_expert (add/multiply) or research_expert (mocked search). "
                "Each run = ~5-15 OpenAI calls per perturbation."
            ),
            "agents": ["supervisor", "math_expert", "research_expert"],
            "available": sup_pkg_ok and sup_key_ok,
            "unavailable_reason": (
                "" if (sup_pkg_ok and sup_key_ok)
                else (
                    "missing OPENAI_API_KEY" if sup_pkg_ok
                    else "install with: pip install drift[validation]"
                )
            ),
            "supports_query_override": True,
            "query_field_label": "User message",
            "query_default": "What is 7 times 8?",
            "needs_openai": True,
        },
    ]


def _build_demo_graph(name: str, overrides: dict[str, Any]) -> tuple[Any, dict, dict[str, Any]]:
    if name == "ticket_triage":
        return _build_ticket_triage_demo(overrides)
    if name == "langgraph_supervisor":
        return _build_langgraph_supervisor_demo(overrides)
    raise ValueError(f"unknown graph_name {name!r}; see /api/adapter-graphs")


def _build_ticket_triage_demo(overrides: dict[str, Any]) -> tuple[Any, dict, dict[str, Any]]:
    """The bundled 3-node demo."""

    def _classify(state: dict) -> dict:
        text = (state.get("text") or "").lower()
        if "urgent" in text or "down" in text:
            priority = "high"
        elif "question" in text:
            priority = "low"
        else:
            priority = "normal"
        return {"priority": priority}

    def _escalate(state: dict) -> dict:
        tid = state["ticket_id"]
        ticket = state["open_tickets"][tid]
        return {
            "reply": f"escalated ticket {tid} ({ticket['issue']}) to on-call",
            "escalated": True,
        }

    def _respond(state: dict) -> dict:
        tid = state["ticket_id"]
        ticket = state["open_tickets"][tid]
        return {
            "reply": f"resolved ticket {tid}: {ticket['issue']}",
            "escalated": False,
        }

    class _DemoGraph:
        def stream(self, state: dict):
            merged = dict(state)
            u1 = _classify(merged)
            merged.update(u1)
            yield {"classify": u1}
            if merged.get("priority") == "high" or merged.get("is_premium"):
                u2 = _escalate(merged)
                merged.update(u2)
                yield {"escalate": u2}
            else:
                u2 = _respond(merged)
                merged.update(u2)
                yield {"respond": u2}

        def invoke(self, state: dict) -> dict:
            out = dict(state)
            for chunk in self.stream(state):
                for upd in chunk.values():
                    out.update(upd)
            return out

    text = str(overrides.get("query") or "site is down can someone help")
    initial_state = {
        "ticket_id": "TKT-42",
        "text": text,
        "is_premium": True,
        "open_tickets": {
            "TKT-42": {"issue": "checkout 500s", "customer": "acme"},
        },
        "reply": "",
        "escalated": False,
        "priority": "",
    }
    meta = {
        "name": "ticket_triage",
        "description": (
            "3-node ticket triage: classify -> (escalate | respond). "
            "Both terminal nodes look up open_tickets[ticket_id] without "
            "a defensive check — production-realistic and brittle."
        ),
        "agents": ["classify", "escalate", "respond"],
    }
    return _DemoGraph(), initial_state, meta


def _build_langgraph_supervisor_demo(overrides: dict[str, Any]) -> tuple[Any, dict, dict[str, Any]]:
    """Canonical langgraph-supervisor README example."""
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "langgraph_supervisor graph requires OPENAI_API_KEY in environment "
            "or .env. Pick a different graph or set the key and restart the server."
        )
    try:
        from langchain_openai import ChatOpenAI  # noqa: F401
        from langgraph.prebuilt import create_react_agent  # noqa: F401
        from langgraph_supervisor import create_supervisor  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"langgraph_supervisor graph requires extra packages; install with "
            f"`pip install drift[validation]`. Underlying error: {e}"
        )

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*langgraph.*")
        from pathlib import Path
        import sys as _sys
        _harness_dir = PROJECT_ROOT / "examples" / "adapters"
        if str(_harness_dir) not in _sys.path:
            _sys.path.insert(0, str(_harness_dir))
        from run_drift_on_langgraph_supervisor import (  # type: ignore
            _build_supervisor_mas,
            _initial_state,
        )
        graph = _build_supervisor_mas(model_name="gpt-4o-mini")

    question = str(overrides.get("query") or "What is 7 times 8?")
    initial_state = _initial_state(question)
    meta = {
        "name": "langgraph_supervisor",
        "description": (
            "Canonical math+research supervisor from langgraph-supervisor-py's "
            "README. Supervisor delegates to math_expert (add/multiply) or "
            "research_expert (mocked search). Real OpenAI calls each step."
        ),
        "agents": ["supervisor", "math_expert", "research_expert"],
    }
    return graph, initial_state, meta


# ---- MAST helpers ---------------------------------------------------------

def _shape_mast_response(entry: dict[str, Any], result: dict[str, Any], *, mode: str) -> dict[str, Any]:
    """Format a MAST per-trace result for the UI."""
    per_mode = result.get("per_mode", [])
    n_tp = sum(1 for m in per_mode if m["outcome"] == "TP")
    n_fp = sum(1 for m in per_mode if m["outcome"] == "FP")
    n_fn = sum(1 for m in per_mode if m["outcome"] == "FN")
    n_tn = sum(1 for m in per_mode if m["outcome"] == "TN")
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else None
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None

    return {
        "demo_meta": entry,
        "mode": mode,
        "trace_id": result.get("trace_id"),
        "mas_name": result.get("mas_name"),
        "benchmark_name": result.get("benchmark_name"),
        "n_chars": result.get("n_chars"),
        "truncated": result.get("truncated", False),
        "latency_s": result.get("latency_s"),
        "predictions": result.get("predictions", []),
        "per_mode": per_mode,
        "summary": {
            "n_tp": n_tp, "n_fp": n_fp, "n_fn": n_fn, "n_tn": n_tn,
            "n_ground_truth_positives": n_tp + n_fn,
            "n_predicted_positives": n_tp + n_fp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
    }


# ---- app ------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(title="drift", version="0.2.0")

    # ---- LangGraph adapter demo (Adapter tab) -----------------------------

    @app.get("/api/adapter-graphs")
    def adapter_graphs() -> dict[str, Any]:
        """List which bundled graphs the Adapter tab can run, with their
        descriptions, default state shape, and availability (deps + API key)."""
        return {"graphs": _list_adapter_graphs()}

    @app.get("/api/results")
    def results_index() -> dict[str, Any]:
        """List every saved experiment JSON under results/, grouped by
        experiment subdirectory. Powers the Results browser in the UI."""
        groups: dict[str, list[dict[str, Any]]] = {}
        if RESULTS_DIR.exists():
            for path in sorted(RESULTS_DIR.rglob("*.json")):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                rel = path.relative_to(RESULTS_DIR)
                group = rel.parts[0] if len(rel.parts) > 1 else "(root)"
                groups.setdefault(group, []).append({
                    "name": rel.name,
                    "path": str(rel).replace("\\", "/"),
                    "size_bytes": stat.st_size,
                    "modified_ts": stat.st_mtime,
                })
        for entries in groups.values():
            entries.sort(key=lambda e: e["modified_ts"], reverse=True)
        return {"groups": groups}

    @app.get("/api/results/{relpath:path}")
    def results_file(relpath: str) -> Any:
        """Serve a single result JSON file by relative path. Restricted to
        results/ and bare *.json to keep the surface boring."""
        results_root = RESULTS_DIR.resolve()
        try:
            full = (results_root / relpath).resolve()
        except (ValueError, OSError) as e:
            raise HTTPException(400, f"bad path: {e}")
        if not str(full).startswith(str(results_root)):
            raise HTTPException(403, "path escapes results/")
        if not full.exists() or not full.is_file():
            raise HTTPException(404, "not found")
        if full.suffix != ".json":
            raise HTTPException(400, "only .json supported")
        try:
            return json.loads(full.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise HTTPException(500, f"could not read: {e}")

    @app.post("/api/adapter-demo")
    async def adapter_demo(req: AdapterDemoRequest) -> dict[str, Any]:
        """Run drift's auto-chaos against a bundled langgraph-shaped graph."""
        from drift.adapters.langgraph import drift_test_async
        from drift.failures.judge import build_judge

        try:
            graph, initial_state, graph_meta = _build_demo_graph(
                req.graph_name, req.state_overrides
            )
        except (ValueError, ImportError, RuntimeError) as e:
            raise HTTPException(400, str(e))

        try:
            judge_llm = build_judge(req.judge, model=req.judge_model)
        except ValueError as e:
            raise HTTPException(400, str(e))

        try:
            result = await drift_test_async(
                graph=graph,
                initial_state=initial_state,
                intensity=req.intensity,
                seed=req.seed,
                auto_chaos_exclude=req.auto_chaos_exclude or None,
                max_perturbations=req.max_perturbations,
                judge_llm=judge_llm,
                user_guidelines=req.user_guidelines or None,
                divergence_mode=req.divergence_mode,
                baseline_rollouts=req.baseline_rollouts,
                max_judge_calls=req.max_judge_calls,
                similarity_threshold=req.similarity_threshold,
            )
        except (TypeError, ValueError) as e:
            raise HTTPException(400, str(e))

        def _divergence_dict(d: Any) -> dict[str, Any]:
            return {
                "name": d.name,
                "tier": d.tier,
                "baseline_value": d.baseline_value,
                "perturbed_value": d.perturbed_value,
                "summary": d.summary,
                "similarity_score": d.similarity_score,
                "within_noise_band": d.within_noise_band,
                "judge_equivalent": d.judge_equivalent,
                "judge_reasoning": d.judge_reasoning,
            }

        def _baseline_dict(b: Any) -> dict[str, Any]:
            return {
                "initial_state": b.initial_state,
                "final_state": b.final_state,
                "crashed": b.crashed,
                "error": b.error,
                "error_type": b.error_type,
                "duration_s": round(b.duration_s, 4),
                "trace": b.trace,
                "judge_findings": b.judge_findings,
                "coordination_findings": b.coordination_findings,
            }

        def _pert_dict(p: Any) -> dict[str, Any]:
            return {
                "event_name": p.event_name,
                "event_summary": p.event_summary,
                "perturbed_field": p.perturbed_field,
                "pattern_type": p.pattern_type,
                "perturbed_initial_state": p.perturbed_initial_state,
                "final_state": p.final_state,
                "crashed": p.crashed,
                "error": p.error,
                "error_type": p.error_type,
                "diverged": p.diverged,
                "divergence_summary": p.divergence_summary,
                "duration_s": round(p.duration_s, 4),
                "trace": p.trace,
                "judge_findings": p.judge_findings,
                "coordination_findings": p.coordination_findings,
                "divergence_details": [_divergence_dict(d) for d in p.divergence_details],
                "filtered_divergences": [_divergence_dict(d) for d in p.filtered_divergences],
            }

        noise_band_dump = {
            name: {
                "name": band.name,
                "sample_count": band.sample_count,
                "distinct_values": band.distinct_values,
                "value_frequencies": band.value_frequencies,
                "text_min_similarity": band.text_min_similarity,
                "text_mean_similarity": band.text_mean_similarity,
                "numeric_min": band.numeric_min,
                "numeric_max": band.numeric_max,
            }
            for name, band in result.noise_band.items()
        }

        return {
            "graph_name": graph_meta["name"],
            "graph_description": graph_meta["description"],
            "graph_agents": graph_meta.get("agents", []),
            "intensity": result.intensity,
            "seed": req.seed,
            "judge": req.judge,
            "judge_model": req.judge_model,
            "n_user_guidelines": len(req.user_guidelines or []),
            "divergence_mode": result.divergence_mode,
            "baseline_rollouts": result.baseline_rollouts,
            "judge_calls_used": result.judge_calls_used,
            "judge_calls_budget": result.judge_calls_budget,
            "noise_band": noise_band_dump,
            "patterns_total": result.patterns_total,
            "n_crashed": result.n_crashed,
            "n_diverged": result.n_diverged,
            "n_unchanged": result.n_unchanged,
            "n_judge_findings": result.n_judge_findings,
            "n_coordination_findings": result.n_coordination_findings,
            "n_filtered_divergences": result.n_filtered_divergences,
            "summary_lines": result.summary_lines(),
            "baseline": _baseline_dict(result.baseline),
            "perturbations": [_pert_dict(p) for p in result.perturbations],
        }

    @app.post("/api/adapter-fork")
    async def adapter_fork(req: AdapterForkRequest) -> dict[str, Any]:
        """Fork-edit-replay a completed adapter run.

        Design spec: docs/design/fork_edit_replay_v1.md.

        Client sends: graph_name + state_overrides (to rebuild the same
        graph), parent baseline trace + initial state (to fork from),
        fork_step + edits (what to change), and the top-vs-bottom opt-in.
        Server rebuilds the graph, wraps the parent data into the shape
        drift_test_fork_async expects, and runs the fork.
        """
        from drift.adapters.langgraph import (
            AdapterResult,
            BaselineResult,
            drift_test_fork_async,
        )

        try:
            graph, _initial_ignored, graph_meta = _build_demo_graph(
                req.graph_name, req.state_overrides
            )
        except (ValueError, ImportError, RuntimeError) as e:
            raise HTTPException(400, str(e))

        # Wrap the parent-baseline snapshot into an AdapterResult shape so
        # drift_test_fork_async can validate + read from it. Only baseline
        # is populated — we don't need the parent's perturbations here.
        parent_shim = AdapterResult(
            baseline=BaselineResult(
                initial_state=dict(req.parent_initial_state),
                trace=list(req.parent_baseline_trace),
            ),
        )

        try:
            fork = await drift_test_fork_async(
                graph=graph,
                parent_result=parent_shim,
                fork_step=req.fork_step,
                edits=req.edits,
                also_apply_at_initial=req.also_apply_at_initial,
                coordination_roles=req.coordination_roles or None,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))

        def _branch_dict(b: Any) -> dict[str, Any]:
            return {
                "initial_state": b.initial_state,
                "trace": b.trace,
                "final_state": b.final_state,
                "crashed": b.crashed,
                "error": b.error,
                "error_type": b.error_type,
                "duration_s": round(b.duration_s, 4),
                "coordination_findings": b.coordination_findings,
            }

        return {
            "graph_name": graph_meta["name"],
            "graph_description": graph_meta["description"],
            "fork_step": fork.fork_step,
            "edits": fork.edits,
            "fork_point_state": fork.fork_point_state,
            "edited_state_at_fork": fork.edited_state_at_fork,
            "fork_branch": _branch_dict(fork.fork_branch),
            "top_edited_branch": (
                _branch_dict(fork.top_edited_branch)
                if fork.top_edited_branch else None
            ),
            "n_coordination_findings": fork.n_coordination_findings,
            "duration_s": round(fork.duration_s, 4),
        }

    # ---- MAST demo --------------------------------------------------------

    @app.get("/api/mast-demos")
    def mast_demos() -> dict[str, Any]:
        """List the curated MAST demo traces with task + outcome metadata."""
        if not MAST_DATASET.exists():
            raise HTTPException(404, f"MAST dataset not bundled at {MAST_DATASET}")
        mast = json.loads(MAST_DATASET.read_text(encoding="utf-8"))
        mast_by_id = {r["trace_id"]: r for r in mast}

        out = []
        for entry in MAST_DEMO_TRACES:
            rec = mast_by_id.get(entry["id"])
            if not rec:
                continue
            cached_path = MAST_CACHED_RESULTS_DIR / f"trace_{entry['id']:03d}.json"
            cached = json.loads(cached_path.read_text(encoding="utf-8")) if cached_path.exists() else None
            gt_modes = []
            if cached:
                gt_modes = [m["name"] for m in cached.get("per_mode", []) if m.get("ground_truth")]
            out.append({
                **entry,
                "trace_chars": len(rec["trace"]),
                "trace_truncated": len(rec["trace"]) > 100_000,
                "benchmark_name": rec.get("benchmark_name"),
                "ground_truth_modes": gt_modes,
                "n_ground_truth_positives": len(gt_modes),
                "has_cached_result": cached is not None,
            })
        return {"traces": out, "judge_model_default": "gpt-4o-mini"}

    @app.post("/api/mast-analyze")
    async def mast_analyze(req: MastAnalyzeRequest) -> dict[str, Any]:
        """Run (or replay cached) drift's judge against one curated MAST trace."""
        entry = next((e for e in MAST_DEMO_TRACES if e["id"] == req.trace_id), None)
        if entry is None:
            raise HTTPException(400, f"trace_id {req.trace_id} is not a curated MAST demo trace")

        if req.mode == "cached":
            cached_path = MAST_CACHED_RESULTS_DIR / f"trace_{req.trace_id:03d}.json"
            if not cached_path.exists():
                raise HTTPException(404, f"no cached result for trace_id {req.trace_id}")
            cached = json.loads(cached_path.read_text(encoding="utf-8"))
            return _shape_mast_response(entry, cached, mode="cached")

        if req.mode == "live":
            from drift.failures.judge import OpenAIJudge
            from drift.failures.mast_eval import judge_one_trace

            mast = json.loads(MAST_DATASET.read_text(encoding="utf-8"))
            rec = next((r for r in mast if r["trace_id"] == req.trace_id), None)
            if rec is None:
                raise HTTPException(404, f"trace_id {req.trace_id} missing from MAST dataset")

            try:
                judge = OpenAIJudge(model=req.judge_model)
            except Exception as e:
                raise HTTPException(400, f"could not build judge: {type(e).__name__}: {e}")

            result = await judge_one_trace(
                rec, judge,
                user_guidelines=req.user_guidelines or None,
            )
            if "error" in result:
                raise HTTPException(500, f"judge call failed: {result['error']}")
            return _shape_mast_response(entry, result, mode="live")

        raise HTTPException(400, f"unknown mode {req.mode!r}; expected 'cached' or 'live'")

    # Static frontend.
    if WEB_DIR.exists():
        app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

        @app.get("/")
        def root() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")
    else:
        @app.get("/")
        def root() -> JSONResponse:
            return JSONResponse({"detail": "web/ directory not found", "api": "/api/adapter-graphs"})

    return app


def serve(host: str = "127.0.0.1", port: int = 8765, reload: bool = False) -> None:
    import uvicorn
    if reload:
        uvicorn.run("drift.server:create_app", host=host, port=port, factory=True, reload=True)
    else:
        uvicorn.run(create_app(), host=host, port=port)
