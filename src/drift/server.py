"""drift web server — FastAPI backend for the local UI.

Endpoints:
  GET  /api/topologies                   — list registered topologies + their roles/detectors
  GET  /api/scenarios                    — list YAML scenarios on disk
  GET  /api/runs                         — summary list of runs in the runs dir
  GET  /api/runs/{run_id}                — full detail (events/actions/snapshots/failures)
  POST /api/runs                         — start a new run (returns immediately; runs in background)
  GET  /api/runs/{run_id}/status         — poll progress of a still-running simulation
  POST /api/compare                      — diff two completed runs

The HTML/CSS/JS frontend is served from /web/.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from drift.events.scheduler import EventScheduler
from drift.llm import ScriptedMockLLM
from drift.llm.base import LLMClient
from drift.observability.logger import RunLogger
from drift.simulation import SimulationRunner
from drift.testing import reset_all_counters
from drift.topologies import Topology, get_topology, list_topologies


# Resolved at startup so endpoints can find the right paths regardless of CWD.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]  # e:\drift
RUNS_DIR: Path = PROJECT_ROOT / "runs"
SCENARIOS_DIR: Path = PROJECT_ROOT / "scenarios"
WEB_DIR: Path = PROJECT_ROOT / "web"


# ---- in-memory tracker for in-flight runs --------------------------------

class _RunState:
    """Lifecycle of one simulation. Stored in a process-local dict so the
    frontend can poll progress while the background task makes progress."""

    def __init__(self, run_id: str, total_steps: int, request: dict[str, Any]) -> None:
        self.run_id = run_id
        self.total_steps = total_steps
        self.completed_steps = 0
        self.status: str = "queued"   # queued | running | done | failed
        self.error: str | None = None
        self.started_at: str = dt.datetime.now().isoformat(timespec="seconds")
        self.finished_at: str | None = None
        self.request = request
        self.failure_count = 0
        # Live snapshot data — updated each tick so the UI can render motion.
        self.world_state: dict[str, Any] = {}
        self.failures_by_type: dict[str, int] = {}
        self.recent_events: list[dict[str, Any]] = []
        self.recent_failures: list[dict[str, Any]] = []
        self.recent_actions: list[dict[str, Any]] = []
        self.last_step_event_count = 0
        self.last_step_failure_count = 0


_RUNS: dict[str, _RunState] = {}
_RUNS_LOCK = threading.Lock()


# ---- request/response models ---------------------------------------------

class StartRunRequest(BaseModel):
    topology: str
    scenario: str | None = None     # filename in scenarios/, or None for empty scenario
    steps: int = Field(default=30, ge=1, le=500)
    seed: int = 42
    llm: str = "mock"               # mock | openai | anthropic
    model: str | None = None
    prompt_variant: str = "naive"   # naive | hardened
    run_id: str | None = None       # optional human-friendly label; otherwise auto-generated


class CompareRequest(BaseModel):
    run_a: str
    run_b: str


# ---- helpers --------------------------------------------------------------

def _build_llm(req: StartRunRequest, topology: Topology) -> LLMClient:
    if req.llm == "mock":
        return ScriptedMockLLM(seed=req.seed, role_handlers=topology.mock_handlers)
    if req.llm == "openai":
        from drift.llm.openai_adapter import OpenAILLM
        return OpenAILLM(model=req.model or "gpt-4o-mini")
    if req.llm == "anthropic":
        from drift.llm.anthropic_adapter import AnthropicLLM
        return AnthropicLLM(model=req.model or "claude-haiku-4-5")
    raise HTTPException(400, f"unknown llm {req.llm!r}")


def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summarize_run_dir(run_dir: Path) -> dict[str, Any]:
    """Build the fields the runs-list view needs without loading every line."""
    failures = _load_jsonl(run_dir / "failures.jsonl")
    actions = _load_jsonl(run_dir / "actions.jsonl")
    snapshots = _load_jsonl(run_dir / "snapshots.jsonl")
    meta = {}
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    return {
        "run_id": run_dir.name,
        "n_actions": len(actions),
        "n_failures": len(failures),
        "n_snapshots": len(snapshots),
        "final_step": snapshots[-1]["timestep"] if snapshots else 0,
        "topology": meta.get("topology"),
        "scenario": meta.get("scenario"),
        "seed": meta.get("seed"),
        "llm": meta.get("llm"),
        "prompt_variant": meta.get("prompt_variant"),
        "started_at": meta.get("started_at"),
        "steps_requested": meta.get("steps"),
    }


# ---- background task ------------------------------------------------------

async def _execute_run(state: _RunState, req: StartRunRequest) -> None:
    """Runs the simulation. Updates `state` so the UI can poll progress."""
    try:
        state.status = "running"
        reset_all_counters()
        topology = get_topology(req.topology)
        llm = _build_llm(req, topology)
        agents = topology.agent_factory(llm)
        for a in agents:
            prompt = topology.prompts.get((a.role, req.prompt_variant)) \
                or topology.prompts.get((a.role, "naive"), a.system_prompt)
            a.system_prompt = prompt
        if req.scenario:
            scen_path = SCENARIOS_DIR / req.scenario
            if not scen_path.exists():
                raise FileNotFoundError(f"scenario not found: {req.scenario}")
            sched = EventScheduler.from_yaml(scen_path, seed=req.seed,
                                             event_registry=topology.event_registry)
        else:
            sched = EventScheduler.empty(seed=req.seed,
                                         event_registry=topology.event_registry)

        run_dir = RUNS_DIR / state.run_id
        logger = RunLogger(base_dir=RUNS_DIR, run_id=state.run_id)

        # Persist the run config alongside the logs so the UI can show it later.
        meta = {
            "topology": req.topology,
            "scenario": req.scenario,
            "seed": req.seed,
            "llm": req.llm,
            "model": req.model,
            "prompt_variant": req.prompt_variant,
            "steps": req.steps,
            "started_at": state.started_at,
        }
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        runner = SimulationRunner(
            agents=agents,
            scheduler=sched,
            steps=req.steps,
            detectors=topology.detectors,
            logger=logger,
            initial_world=topology.initial_world(),
        )

        # Wrap _tick so we can update progress as the simulation advances.
        original_tick = runner._tick

        async def progress_tick(t: int) -> None:
            prev_events = len(runner.events)
            prev_failures = len(runner.failures)
            await original_tick(t)
            with _RUNS_LOCK:
                state.completed_steps = t
                state.failure_count = len(runner.failures)
                state.world_state = runner.world.state.model_dump(mode="json")
                # Tally for the live ticker.
                ftype: dict[str, int] = {}
                for f in runner.failures:
                    ftype[f.failure_type] = ftype.get(f.failure_type, 0) + 1
                state.failures_by_type = ftype
                # Latest 12 events / 8 failures / 4 actions, newest last.
                state.recent_events = [e.model_dump(mode="json") for e in runner.events[-12:]]
                state.recent_failures = [f.model_dump(mode="json") for f in runner.failures[-8:]]
                state.recent_actions = [a.model_dump(mode="json") for a in runner.actions[-4:]]
                state.last_step_event_count = len(runner.events) - prev_events
                state.last_step_failure_count = len(runner.failures) - prev_failures
        runner._tick = progress_tick  # type: ignore[assignment]

        try:
            await runner.run()
        finally:
            logger.close()

        state.status = "done"
        state.finished_at = dt.datetime.now().isoformat(timespec="seconds")
    except Exception as e:
        state.status = "failed"
        state.error = f"{type(e).__name__}: {e}"
        state.finished_at = dt.datetime.now().isoformat(timespec="seconds")
        traceback.print_exc()


# ---- app ------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(title="drift", description="Multi-agent stress-test simulator")

    @app.get("/api/topologies")
    def topologies() -> list[dict[str, Any]]:
        out = []
        for name in list_topologies():
            t = get_topology(name)
            sample_agents = t.agent_factory(ScriptedMockLLM(seed=0))
            roles = [a.role for a in sample_agents]
            out.append({
                "name": t.name,
                "description": t.description,
                "roles": roles,
                "events": sorted(t.event_registry.keys()),
                "detectors": [d.__name__.replace("detect_", "") for d in t.detectors],
            })
        return out

    @app.get("/api/scenarios")
    def scenarios() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not SCENARIOS_DIR.exists():
            return out
        for p in sorted(SCENARIOS_DIR.glob("*.yaml")):
            try:
                import yaml
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except Exception:
                data = {}
            out.append({
                "filename": p.name,
                "name": data.get("name", p.stem),
                "scripted_count": len(data.get("scripted", []) or []),
                "stochastic_count": len(data.get("stochastic", []) or []),
                "events_used": sorted({e["name"] for e in (data.get("scripted") or [])}),
            })
        return out

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        if not RUNS_DIR.exists():
            return []
        return [
            _summarize_run_dir(d)
            for d in sorted(RUNS_DIR.iterdir(), reverse=True)
            if d.is_dir() and (d / "snapshots.jsonl").exists()
        ]

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, Any]:
        run_dir = RUNS_DIR / run_id
        if not run_dir.is_dir():
            raise HTTPException(404, f"run {run_id!r} not found")
        return {
            "summary":   _summarize_run_dir(run_dir),
            "events":    _load_jsonl(run_dir / "events.jsonl"),
            "actions":   _load_jsonl(run_dir / "actions.jsonl"),
            "snapshots": _load_jsonl(run_dir / "snapshots.jsonl"),
            "failures":  _load_jsonl(run_dir / "failures.jsonl"),
        }

    @app.post("/api/runs")
    async def start_run(req: StartRunRequest) -> dict[str, Any]:
        if req.topology not in list_topologies():
            raise HTTPException(400, f"unknown topology: {req.topology}")
        run_id = req.run_id or f"{req.topology}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        # Reject reuse of an existing run_id; runs/<id> would otherwise collide.
        if (RUNS_DIR / run_id).exists():
            raise HTTPException(409, f"run_id {run_id!r} already exists")
        RUNS_DIR.mkdir(parents=True, exist_ok=True)

        state = _RunState(run_id=run_id, total_steps=req.steps, request=req.model_dump())
        with _RUNS_LOCK:
            _RUNS[run_id] = state

        # Schedule the simulation as a background task on the running event loop.
        asyncio.create_task(_execute_run(state, req))
        return {"run_id": run_id, "status": state.status}

    @app.get("/api/runs/{run_id}/status")
    def run_status(run_id: str) -> dict[str, Any]:
        with _RUNS_LOCK:
            s = _RUNS.get(run_id)
        if s is None:
            # Maybe we restarted the server but the run completed before; check disk.
            if (RUNS_DIR / run_id / "snapshots.jsonl").exists():
                return {"run_id": run_id, "status": "done", "completed_steps": -1, "total_steps": -1}
            raise HTTPException(404, f"unknown run {run_id!r}")
        return {
            "run_id": s.run_id,
            "status": s.status,
            "completed_steps": s.completed_steps,
            "total_steps": s.total_steps,
            "failure_count": s.failure_count,
            "error": s.error,
            "started_at": s.started_at,
            "finished_at": s.finished_at,
            "topology": s.request.get("topology"),
            "scenario": s.request.get("scenario"),
            "world_state": s.world_state,
            "failures_by_type": s.failures_by_type,
            "recent_events": s.recent_events,
            "recent_failures": s.recent_failures,
            "recent_actions": s.recent_actions,
            "last_step_event_count": s.last_step_event_count,
            "last_step_failure_count": s.last_step_failure_count,
        }

    @app.post("/api/compare")
    def compare(req: CompareRequest) -> dict[str, Any]:
        a_dir = RUNS_DIR / req.run_a
        b_dir = RUNS_DIR / req.run_b
        if not a_dir.is_dir() or not b_dir.is_dir():
            raise HTTPException(404, "both runs must exist")
        a = _summarize_run_dir(a_dir)
        b = _summarize_run_dir(b_dir)
        a_failures = _load_jsonl(a_dir / "failures.jsonl")
        b_failures = _load_jsonl(b_dir / "failures.jsonl")
        a_actions = _load_jsonl(a_dir / "actions.jsonl")
        b_actions = _load_jsonl(b_dir / "actions.jsonl")
        a_snap = _load_jsonl(a_dir / "snapshots.jsonl")
        b_snap = _load_jsonl(b_dir / "snapshots.jsonl")

        def by_type(failures):
            out = {}
            for f in failures:
                out[f["failure_type"]] = out.get(f["failure_type"], 0) + 1
            return out

        def by_agent_kind(actions):
            out = {}
            for x in actions:
                out.setdefault(x["agent_name"], {})
                out[x["agent_name"]][x["kind"]] = out[x["agent_name"]].get(x["kind"], 0) + 1
            return out

        return {
            "a": {**a, "failures_by_type": by_type(a_failures), "actions_by_agent_kind": by_agent_kind(a_actions),
                  "final": a_snap[-1] if a_snap else {}},
            "b": {**b, "failures_by_type": by_type(b_failures), "actions_by_agent_kind": by_agent_kind(b_actions),
                  "final": b_snap[-1] if b_snap else {}},
        }

    # Static frontend.
    if WEB_DIR.exists():
        app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

        @app.get("/")
        def root() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")
    else:
        @app.get("/")
        def root() -> JSONResponse:
            return JSONResponse({"detail": "web/ directory not found", "api": "/api/topologies"})

    return app


def serve(host: str = "127.0.0.1", port: int = 8765, reload: bool = False) -> None:
    import uvicorn
    if reload:
        uvicorn.run("drift.server:create_app", host=host, port=port, factory=True, reload=True)
    else:
        uvicorn.run(create_app(), host=host, port=port)
