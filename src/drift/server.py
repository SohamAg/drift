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


# Starter snippet for the Custom (BYOA) tab. Demonstrates the full
# bring-your-own pattern: custom WorldState subclass with extra domain
# fields, custom chaos Event subclass that mutates state, initial_state()
# and events() callables, plus four decorated agents. Designed to trigger
# multiple detectors (contradictory_review, hallucinated_reference,
# security_bypass) so the user sees immediate signal across families.
_BYOA_EXAMPLE_CODE = '''\
# ──────────────────────────────────────────────────────────────────────
# 1. Define your environment.  Subclass drift.WorldState to add the fields
# your domain needs. drift's general detectors read `open_cases`; the
# code-review topology adds attention to `security_status`.
# ──────────────────────────────────────────────────────────────────────
from drift.world import Case


class CodeReviewState(drift.WorldState):
    repository: str = "demo/repo"
    security_status: dict = {}   # PR id -> "blocked" | "clear"


# ──────────────────────────────────────────────────────────────────────
# 2. Define chaos events.  Each Event subclass has an apply(world) method
# that mutates the world state. Drift calls apply() at the scheduled step.
# ──────────────────────────────────────────────────────────────────────
class SecurityFinding(drift.Event):
    name = "SecurityFinding"

    def __init__(self, pr_id):
        super().__init__()
        self.pr_id = pr_id

    def apply(self, world):
        world.state.security_status[self.pr_id] = "blocked"
        return drift.EventRecord(
            event_id=self.event_id,
            timestep=world.state.timestep,
            name=self.name,
            summary=f"security blocked {self.pr_id}",
        )


# ──────────────────────────────────────────────────────────────────────
# 3. Pre-populate the world and schedule events.
# ──────────────────────────────────────────────────────────────────────
def initial_state():
    return CodeReviewState(
        open_cases={
            "PR-1": Case(case_id="PR-1", customer_id="alice",
                         issue="add dark mode toggle", opened_at_step=0),
        },
    )


def events():
    """Return [(timestep, event_instance), ...] — drift fires each at its step."""
    return [
        (3, SecurityFinding("PR-1")),
    ]


# ──────────────────────────────────────────────────────────────────────
# 4. Define your agents.  Replace the bodies with your own LLM / RAG /
# tool / framework calls — drift only needs the structured Action back.
# ──────────────────────────────────────────────────────────────────────
@drift.agent(role="reviewer", name="reviewer_a")
async def reviewer_a(state, memory):
    if state.open_cases:
        target = sorted(state.open_cases)[0]
        return drift.Action(
            kind="approve_review",
            target_case_id=target,
            rationale=f"reviewer_a approves {target}",
        )
    return drift.Action(kind="no_op")


@drift.agent(role="reviewer", name="reviewer_b")
async def reviewer_b(state, memory):
    # At t=4, deliberately reference a PR that doesn't exist — triggers
    # hallucinated_reference.
    if state.timestep == 4:
        return drift.Action(
            kind="reject_review",
            target_case_id="PR-PHANTOM",
            rationale="reviewer_b rejects a PR that doesn't exist",
        )
    if state.open_cases:
        target = sorted(state.open_cases)[0]
        return drift.Action(
            kind="reject_review",
            target_case_id=target,
            rationale=f"reviewer_b disagrees about {target}",
        )
    return drift.Action(kind="no_op")


@drift.agent(role="security")
async def security(state, memory):
    # Reads state.security_status — the SecurityFinding event sets it at t=3.
    for pr_id, status in state.security_status.items():
        if status == "blocked":
            return drift.Action(
                kind="security_block",
                target_case_id=pr_id,
                rationale=f"security has findings on {pr_id}",
            )
    return drift.Action(kind="no_op")


@drift.agent(role="merger")
async def merger(state, memory):
    # A simple merger that always tries to merge PR-1. Once security has
    # blocked it (from t=3 onward) this should trigger security_bypass.
    return drift.Action(
        kind="merge",
        target_case_id="PR-1",
        rationale="merger always merges",
    )
'''


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
    mode: str = "auto"   # "total" | "post_branch" | "auto" — auto picks post_branch when a relationship exists


class AnalyzeRequest(BaseModel):
    """Analyze a trace pasted/uploaded by the user. Backs the Analyze tab."""
    topology: str
    trace: str   # raw JSONL text — one record per line, each with a "type" field


class BYOARequest(BaseModel):
    """Run drift on user-supplied agent code. Backs the Custom (BYOA) tab.

    The code is executed as Python in the server process. Drift is pre-
    imported into the namespace. The user defines @drift.agent-decorated
    async functions; optionally an `initial_state()` callable returning a
    WorldState; optionally an `events()` callable returning a list of
    (timestep, Event) pairs.

    Note: this is `exec()` over arbitrary user code. Safe for local
    development; would need sandboxing for a hosted deployment.
    """
    code: str
    detector_topology: str = "support"   # which topology's detectors to layer on top of the general ones
    steps: int = Field(default=20, ge=1, le=200)
    seed: int = 42
    # Auto-chaos: drift generates chaos events from the user's WorldState
    # schema. "off" disables; "light"/"moderate"/"aggressive" scale density.
    # The auto-generated events run alongside any events() the user code defines.
    auto_chaos: str = "off"
    auto_chaos_exclude: list[str] = Field(default_factory=list)


class ForkRunRequest(BaseModel):
    """Fork an existing run at a chosen step with optional overrides."""
    branch_at_step: int = Field(ge=0)
    seed: int | None = None
    prompt_variants: dict[str, str] = Field(default_factory=dict)  # role -> 'naive'|'hardened'
    disabled_agents: list[str] = Field(default_factory=list)
    extend_by: int | None = None
    new_run_id: str | None = None


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


def _build_timeline(
    a_failures: list[dict],
    b_failures: list[dict],
    *,
    a_events: list[dict],
    b_events: list[dict],
    a_snap: list[dict],
    b_snap: list[dict],
    divergence_step: int | None,
) -> dict[str, Any]:
    """Per-timestep rollup for the divergence visualization.

    For each step that appears in either run, return what happened in A and
    what happened in B: event names that fired, failure types that triggered,
    and the snapshot's sentiment + open-cases count for quick visual context.
    """
    def per_step_rollup(failures, events, snaps):
        steps: dict[int, dict[str, Any]] = {}
        for e in events:
            t = int(e["timestep"])
            steps.setdefault(t, {"events": [], "failures": [], "sentiment": None, "open": None})
            steps[t]["events"].append(e["name"])
        for f in failures:
            t = int(f["timestep"])
            steps.setdefault(t, {"events": [], "failures": [], "sentiment": None, "open": None})
            steps[t]["failures"].append(f["failure_type"])
        for s in snaps:
            t = int(s["timestep"])
            steps.setdefault(t, {"events": [], "failures": [], "sentiment": None, "open": None})
            steps[t]["sentiment"] = s.get("customer_sentiment")
            steps[t]["open"] = len(s.get("open_cases", {}))
        return steps

    a_steps = per_step_rollup(a_failures, a_events, a_snap)
    b_steps = per_step_rollup(b_failures, b_events, b_snap)
    all_steps = sorted(set(a_steps.keys()) | set(b_steps.keys()))
    return {
        "divergence_step": divergence_step,
        "steps": [
            {
                "t": t,
                "a": a_steps.get(t),
                "b": b_steps.get(t),
            }
            for t in all_steps
        ],
    }


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
        # Lineage — present when this run is a fork.
        "parent_run_id": meta.get("parent_run_id"),
        "branch_at_step": meta.get("branch_at_step"),
        "fork_overrides": meta.get("fork_overrides"),
    }


# ---- background task ------------------------------------------------------

def _wire_progress(state: _RunState, runner: SimulationRunner) -> None:
    """Hook runner._tick so every step updates the polled `_RunState`.
    Shared between fresh runs and forks."""
    original_tick = runner._tick

    async def progress_tick(t: int) -> None:
        prev_events = len(runner.events)
        prev_failures = len(runner.failures)
        await original_tick(t)
        with _RUNS_LOCK:
            state.completed_steps = t
            state.failure_count = len(runner.failures)
            state.world_state = runner.world.state.model_dump(mode="json")
            ftype: dict[str, int] = {}
            for f in runner.failures:
                ftype[f.failure_type] = ftype.get(f.failure_type, 0) + 1
            state.failures_by_type = ftype
            state.recent_events = [e.model_dump(mode="json") for e in runner.events[-12:]]
            state.recent_failures = [f.model_dump(mode="json") for f in runner.failures[-8:]]
            state.recent_actions = [a.model_dump(mode="json") for a in runner.actions[-4:]]
            state.last_step_event_count = len(runner.events) - prev_events
            state.last_step_failure_count = len(runner.failures) - prev_failures
    runner._tick = progress_tick  # type: ignore[assignment]


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
        _wire_progress(state, runner)
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


async def _execute_fork(state: _RunState, parent_run_id: str, req: ForkRunRequest) -> None:
    """Runs a fork via drift.fork.build_fork. Updates `state` for polling."""
    from drift.fork import ForkConfig, ForkOverrides, build_fork
    try:
        state.status = "running"
        reset_all_counters()
        overrides = ForkOverrides(
            seed=req.seed,
            prompt_variants=dict(req.prompt_variants or {}),
            disabled_agents=set(req.disabled_agents or []),
        )
        cfg = ForkConfig(
            parent_run_id=parent_run_id,
            branch_at_step=req.branch_at_step,
            overrides=overrides,
            new_run_id=state.run_id,
            extend_by=req.extend_by,
        )
        runner, _meta = build_fork(cfg, runs_dir=RUNS_DIR)
        _wire_progress(state, runner)
        try:
            await runner.run()
        finally:
            if runner.logger:
                runner.logger.close()

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

    @app.post("/api/runs/{parent_run_id}/fork")
    async def fork_run(parent_run_id: str, req: ForkRunRequest) -> dict[str, Any]:
        parent_dir = RUNS_DIR / parent_run_id
        if not parent_dir.is_dir():
            raise HTTPException(404, f"parent run not found: {parent_run_id}")
        # Default new_run_id mirrors the fork.py convention.
        new_run_id = req.new_run_id or f"{parent_run_id}__fork_at_{req.branch_at_step}"
        if (RUNS_DIR / new_run_id).exists():
            raise HTTPException(409, f"run_id {new_run_id!r} already exists")

        # Approximate total steps for the live progress bar. The actual step
        # count is determined inside build_fork (parent total - branch_at_step
        # by default, or req.extend_by if given).
        try:
            parent_meta_path = parent_dir / "run_meta.json"
            parent_meta = json.loads(parent_meta_path.read_text(encoding="utf-8"))
            parent_total = int(parent_meta.get("steps", 0))
        except Exception:
            parent_total = 0
        approx_steps = req.extend_by if req.extend_by is not None \
            else max(1, parent_total - req.branch_at_step)

        state = _RunState(
            run_id=new_run_id,
            total_steps=approx_steps,
            request={
                "parent_run_id": parent_run_id,
                "branch_at_step": req.branch_at_step,
                "topology": parent_meta.get("topology") if "parent_meta" in dir() else None,
                **req.model_dump(),
            },
        )
        # Decorate the request dict with topology if we can read it from parent meta.
        try:
            state.request["topology"] = parent_meta.get("topology")
        except Exception:
            pass
        with _RUNS_LOCK:
            _RUNS[new_run_id] = state

        asyncio.create_task(_execute_fork(state, parent_run_id, req))
        return {"run_id": new_run_id, "status": state.status, "parent_run_id": parent_run_id}

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
        a_summary = _summarize_run_dir(a_dir)
        b_summary = _summarize_run_dir(b_dir)
        a_failures = _load_jsonl(a_dir / "failures.jsonl")
        b_failures = _load_jsonl(b_dir / "failures.jsonl")
        a_actions = _load_jsonl(a_dir / "actions.jsonl")
        b_actions = _load_jsonl(b_dir / "actions.jsonl")
        a_snap = _load_jsonl(a_dir / "snapshots.jsonl")
        b_snap = _load_jsonl(b_dir / "snapshots.jsonl")

        # ----- relationship detection ---------------------------------
        # Cases:
        #   parent_child: one run is the parent of the other (branch_at = child's branch_at_step)
        #   siblings: both runs share the same parent AND branch_at_step
        #   unrelated: no shared lineage
        a_pid = a_summary.get("parent_run_id")
        b_pid = b_summary.get("parent_run_id")
        a_bat = a_summary.get("branch_at_step")
        b_bat = b_summary.get("branch_at_step")
        relationship = "unrelated"
        divergence_step: int | None = None
        if b_pid == a_summary["run_id"] and b_bat is not None:
            relationship = "parent_child"
            divergence_step = int(b_bat)
        elif a_pid == b_summary["run_id"] and a_bat is not None:
            relationship = "parent_child"
            divergence_step = int(a_bat)
        elif a_pid and b_pid and a_pid == b_pid and a_bat == b_bat and a_bat is not None:
            relationship = "siblings"
            divergence_step = int(a_bat)

        # ----- mode resolution ---------------------------------------
        mode = req.mode
        if mode == "auto":
            mode = "post_branch" if (relationship != "unrelated") else "total"
        if mode == "post_branch" and divergence_step is None:
            mode = "total"

        # ----- aggregation helpers (optionally filtered by step) -----
        def by_type(failures, after_step: int | None):
            out: dict[str, int] = {}
            for f in failures:
                if after_step is not None and f.get("timestep", 0) <= after_step:
                    continue
                out[f["failure_type"]] = out.get(f["failure_type"], 0) + 1
            return out

        def by_agent_kind(actions, after_step: int | None):
            out: dict[str, dict[str, int]] = {}
            for x in actions:
                if after_step is not None and x.get("timestep", 0) <= after_step:
                    continue
                out.setdefault(x["agent_name"], {})
                out[x["agent_name"]][x["kind"]] = out[x["agent_name"]].get(x["kind"], 0) + 1
            return out

        # In post_branch mode, count only what happened strictly after the branch step.
        cutoff = divergence_step if mode == "post_branch" else None

        # Action and failure totals filtered to the active mode.
        a_filt_failures = [f for f in a_failures if cutoff is None or f.get("timestep", 0) > cutoff]
        b_filt_failures = [f for f in b_failures if cutoff is None or f.get("timestep", 0) > cutoff]
        a_filt_actions  = [x for x in a_actions  if cutoff is None or x.get("timestep", 0) > cutoff]
        b_filt_actions  = [x for x in b_actions  if cutoff is None or x.get("timestep", 0) > cutoff]

        return {
            "relationship": relationship,
            "divergence_step": divergence_step,
            "mode": mode,
            "a": {
                **a_summary,
                "failures_by_type": by_type(a_failures, cutoff),
                "actions_by_agent_kind": by_agent_kind(a_actions, cutoff),
                "final": a_snap[-1] if a_snap else {},
                "n_failures_in_mode": len(a_filt_failures),
                "n_actions_in_mode": len(a_filt_actions),
            },
            "b": {
                **b_summary,
                "failures_by_type": by_type(b_failures, cutoff),
                "actions_by_agent_kind": by_agent_kind(b_actions, cutoff),
                "final": b_snap[-1] if b_snap else {},
                "n_failures_in_mode": len(b_filt_failures),
                "n_actions_in_mode": len(b_filt_actions),
            },
            # Per-step rollups for the divergence timeline. Keys = timesteps.
            "timeline": _build_timeline(a_failures, b_failures,
                                        a_events=_load_jsonl(a_dir / "events.jsonl"),
                                        b_events=_load_jsonl(b_dir / "events.jsonl"),
                                        a_snap=a_snap, b_snap=b_snap,
                                        divergence_step=divergence_step),
        }

    @app.post("/api/analyze")
    def analyze(req: AnalyzeRequest) -> dict[str, Any]:
        """Run drift's detectors against a user-supplied trace.

        The trace is raw JSONL text where each line carries a "type" field of
        "snapshot" | "action" | "event". See TRACE_SCHEMA.md for the contract.
        Returns the same shape the CLI's analyze command produces.
        """
        from drift.analyze import analyze_records, _split_by_type

        if req.topology not in list_topologies():
            raise HTTPException(400, f"unknown topology: {req.topology}")

        # Parse each line. Surface line-level errors with a useful index so
        # the user can fix their paste without guessing.
        records: list[dict] = []
        for i, line in enumerate(req.trace.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise HTTPException(400, f"line {i}: invalid JSON ({e.msg})")

        try:
            snap_raw, act_raw, evt_raw = _split_by_type(records)
        except ValueError as e:
            raise HTTPException(400, str(e))

        try:
            failures, summary = analyze_records(snap_raw, act_raw, evt_raw, req.topology)
        except ValueError as e:
            raise HTTPException(400, str(e))

        return {
            "summary": summary,
            "failures": [f.model_dump(mode="json") for f in failures],
        }

    @app.post("/api/byoa")
    async def byoa(req: BYOARequest) -> dict[str, Any]:
        """Execute user-supplied agent code and run drift over it.

        Looks for in the user's code namespace:
          - any number of @drift.agent-decorated functions (collected as agents)
          - optional `initial_state()` returning a WorldState
          - optional `events()` returning a list of (timestep, Event) tuples

        Then runs drift.run_async with the detector list layered with the
        requested topology's domain-specific detectors.
        """
        import drift
        from drift.sdk import _BYOAgent, run_async
        from drift.topologies import get_topology

        if req.detector_topology not in list_topologies():
            raise HTTPException(400, f"unknown topology: {req.detector_topology}")

        # Build the execution namespace. We expose `drift` and a few core
        # symbols so users don't have to remember to import them. asyncio
        # is also available because users may write async helpers.
        namespace: dict[str, Any] = {
            "__name__": "__byoa__",
            "__builtins__": __builtins__,
            "drift": drift,
            "asyncio": asyncio,
        }

        # Reset action/failure counters so each BYOA run starts clean.
        from drift.testing import reset_all_counters
        reset_all_counters()

        try:
            exec(req.code, namespace)
        except SyntaxError as e:
            raise HTTPException(400, f"syntax error: {e.msg} (line {e.lineno})")
        except Exception as e:
            raise HTTPException(400, f"error executing user code: {type(e).__name__}: {e}")

        # Collect every _BYOAgent instance the user code produced.
        agents = [v for v in namespace.values() if isinstance(v, _BYOAgent)]
        if not agents:
            raise HTTPException(
                400,
                "no @drift.agent-decorated functions found. Define at least one "
                "function decorated with @drift.agent(role=...) at module level."
            )

        # Optional initial state.
        state = None
        init_fn = namespace.get("initial_state")
        if callable(init_fn):
            try:
                state = init_fn()
            except Exception as e:
                raise HTTPException(400, f"initial_state() raised: {type(e).__name__}: {e}")

        # Optional events list.
        events = None
        events_fn = namespace.get("events")
        if callable(events_fn):
            try:
                events = events_fn()
            except Exception as e:
                raise HTTPException(400, f"events() raised: {type(e).__name__}: {e}")

        # Detector list: GENERAL + the requested topology's specific bundle.
        topology = get_topology(req.detector_topology)
        detectors = topology.detectors

        try:
            result = await run_async(
                agents=agents,
                state=state,
                events=events,
                steps=req.steps,
                seed=req.seed,
                detectors=detectors,
                auto_chaos=req.auto_chaos if req.auto_chaos != "off" else None,
                auto_chaos_exclude=req.auto_chaos_exclude or None,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(500, f"run failed: {type(e).__name__}: {e}")

        return {
            "summary": {
                "agents": [{"name": a.name, "role": a.role} for a in agents],
                "steps_requested": req.steps,
                "steps_completed": result.final_state.timestep,
                "n_actions": len(result.actions),
                "n_events": len(result.events),
                "n_failures": len(result.failures),
                "n_auto_chaos_injected": len(result.auto_chaos_injected),
                "auto_chaos": req.auto_chaos,
                "detector_topology": req.detector_topology,
            },
            "failures": [f.model_dump(mode="json") for f in result.failures],
            "actions": [a.model_dump(mode="json") for a in result.actions],
            "events": [e.model_dump(mode="json") for e in result.events],
            "auto_chaos_injected": [e.model_dump(mode="json") for e in result.auto_chaos_injected],
            "final_state": result.final_state.model_dump(mode="json"),
        }

    @app.get("/api/byoa-example")
    def byoa_example() -> dict[str, Any]:
        """Return a starter snippet the Custom tab can pre-populate."""
        return {
            "detector_topology": "code_review",
            "code": _BYOA_EXAMPLE_CODE,
        }

    @app.get("/api/sample-trace")
    def sample_trace() -> dict[str, Any]:
        """Return the bundled support_sample.jsonl content so the UI's
        'Load sample' button has a single source of truth."""
        path = PROJECT_ROOT / "examples" / "traces" / "support_sample.jsonl"
        if not path.exists():
            raise HTTPException(404, "sample trace not bundled")
        return {"topology": "support", "trace": path.read_text(encoding="utf-8")}

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
