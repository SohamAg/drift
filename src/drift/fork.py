"""Counterfactual replay — fork a completed run at a chosen timestep.

A fork produces a new run that:
  - inherits the world state captured in the parent's snapshot at branch_at_step
  - inherits each agent's memory by replaying parent observations + actions
    up to that step (warm fork)
  - then re-runs from (branch_at_step + 1) forward with overrides applied:
      seed, per-role prompt variants, disabled agents

Limitations of warm-fork memory replay (v0):
  - Agent observations are reconstructed from the post-step snapshot at each
    historical step, not the pre-action observation point. Memory contents
    are therefore approximate, but the BOUNDED-memory cap (32 by default)
    means only the most recent entries influence next decisions, so the
    drift from "exact memory" stays small in practice.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from drift.agents.base import Agent
from drift.events.scheduler import EventScheduler
from drift.llm import ScriptedMockLLM
from drift.llm.base import LLMClient
from drift.observability.logger import RunLogger
from drift.simulation import SimulationRunner
from drift.testing import reset_all_counters
from drift.topologies import get_topology
from drift.world import World, WorldState


# ---- helpers --------------------------------------------------------------

def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing run_meta.json in {run_dir}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


# ---- fork config ----------------------------------------------------------

@dataclass
class ForkOverrides:
    """What can be changed at the fork point. None = inherit from parent."""
    seed: int | None = None
    prompt_variants: dict[str, str] = field(default_factory=dict)  # role -> 'naive'|'hardened'
    disabled_agents: set[str] = field(default_factory=set)


@dataclass
class ForkConfig:
    parent_run_id: str
    branch_at_step: int
    overrides: ForkOverrides = field(default_factory=ForkOverrides)
    new_run_id: str | None = None
    extend_by: int | None = None  # additional steps to run; default = original_total - branch_at_step
    llm: LLMClient | None = None  # if None, builds a mock seeded by the effective seed


# ---- core fork builder ----------------------------------------------------

def build_fork(
    cfg: ForkConfig,
    runs_dir: Path = Path("runs"),
) -> tuple[SimulationRunner, dict[str, Any]]:
    """Construct a SimulationRunner configured for a counterfactual fork.

    Returns (runner, fork_meta_dict). The runner has not yet been .run().
    """
    parent_dir = runs_dir / cfg.parent_run_id
    if not parent_dir.is_dir():
        raise FileNotFoundError(f"parent run not found: {parent_dir}")

    parent_meta = _read_meta(parent_dir)
    snapshots = _read_jsonl(parent_dir / "snapshots.jsonl")
    actions   = _read_jsonl(parent_dir / "actions.jsonl")

    if cfg.branch_at_step < 0 or cfg.branch_at_step > len(snapshots):
        raise ValueError(
            f"branch_at_step={cfg.branch_at_step} out of range "
            f"(parent has snapshots 1..{len(snapshots)})"
        )

    topology_name = parent_meta["topology"]
    topology = get_topology(topology_name)

    # Effective seed: override wins, else parent's seed.
    effective_seed = cfg.overrides.seed if cfg.overrides.seed is not None else int(parent_meta["seed"])

    # World restoration --------------------------------------------------
    # branch_at_step=0 means "start from the very beginning" — same as a fresh run.
    if cfg.branch_at_step == 0:
        initial = topology.initial_world()
    else:
        snap_dict = snapshots[cfg.branch_at_step - 1]  # snapshots[i] = state after step i+1 of 1-indexed
        initial = World(initial=WorldState.model_validate(snap_dict))
        initial.state.timestep = cfg.branch_at_step

    # LLM --------------------------------------------------------------
    llm = cfg.llm or ScriptedMockLLM(seed=effective_seed, role_handlers=topology.mock_handlers)

    # Agents ----------------------------------------------------------
    agents: list[Agent] = topology.agent_factory(llm)
    # Apply per-role prompt variants: override wins per role; fall back to parent default.
    parent_variant = parent_meta.get("prompt_variant", "naive")
    for a in agents:
        variant = cfg.overrides.prompt_variants.get(a.role, parent_variant)
        prompt = topology.prompts.get((a.role, variant)) \
            or topology.prompts.get((a.role, "naive"), a.system_prompt)
        a.system_prompt = prompt

    # Memory rehydration: walk parent history up to branch step and rebuild
    # each agent's bounded memory. We use the post-step snapshot as the
    # observation basis (see module docstring caveat).
    _rehydrate_memory(agents, snapshots, actions, cfg.branch_at_step)

    # Scheduler ------------------------------------------------------
    scenario_path = None
    if parent_meta.get("scenario"):
        scenario_path = Path("scenarios") / parent_meta["scenario"]
    sched = (
        EventScheduler.from_yaml(scenario_path, seed=effective_seed,
                                 event_registry=topology.event_registry)
        if scenario_path and scenario_path.exists()
        else EventScheduler.empty(seed=effective_seed,
                                  event_registry=topology.event_registry)
    )
    # If we're keeping the parent's seed, advance the scheduler's RNG so that
    # stochastic rolls for steps >= branch+1 align with where the parent left off.
    # If the user overrode the seed, just start fresh.
    if cfg.overrides.seed is None:
        for t in range(1, cfg.branch_at_step + 1):
            sched.tick(t)  # discard — we only want the RNG state side-effect

    # Steps to run ----------------------------------------------------
    parent_total = int(parent_meta["steps"])
    extend = cfg.extend_by if cfg.extend_by is not None else (parent_total - cfg.branch_at_step)
    if extend <= 0:
        raise ValueError(f"nothing to simulate: branch_at_step={cfg.branch_at_step}, extend_by={extend}")

    # Logger ----------------------------------------------------------
    new_run_id = cfg.new_run_id or f"{cfg.parent_run_id}__fork_at_{cfg.branch_at_step}"
    if (runs_dir / new_run_id).exists():
        raise FileExistsError(f"run_id already exists: {new_run_id}")
    logger = RunLogger(base_dir=runs_dir, run_id=new_run_id)

    # Persist fork metadata alongside the new run's logs.
    fork_meta: dict[str, Any] = {
        "topology": topology_name,
        "scenario": parent_meta.get("scenario"),
        "seed": effective_seed,
        "llm": parent_meta.get("llm", "mock"),
        "model": parent_meta.get("model"),
        "prompt_variant": parent_variant,
        "steps": extend,
        "started_at": None,  # filled in at run time by callers if desired
        # fork-specific
        "parent_run_id": cfg.parent_run_id,
        "branch_at_step": cfg.branch_at_step,
        "fork_overrides": {
            "seed": cfg.overrides.seed,
            "prompt_variants": cfg.overrides.prompt_variants,
            "disabled_agents": sorted(cfg.overrides.disabled_agents),
        },
    }
    (logger.run_dir / "run_meta.json").write_text(
        json.dumps(fork_meta, indent=2), encoding="utf-8"
    )

    runner = SimulationRunner(
        agents=agents,
        scheduler=sched,
        steps=extend,
        detectors=topology.detectors,
        logger=logger,
        initial_world=initial,
        start_step=cfg.branch_at_step,
        disabled_agents=cfg.overrides.disabled_agents,
    )
    return runner, fork_meta


def _rehydrate_memory(
    agents: list[Agent],
    snapshots: list[dict],
    actions: list[dict],
    branch_at_step: int,
) -> None:
    """Reconstruct each agent's bounded memory by replaying parent history.

    For each historical step t in [1..branch_at_step]:
      1. Recompute observation from the snapshot at t (post-step, approximate).
      2. Append observation entry to memory.
      3. Append the agent's action entry from the parent's action log, if any.
    """
    if branch_at_step <= 0:
        return

    actions_by_agent: dict[str, list[dict]] = {}
    for a in actions:
        actions_by_agent.setdefault(a["agent_name"], []).append(a)

    for agent in agents:
        my_actions = sorted(
            actions_by_agent.get(agent.name, []),
            key=lambda x: x["timestep"],
        )
        action_by_step = {a["timestep"]: a for a in my_actions}

        for t in range(1, branch_at_step + 1):
            snap_dict = snapshots[t - 1]
            try:
                state = WorldState.model_validate(snap_dict)
            except Exception:
                continue
            obs = agent.observe(state)
            agent.memory.remember(t, "observation",
                                  obs.model_dump_json(exclude_none=True))
            a = action_by_step.get(t)
            if a is not None:
                summary = f"{a['kind']} {a.get('target_case_id') or ''} :: {a.get('rationale','')}"
                agent.memory.remember(t, "action", summary)


# ---- one-shot helper for tests / CLI -------------------------------------

async def run_fork(cfg: ForkConfig, runs_dir: Path = Path("runs")):
    """Build + execute a fork. Resets module counters first for deterministic IDs."""
    reset_all_counters()
    runner, meta = build_fork(cfg, runs_dir=runs_dir)
    try:
        result = await runner.run()
    finally:
        if runner.logger:
            runner.logger.close()
    return result, meta
