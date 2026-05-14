"""Counterfactual replay tests.

Strategy: run a deterministic parent, then fork it with various overrides and
assert that the fork (a) reproduces the parent's pre-branch state, (b) honors
the override, and (c) writes proper lineage metadata.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from drift.events.scheduler import EventScheduler
from drift.fork import ForkConfig, ForkOverrides, build_fork
from drift.llm.mock import ScriptedMockLLM
from drift.observability.logger import RunLogger
from drift.simulation import SimulationRunner
from drift.testing import reset_all_counters
from drift.topologies import get_topology


REPO = Path(__file__).resolve().parents[1]


def _run_parent(tmp_path: Path, *, topology="support",
                scenario="scenarios/policy_chaos.yaml",
                steps=15, seed=42, run_id="parent",
                prompt_variant="naive"):
    """Set up a deterministic parent run and write it to tmp_path/runs."""
    reset_all_counters()
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    topo = get_topology(topology)
    llm = ScriptedMockLLM(seed=seed, role_handlers=topo.mock_handlers)
    agents = topo.agent_factory(llm)
    for a in agents:
        prompt = topo.prompts.get((a.role, prompt_variant)) \
              or topo.prompts.get((a.role, "naive"), a.system_prompt)
        a.system_prompt = prompt

    sched = EventScheduler.from_yaml(
        REPO / scenario, seed=seed, event_registry=topo.event_registry
    )
    logger = RunLogger(base_dir=runs_dir, run_id=run_id)

    meta = {
        "topology": topology,
        "scenario": scenario.split("/")[-1],
        "seed": seed,
        "llm": "mock",
        "model": None,
        "prompt_variant": prompt_variant,
        "steps": steps,
    }
    (logger.run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    runner = SimulationRunner(
        agents=agents, scheduler=sched, steps=steps,
        detectors=topo.detectors,
        logger=logger, initial_world=topo.initial_world(),
    )
    result = asyncio.run(runner.run())
    logger.close()
    return runs_dir, result


def test_fork_at_zero_reproduces_parent_when_no_overrides(tmp_path):
    """Forking at step 0 with no overrides should yield the same action stream as the parent."""
    runs_dir, parent = _run_parent(tmp_path, steps=10, seed=99)

    reset_all_counters()
    runner, fork_meta = build_fork(
        ForkConfig(parent_run_id="parent", branch_at_step=0,
                   new_run_id="fork_zero", extend_by=10),
        runs_dir=runs_dir,
    )
    fork = asyncio.run(runner.run())
    runner.logger.close()

    a = [(x.timestep, x.agent_name, x.kind, x.target_case_id) for x in parent.actions]
    b = [(x.timestep, x.agent_name, x.kind, x.target_case_id) for x in fork.actions]
    assert a == b, "fork from t=0 should match parent action stream"


def test_fork_preserves_world_state_at_branch(tmp_path):
    """The fork's world at branch_at_step should equal the parent's snapshot there."""
    runs_dir, parent = _run_parent(tmp_path, steps=15, seed=7)

    branch_at = 8
    parent_snap = json.loads(
        (runs_dir / "parent" / "snapshots.jsonl").read_text(encoding="utf-8").splitlines()[branch_at - 1]
    )

    reset_all_counters()
    runner, fork_meta = build_fork(
        ForkConfig(parent_run_id="parent", branch_at_step=branch_at,
                   new_run_id="fork_branch", extend_by=1),
        runs_dir=runs_dir,
    )
    # World is reset to parent's snapshot. Check key fields before .run() advances it.
    assert runner.world.state.timestep == branch_at
    assert runner.world.state.customer_sentiment == pytest.approx(parent_snap["customer_sentiment"])
    assert runner.world.state.refund_policy_version == parent_snap["refund_policy_version"]
    assert set(runner.world.state.open_cases.keys()) == set(parent_snap["open_cases"].keys())
    runner.logger.close()


def test_warm_memory_replays_parent_history(tmp_path):
    """After forking, each agent's memory should contain entries up to branch_at_step."""
    runs_dir, _ = _run_parent(tmp_path, steps=12, seed=5)

    reset_all_counters()
    runner, _ = build_fork(
        ForkConfig(parent_run_id="parent", branch_at_step=6,
                   new_run_id="fork_mem", extend_by=2),
        runs_dir=runs_dir,
    )
    # Memory cap is 32 by default; replaying 6 steps × 2 entries/step = 12 entries.
    for agent in runner.agents:
        assert len(agent.memory.log) > 0, f"{agent.name} memory should be rehydrated"
        timesteps = {e.timestep for e in agent.memory.log}
        assert max(timesteps) == 6, f"{agent.name} memory should include step 6"
    runner.logger.close()


def test_disabled_agent_stops_acting_in_branch(tmp_path):
    """Disabled agents emit no actions after the branch point."""
    runs_dir, _ = _run_parent(tmp_path, steps=10, seed=3)

    reset_all_counters()
    runner, _ = build_fork(
        ForkConfig(
            parent_run_id="parent", branch_at_step=4,
            new_run_id="fork_disable",
            overrides=ForkOverrides(disabled_agents={"refund"}),
            extend_by=6,
        ),
        runs_dir=runs_dir,
    )
    fork_result = asyncio.run(runner.run())
    runner.logger.close()
    # No refund actions should appear in the fork's own action log.
    refund_actions = [a for a in fork_result.actions if a.agent_name == "refund"]
    assert refund_actions == [], "disabled agent produced actions"


def test_seed_override_diverges_action_stream(tmp_path):
    """Changing the seed at fork time should diverge subsequent decisions."""
    runs_dir, parent = _run_parent(tmp_path, steps=12, seed=11)

    reset_all_counters()
    runner, _ = build_fork(
        ForkConfig(
            parent_run_id="parent", branch_at_step=4,
            new_run_id="fork_seed",
            overrides=ForkOverrides(seed=9999),
            extend_by=8,
        ),
        runs_dir=runs_dir,
    )
    fork = asyncio.run(runner.run())
    runner.logger.close()

    parent_post = [(x.timestep, x.agent_name, x.kind, x.target_case_id)
                   for x in parent.actions if x.timestep > 4]
    fork_post = [(x.timestep, x.agent_name, x.kind, x.target_case_id)
                 for x in fork.actions]
    assert parent_post != fork_post, "seed override produced identical post-branch stream"


def test_fork_writes_lineage_metadata(tmp_path):
    """The fork's run_meta.json should record parent_run_id and branch_at_step."""
    runs_dir, _ = _run_parent(tmp_path, steps=8, seed=1)

    reset_all_counters()
    runner, _ = build_fork(
        ForkConfig(
            parent_run_id="parent", branch_at_step=3,
            new_run_id="fork_meta",
            overrides=ForkOverrides(prompt_variants={"refund": "hardened"}),
            extend_by=5,
        ),
        runs_dir=runs_dir,
    )
    asyncio.run(runner.run())
    runner.logger.close()

    meta = json.loads((runs_dir / "fork_meta" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["parent_run_id"] == "parent"
    assert meta["branch_at_step"] == 3
    assert meta["fork_overrides"]["prompt_variants"] == {"refund": "hardened"}


def test_fork_rejects_out_of_range_step(tmp_path):
    runs_dir, _ = _run_parent(tmp_path, steps=5, seed=1)
    with pytest.raises(ValueError):
        build_fork(
            ForkConfig(parent_run_id="parent", branch_at_step=99,
                       new_run_id="fork_bad"),
            runs_dir=runs_dir,
        )


def test_fork_rejects_duplicate_run_id(tmp_path):
    runs_dir, _ = _run_parent(tmp_path, steps=5, seed=1, run_id="parent")
    with pytest.raises(FileExistsError):
        build_fork(
            ForkConfig(parent_run_id="parent", branch_at_step=2,
                       new_run_id="parent"),  # collides with parent
            runs_dir=runs_dir,
        )
