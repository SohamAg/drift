import asyncio
from pathlib import Path

from drift.agents import EscalationAgent, PolicyAgent, RefundAgent, SupportAgent
from drift.events.scheduler import EventScheduler
from drift.llm import ScriptedMockLLM
from drift.observability.logger import RunLogger
from drift.simulation import SimulationRunner
from drift.world import World, WorldState


def _build(steps: int, seed: int, scenario: Path | None, log_dir: Path | None):
    llm = ScriptedMockLLM(seed=seed)
    agents = [
        SupportAgent(name="support", llm=llm),
        RefundAgent(name="refund", llm=llm),
        EscalationAgent(name="escalation", llm=llm),
        PolicyAgent(name="policy", llm=llm),
    ]
    sched = EventScheduler.from_yaml(scenario, seed=seed) if scenario else EventScheduler.empty(seed=seed)
    logger = RunLogger(base_dir=log_dir) if log_dir else None
    runner = SimulationRunner(
        agents=agents, scheduler=sched, steps=steps,
        logger=logger, initial_world=World(initial=WorldState()),
    )
    return runner, logger


def test_smoke_50_steps_writes_logs(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    scenario = repo_root / "scenarios" / "black_friday.yaml"
    runner, logger = _build(steps=50, seed=42, scenario=scenario, log_dir=tmp_path)
    try:
        result = asyncio.run(runner.run())
    finally:
        if logger:
            logger.close()
    assert result.final_state.timestep == 50
    run_dir = result.run_dir
    assert run_dir is not None
    for fname in ("events.jsonl", "actions.jsonl", "snapshots.jsonl", "failures.jsonl"):
        assert (run_dir / fname).exists()
    # Snapshots should be 50 lines (one per timestep).
    assert (run_dir / "snapshots.jsonl").read_text(encoding="utf-8").count("\n") == 50


def test_seed_determinism():
    """Two runs with the same seed produce identical action streams."""
    runner_a, _ = _build(steps=30, seed=7, scenario=None, log_dir=None)
    runner_b, _ = _build(steps=30, seed=7, scenario=None, log_dir=None)
    a = asyncio.run(runner_a.run())
    b = asyncio.run(runner_b.run())
    a_keys = [(x.timestep, x.agent_name, x.kind, x.target_case_id) for x in a.actions]
    b_keys = [(x.timestep, x.agent_name, x.kind, x.target_case_id) for x in b.actions]
    assert a_keys == b_keys


def test_emergent_failures_show_up():
    """With the black_friday scenario + seed 42, multiple failure types should fire."""
    repo_root = Path(__file__).resolve().parents[1]
    scenario = repo_root / "scenarios" / "black_friday.yaml"
    runner, _ = _build(steps=50, seed=42, scenario=scenario, log_dir=None)
    result = asyncio.run(runner.run())
    distinct_types = {f.failure_type for f in result.failures}
    # The whole point of the simulator: at least 3 distinct emergent failure
    # modes fire on the canonical scenario. If this drops, we've regressed
    # the mock's flakiness or detector sensitivity.
    assert len(distinct_types) >= 3, f"only saw: {distinct_types}"
