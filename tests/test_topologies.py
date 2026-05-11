"""Per-topology smoke tests.

Each topology should: (1) run end-to-end without errors on its scenario,
(2) produce at least one of its domain-specific failures, and (3) produce
deterministic output across two runs with the same seed.
"""
import asyncio
from pathlib import Path

import pytest

from drift.events.scheduler import EventScheduler
from drift.llm.mock import ScriptedMockLLM
from drift.observability.logger import RunLogger
from drift.simulation import SimulationRunner
from drift.testing import reset_all_counters
from drift.topologies import get_topology, list_topologies


REPO = Path(__file__).resolve().parents[1]


def _build(topology_name: str, scenario_path: Path | None, steps: int = 30, seed: int = 42, log_dir: Path | None = None):
    reset_all_counters()  # determinism: IDs start fresh per run
    topology = get_topology(topology_name)
    llm = ScriptedMockLLM(seed=seed, role_handlers=topology.mock_handlers)
    agents = topology.agent_factory(llm)
    sched = (
        EventScheduler.from_yaml(scenario_path, seed=seed, event_registry=topology.event_registry)
        if scenario_path
        else EventScheduler.empty(seed=seed, event_registry=topology.event_registry)
    )
    logger = RunLogger(base_dir=log_dir) if log_dir else None
    runner = SimulationRunner(
        agents=agents,
        scheduler=sched,
        steps=steps,
        detectors=topology.detectors,
        logger=logger,
        initial_world=topology.initial_world(),
    )
    return runner, logger


def test_topologies_registered():
    names = list_topologies()
    assert "support" in names
    assert "code_review" in names
    assert "ops" in names


@pytest.mark.parametrize(
    "topology,scenario,expected_types",
    [
        ("support",     "scenarios/policy_chaos.yaml",
            {"policy_inconsistency", "sentiment_collapse"}),
        ("code_review", "scenarios/release_pressure.yaml",
            {"contradictory_review", "merge_without_approval"}),
        ("ops",         "scenarios/ops_storm.yaml",
            {"contradictory_diagnosis", "comms_lag"}),
    ],
)
def test_topology_smoke_emits_expected_failures(topology, scenario, expected_types, tmp_path):
    runner, logger = _build(topology, REPO / scenario, steps=40, seed=42, log_dir=tmp_path)
    try:
        result = asyncio.run(runner.run())
    finally:
        if logger:
            logger.close()
    assert result.final_state.timestep == 40
    types_seen = {f.failure_type for f in result.failures}
    overlap = expected_types & types_seen
    # Each topology must demonstrably fire at least one of its named failures —
    # the demo would be hollow otherwise.
    assert overlap, (
        f"topology={topology}: expected to see at least one of {expected_types}, "
        f"got {types_seen}"
    )


@pytest.mark.parametrize("topology", ["support", "code_review", "ops"])
def test_topology_seed_determinism(topology):
    scenario_map = {
        "support":     "scenarios/policy_chaos.yaml",
        "code_review": "scenarios/release_pressure.yaml",
        "ops":         "scenarios/ops_storm.yaml",
    }
    s = REPO / scenario_map[topology]

    # Reset counters *immediately* before each run, since IDs are minted
    # during run() not during _build().
    reset_all_counters()
    a, _ = _build(topology, s, steps=20, seed=123, log_dir=None)
    ra = asyncio.run(a.run())

    reset_all_counters()
    b, _ = _build(topology, s, steps=20, seed=123, log_dir=None)
    rb = asyncio.run(b.run())

    a_keys = [(x.timestep, x.agent_name, x.kind, x.target_case_id) for x in ra.actions]
    b_keys = [(x.timestep, x.agent_name, x.kind, x.target_case_id) for x in rb.actions]
    assert a_keys == b_keys
