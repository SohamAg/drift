"""Registry-integrity tests for examples/targets.

These do NOT build graphs or call an LLM — they check that the registry
stays internally coherent as targets are added. The expensive check (every
target actually compiles) lives in examples/targets/verify_targets.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))

from targets import TOPOLOGIES, all_targets, describe, get, registry  # noqa: E402

REQUIRED_ATTRS = (
    "NAME", "DESCRIPTION", "TOPOLOGY", "SOURCE", "REQUIRES",
    "NEEDS_API_KEY", "VISIBLE_TO_DRIFT", "DEFAULT_QUERY",
)


def test_every_target_imports() -> None:
    mods = all_targets()
    assert len(mods) == 4


@pytest.mark.parametrize("mod", all_targets(), ids=lambda m: m.NAME)
def test_target_contract(mod) -> None:
    """Each target exposes the full metadata + callable contract."""
    for attr in REQUIRED_ATTRS:
        assert hasattr(mod, attr), f"{mod.__name__} missing {attr}"
    assert callable(mod.build)
    assert callable(mod.initial_state)
    assert mod.TOPOLOGY in TOPOLOGIES, f"{mod.NAME}: unknown topology {mod.TOPOLOGY}"
    assert isinstance(mod.REQUIRES, list) and mod.REQUIRES
    assert mod.DESCRIPTION.strip()
    assert mod.DEFAULT_QUERY.strip()


def test_names_unique() -> None:
    names = [m.NAME for m in all_targets()]
    assert len(names) == len(set(names))


def test_initial_state_is_dict_without_llm() -> None:
    """initial_state must be constructible without building the graph."""
    for mod in all_targets():
        st = mod.initial_state()
        assert isinstance(st, dict) and st, f"{mod.NAME}: empty initial_state"


def test_topology_coverage() -> None:
    """The registry's point is breadth — guard against it collapsing to one shape.

    Bar is 4 today (supervisor, swarm, tool_dispatch, hierarchical) because
    every target must be genuine third-party code (docs/targets_setup.md
    hard rule) — raise this as real third-party targets are added, don't
    pad it with in-repo constructions to hit a bigger number.
    """
    topos = {m.TOPOLOGY for m in all_targets()}
    assert len(topos) >= 4, f"only {len(topos)} topologies covered: {topos}"
    assert "supervisor" in topos
    assert "swarm" in topos


def test_blind_spot_target_present() -> None:
    """The deepagents negative control must stay in the registry.

    If this disappears, drift's tool-call observation boundary stops being
    re-confirmed on every harness run.
    """
    blind = [m.NAME for m in all_targets() if not m.VISIBLE_TO_DRIFT]
    assert blind == ["deepagents_tooldispatch"]


def test_describe_and_get() -> None:
    d = describe(get("swarm_peer"))
    assert d["topology"] == "swarm"
    assert d["visible_to_drift"] is True
    assert "langgraph_swarm" in d["requires"]
    assert isinstance(d["missing_deps"], list)


def test_get_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get("does_not_exist")


def test_registry_shape() -> None:
    rows = registry()
    assert len(rows) == 4
    assert all("importable" in r for r in rows)


def test_registry_module_path_is_importable():
    """`module` must be the real importable path (mod.__name__), not an
    assumption that NAME matches the file's basename — used by the server
    to build()/initial_state() a target by module path."""
    import importlib

    for row in registry():
        assert row["module"].startswith("targets.")
        mod = importlib.import_module(row["module"])
        assert mod.NAME == row["name"]
