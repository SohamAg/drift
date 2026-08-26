"""Target registry — real LangGraph multi-agent systems drift is tested against.

Each target module exposes a uniform interface so a single harness can
enumerate, build, and run drift against all of them without special-casing:

    NAME          str   — registry key
    DESCRIPTION   str   — one line, what the system is
    TOPOLOGY      str   — coordination shape (see TOPOLOGIES below)
    SOURCE        str   — where it comes from (official lib, docs, custom)
    REQUIRES      list  — importable module names it needs
    NEEDS_API_KEY bool  — whether building/running calls a real LLM
    VISIBLE_TO_DRIFT bool — whether agents are graph NODES (True) or hidden
                            inside tool calls (False). See the deepagents
                            finding: drift observes graph super-steps, so
                            tool-call-dispatched sub-agents are invisible.

    build(model=...) -> compiled graph
    initial_state(query) -> dict
    DEFAULT_QUERY str

Why a registry rather than one script per system: the interesting question
isn't "does drift run on system X," it's "what does drift find across
coordination topologies." That comparison needs the targets to be
enumerable and uniformly invokable.
"""
from __future__ import annotations

import importlib
from typing import Any

#: Coordination shapes we want coverage across. A target's TOPOLOGY must be
#: one of these so the harness can group findings by shape.
TOPOLOGIES = (
    "supervisor",       # central router delegating to specialists
    "swarm",            # peer-to-peer handoff, no central router
    "hierarchical",     # supervisor of supervisors (nested teams)
    "pipeline",         # fixed sequential stages
    "loop",             # two nodes cycling until a condition (generator/critic)
    "fanout",           # one node spawning N parallel workers that merge
    "tool_dispatch",    # sub-agents hidden inside tool calls (drift-blind)
)

#: Module names under examples.targets, in registry order.
_TARGET_MODULES = (
    "supervisor_basic",
    "supervisor_extended",
    "swarm_peer",
    "hierarchical_teams",
    "collaboration_pair",
    "plan_execute",
    "reflection_loop",
    "map_reduce_fanout",
    "hitl_interrupt",
    "deepagents_tooldispatch",
)


def _load(name: str):
    return importlib.import_module(f"{__name__}.{name}")


def all_targets() -> list[Any]:
    """Import every target module. Raises if a module itself is broken."""
    return [_load(n) for n in _TARGET_MODULES]


def get(name: str):
    """Look up one target module by its NAME (not its module path)."""
    for mod in all_targets():
        if mod.NAME == name:
            return mod
    raise KeyError(f"no target named {name!r}; have {[m.NAME for m in all_targets()]}")


def describe(mod: Any) -> dict[str, Any]:
    """Registry metadata for one target, plus a live dependency check."""
    missing = []
    for dep in mod.REQUIRES:
        try:
            importlib.import_module(dep)
        except ImportError:
            missing.append(dep)
    return {
        "name": mod.NAME,
        "description": mod.DESCRIPTION,
        "topology": mod.TOPOLOGY,
        "source": mod.SOURCE,
        "requires": list(mod.REQUIRES),
        "missing_deps": missing,
        "needs_api_key": mod.NEEDS_API_KEY,
        "visible_to_drift": mod.VISIBLE_TO_DRIFT,
        "default_query": mod.DEFAULT_QUERY,
        "importable": not missing,
    }


def registry() -> list[dict[str, Any]]:
    """Full registry with dependency status for every target."""
    return [describe(m) for m in all_targets()]


__all__ = ["TOPOLOGIES", "all_targets", "describe", "get", "registry"]
