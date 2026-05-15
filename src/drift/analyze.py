"""Trace analyzer — run drift's failure detectors against an external trace.

Drift's value is the named taxonomy of multi-agent coordination failures and
the detector functions that catch them. This module lets you point that
detector library at an action log produced by *any* multi-agent system, not
just drift's own simulator.

The accepted trace format is the same JSONL shape drift already emits to
runs/<run_id>/. Two ways to call it:

  1. Directory mode — pass a directory containing:
       snapshots.jsonl  (one WorldState per timestep)
       actions.jsonl    (one Action per agent decision)
       events.jsonl     (optional; one EventRecord per exogenous event)

  2. Single-file mode — pass a .jsonl where every line has a "type" field
     of "snapshot", "action", or "event" plus the usual fields.

Anyone exporting their multi-agent system's behavior in this shape can run
the detectors over it without using drift's simulator at all. See
TRACE_SCHEMA.md for the field-level contract.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from drift.agents.base import Action
from drift.events.base import EventRecord
from drift.failures.base import DetectorContext, FailureRecord
from drift.topologies import Topology, get_topology
from drift.world import WorldHistory, WorldState


# --------------------------------------------------------------- loading --

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _split_by_type(records: Iterable[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Partition a mixed-record JSONL by its `type` field."""
    snapshots: list[dict] = []
    actions: list[dict] = []
    events: list[dict] = []
    for r in records:
        t = r.get("type")
        # Allow records to omit the "type" field if the file is type-pure;
        # caller is responsible in that case.
        if t == "snapshot":
            snapshots.append(_strip_type(r))
        elif t == "action":
            actions.append(_strip_type(r))
        elif t == "event":
            events.append(_strip_type(r))
        elif t is None:
            # Type-less records are skipped here; use directory mode for those.
            continue
        else:
            raise ValueError(f"unknown record type in trace: {t!r}")
    return snapshots, actions, events


def _strip_type(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "type"}


def load_trace(path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (snapshots, actions, events) raw-dict lists from a trace path.

    Accepts either a directory (drift native layout) or a single mixed JSONL.
    """
    path = Path(path)
    if path.is_dir():
        snapshots = _load_jsonl(path / "snapshots.jsonl")
        actions = _load_jsonl(path / "actions.jsonl")
        events = _load_jsonl(path / "events.jsonl")
        return snapshots, actions, events
    if path.is_file():
        return _split_by_type(_load_jsonl(path))
    raise FileNotFoundError(f"trace path not found: {path}")


# --------------------------------------------------------------- replay --

def analyze_records(
    snapshots_raw: list[dict],
    actions_raw: list[dict],
    events_raw: list[dict],
    topology_name: str,
) -> tuple[list[FailureRecord], dict]:
    """Replay in-memory trace records through detectors. Same semantics as
    analyze_trace, but accepts already-parsed dict lists (no file I/O).
    Useful for the web UI, tests, and any other in-memory caller.
    """
    return _analyze(snapshots_raw, actions_raw, events_raw, topology_name)


def analyze_trace(
    trace_path: Path,
    topology_name: str,
) -> tuple[list[FailureRecord], dict]:
    """Replay a trace through a topology's detectors. Returns (failures, summary).

    The detectors are invoked once per snapshot timestep, against the cumulative
    set of actions and events up to and including that step — matching what
    drift's SimulationRunner does live. `already_reported` dedupes across steps.
    """
    snap_raw, act_raw, evt_raw = load_trace(Path(trace_path))
    if not snap_raw:
        raise ValueError(
            f"trace at {trace_path} has no snapshots; cannot run detectors. "
            "At minimum, provide one snapshot record per timestep."
        )
    return _analyze(snap_raw, act_raw, evt_raw, topology_name)


def _analyze(
    snap_raw: list[dict],
    act_raw: list[dict],
    evt_raw: list[dict],
    topology_name: str,
) -> tuple[list[FailureRecord], dict]:
    topology: Topology = get_topology(topology_name)
    if not snap_raw:
        raise ValueError(
            "trace has no snapshots; cannot run detectors. "
            "At minimum, provide one snapshot record per timestep."
        )

    snapshots: list[WorldState] = [WorldState.model_validate(s) for s in snap_raw]
    actions: list[Action] = [Action.model_validate(a) for a in act_raw]
    events: list[EventRecord] = [EventRecord.model_validate(e) for e in evt_raw]

    # Group by timestep so we can replay cumulatively.
    snaps_by_step: dict[int, WorldState] = {s.timestep: s for s in snapshots}
    actions_by_step: dict[int, list[Action]] = defaultdict(list)
    for a in actions:
        actions_by_step[a.timestep].append(a)
    events_by_step: dict[int, list[EventRecord]] = defaultdict(list)
    for e in events:
        events_by_step[e.timestep].append(e)

    history = WorldHistory()
    cum_actions: list[Action] = []
    cum_events: list[EventRecord] = []
    reported: set[str] = set()
    failures: list[FailureRecord] = []

    sorted_steps = sorted(snaps_by_step.keys())
    for t in sorted_steps:
        history.record(snaps_by_step[t], [])
        cum_actions.extend(actions_by_step.get(t, []))
        cum_events.extend(events_by_step.get(t, []))

        ctx = DetectorContext(
            timestep=t,
            history=history,
            actions=cum_actions,
            events=cum_events,
            already_reported=reported,
        )
        for detector in topology.detectors:
            for failure in detector(ctx):
                failures.append(failure)

    summary = {
        "topology": topology_name,
        "n_snapshots": len(snapshots),
        "n_actions": len(actions),
        "n_events": len(events),
        "first_step": sorted_steps[0] if sorted_steps else None,
        "last_step": sorted_steps[-1] if sorted_steps else None,
        "failures_by_type": _count_by_type(failures),
    }
    return failures, summary


def _count_by_type(failures: list[FailureRecord]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for f in failures:
        counts[f.failure_type] += 1
    return dict(counts)
