"""Ops topology detectors."""
from __future__ import annotations

from collections import defaultdict

from drift.failures.base import DetectorContext, FailureRecord
from drift.failures.detectors import _maybe_report

# How many steps after remediation before lack of comms = silent_remediation.
SILENT_REMEDIATION_LAG = 3
# How many steps after triage of a high-severity incident with no comms = comms_lag.
COMMS_LAG_STEPS = 5


def detect_silent_remediation(ctx: DetectorContext) -> list[FailureRecord]:
    """A remediation action shipped, but no communicate action followed within
    SILENT_REMEDIATION_LAG steps. The incident was fixed silently — customers
    don't know. Pages happen, trust erodes."""
    by_inc: dict[str, list] = defaultdict(list)
    for a in ctx.actions:
        if a.target_case_id:
            by_inc[a.target_case_id].append(a)

    out: list[FailureRecord] = []
    for inc_id, actions in by_inc.items():
        actions.sort(key=lambda x: (x.timestep, x.action_id))
        rem_at = None
        for a in actions:
            if a.kind == "remediate":
                rem_at = a.timestep
            elif a.kind == "communicate" and rem_at is not None:
                rem_at = None  # comms followed; clear
        if rem_at is not None and (ctx.timestep - rem_at) >= SILENT_REMEDIATION_LAG:
            fp = f"silent_remediation:{inc_id}:{rem_at}"
            out += _maybe_report(
                ctx, fp,
                FailureRecord(
                    timestep=ctx.timestep,
                    failure_type="silent_remediation",
                    agents_involved=["remediation", "comms"],
                    evidence_action_ids=[a.action_id for a in actions if a.kind == "remediate"],
                    summary=f"incident {inc_id} fixed at t={rem_at}, no comms by t={ctx.timestep}",
                    snapshot_timestep=ctx.timestep,
                ),
            )
    return out


def detect_comms_lag(ctx: DetectorContext) -> list[FailureRecord]:
    """A high-severity incident was triaged but no comms went out within
    COMMS_LAG_STEPS — that's a status-page failure waiting to happen."""
    snap = ctx.history.latest()
    if snap is None:
        return []

    by_inc: dict[str, list] = defaultdict(list)
    for a in ctx.actions:
        if a.target_case_id:
            by_inc[a.target_case_id].append(a)

    out: list[FailureRecord] = []
    # Walk both currently-open and previously-known incidents.
    known = set()
    for s in ctx.history.all_snapshots():
        known.update(s.open_cases.keys())

    for inc_id in known:
        actions = sorted(by_inc.get(inc_id, []), key=lambda x: (x.timestep, x.action_id))
        triaged_at = next((a.timestep for a in actions if a.kind == "triage"), None)
        if triaged_at is None:
            continue
        # find first comms after triage
        comms_at = next((a.timestep for a in actions if a.kind == "communicate" and a.timestep >= triaged_at), None)

        # Only fire on high-sev incidents; check the latest snapshot we know it from
        sev = None
        for s in ctx.history.all_snapshots():
            if inc_id in s.open_cases:
                sev = s.open_cases[inc_id].extra.get("severity")
                break
        if sev not in ("high", "sev1"):
            continue

        if comms_at is None and (ctx.timestep - triaged_at) >= COMMS_LAG_STEPS:
            fp = f"comms_lag:{inc_id}:{triaged_at}"
            out += _maybe_report(
                ctx, fp,
                FailureRecord(
                    timestep=ctx.timestep,
                    failure_type="comms_lag",
                    agents_involved=["triage", "comms"],
                    evidence_action_ids=[],
                    summary=f"sev={sev} incident {inc_id} triaged at t={triaged_at}, no comms in {ctx.timestep - triaged_at} steps",
                    snapshot_timestep=ctx.timestep,
                ),
            )
    return out


def detect_contradictory_diagnosis(ctx: DetectorContext) -> list[FailureRecord]:
    """The same incident received two different diagnoses. The DiagnosisAgent
    is supposed to be authoritative — distinct rationales on the same incident
    are a sign the agent is not maintaining state."""
    by_inc: dict[str, list[str]] = defaultdict(list)
    by_inc_actions: dict[str, list[str]] = defaultdict(list)
    for a in ctx.actions:
        if a.kind == "diagnose" and a.target_case_id:
            by_inc[a.target_case_id].append(a.rationale[:40])
            by_inc_actions[a.target_case_id].append(a.action_id)

    out: list[FailureRecord] = []
    for inc_id, rationales in by_inc.items():
        if len(set(rationales)) > 1:
            fp = f"contradictory_diagnosis:{inc_id}"
            out += _maybe_report(
                ctx, fp,
                FailureRecord(
                    timestep=ctx.timestep,
                    failure_type="contradictory_diagnosis",
                    agents_involved=["diagnosis"],
                    evidence_action_ids=by_inc_actions[inc_id],
                    summary=f"incident {inc_id} has {len(set(rationales))} distinct root-cause hypotheses",
                    snapshot_timestep=ctx.timestep,
                ),
            )
    return out


OPS_DETECTORS = [
    detect_silent_remediation,
    detect_comms_lag,
    detect_contradictory_diagnosis,
]
