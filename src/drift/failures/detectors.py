"""The six emergent-failure detectors.

Each is a pure function over (history, actions, events). To avoid
re-firing the same failure on every subsequent step, each detector
fingerprints what it has reported and the runner stores fingerprints
in `DetectorContext.already_reported`.
"""
from __future__ import annotations

from drift.failures.base import DetectorContext, FailureRecord


# Tuning knobs — kept here so the simulation's sensitivity is in one place.
SENTIMENT_COLLAPSE_THRESHOLD = 0.25
SENTIMENT_COLLAPSE_WINDOW = 5
ESCALATION_LOOP_THRESHOLD = 3
QUEUE_EXPLOSION_WINDOW = 5
QUEUE_EXPLOSION_GROWTH = 4  # net +N over the window


def _maybe_report(ctx: DetectorContext, fingerprint: str, record: FailureRecord) -> list[FailureRecord]:
    if fingerprint in ctx.already_reported:
        return []
    ctx.already_reported.add(fingerprint)
    return [record]


def detect_contradictory_refunds(ctx: DetectorContext) -> list[FailureRecord]:
    """Same case_id received both approve and deny, ever."""
    by_case: dict[str, set[str]] = {}
    by_case_actions: dict[str, list[str]] = {}
    by_case_agents: dict[str, set[str]] = {}
    for a in ctx.actions:
        if a.kind in ("refund_approve", "refund_deny") and a.target_case_id:
            by_case.setdefault(a.target_case_id, set()).add(a.kind)
            by_case_actions.setdefault(a.target_case_id, []).append(a.action_id)
            by_case_agents.setdefault(a.target_case_id, set()).add(a.agent_name)

    out: list[FailureRecord] = []
    for case_id, kinds in by_case.items():
        if {"refund_approve", "refund_deny"}.issubset(kinds):
            fp = f"contradictory_refund:{case_id}"
            out += _maybe_report(
                ctx, fp,
                FailureRecord(
                    timestep=ctx.timestep,
                    failure_type="contradictory_refund",
                    agents_involved=sorted(by_case_agents[case_id]),
                    evidence_action_ids=by_case_actions[case_id],
                    summary=f"case {case_id} got both approve and deny",
                    snapshot_timestep=ctx.timestep,
                ),
            )
    return out


def detect_escalation_loop(ctx: DetectorContext) -> list[FailureRecord]:
    """Same case appears in escalation queue more than ESCALATION_LOOP_THRESHOLD times."""
    snap = ctx.history.latest()
    if snap is None:
        return []
    out: list[FailureRecord] = []
    for case_id, case in snap.open_cases.items():
        if case.escalation_count > ESCALATION_LOOP_THRESHOLD:
            fp = f"escalation_loop:{case_id}:{case.escalation_count}"
            evidence = [a.action_id for a in ctx.actions if a.target_case_id == case_id]
            agents = sorted({a.agent_name for a in ctx.actions if a.target_case_id == case_id})
            out += _maybe_report(
                ctx, fp,
                FailureRecord(
                    timestep=ctx.timestep,
                    failure_type="escalation_loop",
                    agents_involved=agents,
                    evidence_action_ids=evidence,
                    summary=f"case {case_id} escalated {case.escalation_count} times",
                    snapshot_timestep=snap.timestep,
                ),
            )
    return out


def detect_policy_inconsistency(ctx: DetectorContext) -> list[FailureRecord]:
    """Action references a policy version that disagrees with the current world."""
    snap = ctx.history.latest()
    if snap is None:
        return []
    current_v = snap.refund_policy_version
    out: list[FailureRecord] = []
    for a in ctx.actions:
        if a.referenced_policy_version is not None and a.referenced_policy_version != current_v:
            fp = f"policy_inconsistency:{a.action_id}"
            out += _maybe_report(
                ctx, fp,
                FailureRecord(
                    timestep=ctx.timestep,
                    failure_type="policy_inconsistency",
                    agents_involved=[a.agent_name],
                    evidence_action_ids=[a.action_id],
                    summary=f"{a.agent_name} cited policy v{a.referenced_policy_version}; world is v{current_v}",
                    snapshot_timestep=snap.timestep,
                ),
            )
    return out


def detect_sentiment_collapse(ctx: DetectorContext) -> list[FailureRecord]:
    """customer_sentiment below threshold for K consecutive snapshots."""
    window = ctx.history.window(SENTIMENT_COLLAPSE_WINDOW)
    if len(window) < SENTIMENT_COLLAPSE_WINDOW:
        return []
    if all(s.customer_sentiment < SENTIMENT_COLLAPSE_THRESHOLD for s in window):
        first = window[0].timestep
        fp = f"sentiment_collapse:{first}"
        return _maybe_report(
            ctx, fp,
            FailureRecord(
                timestep=ctx.timestep,
                failure_type="sentiment_collapse",
                agents_involved=[],
                evidence_action_ids=[],
                summary=(
                    f"sentiment < {SENTIMENT_COLLAPSE_THRESHOLD} "
                    f"for {SENTIMENT_COLLAPSE_WINDOW} steps starting at t={first}"
                ),
                snapshot_timestep=window[-1].timestep,
            ),
        )
    return []


def _all_known_case_ids(ctx: DetectorContext) -> set[str]:
    """Every case_id that has appeared in any snapshot, ever."""
    seen: set[str] = set()
    for snap in ctx.history.all_snapshots():
        seen.update(snap.open_cases.keys())
    return seen


def detect_hallucinated_reference(ctx: DetectorContext) -> list[FailureRecord]:
    """Agent invented a case_id that has never existed in the world.

    Distinct from `stale_snapshot_reference`: this is a true fabrication.
    """
    snap = ctx.history.latest()
    if snap is None:
        return []
    known = _all_known_case_ids(ctx)
    out: list[FailureRecord] = []
    for a in ctx.actions:
        if a.kind in ("no_op", "policy_update"):
            continue
        if a.target_case_id and a.target_case_id not in known:
            fp = f"hallucinated_ref:{a.action_id}"
            out += _maybe_report(
                ctx, fp,
                FailureRecord(
                    timestep=ctx.timestep,
                    failure_type="hallucinated_reference",
                    agents_involved=[a.agent_name],
                    evidence_action_ids=[a.action_id],
                    summary=f"{a.agent_name} fabricated case_id {a.target_case_id}",
                    snapshot_timestep=snap.timestep,
                ),
            )
    return out


def detect_stale_snapshot_reference(ctx: DetectorContext) -> list[FailureRecord]:
    """Agent referenced a case that was open at observation time but
    has since been removed by another agent acting earlier in the
    same step's ordering. This is a coordination failure, not a
    hallucination — distinguishing it matters because the fix is
    different (better intra-step messaging, not better grounding).
    """
    snap = ctx.history.latest()
    if snap is None:
        return []
    known = _all_known_case_ids(ctx)
    open_ids = set(snap.open_cases.keys())
    out: list[FailureRecord] = []
    for a in ctx.actions:
        if a.kind in ("no_op", "policy_update"):
            continue
        cid = a.target_case_id
        if cid and cid in known and cid not in open_ids:
            fp = f"stale_ref:{a.action_id}"
            out += _maybe_report(
                ctx, fp,
                FailureRecord(
                    timestep=ctx.timestep,
                    failure_type="stale_snapshot_reference",
                    agents_involved=[a.agent_name],
                    evidence_action_ids=[a.action_id],
                    summary=f"{a.agent_name} acted on {cid} after it was already removed",
                    snapshot_timestep=snap.timestep,
                ),
            )
    return out


def detect_queue_explosion(ctx: DetectorContext) -> list[FailureRecord]:
    """Escalation queue grew by > threshold over a rolling window."""
    window = ctx.history.window(QUEUE_EXPLOSION_WINDOW)
    if len(window) < QUEUE_EXPLOSION_WINDOW:
        return []
    growth = len(window[-1].escalation_queue) - len(window[0].escalation_queue)
    if growth >= QUEUE_EXPLOSION_GROWTH:
        first = window[0].timestep
        fp = f"queue_explosion:{first}:{growth}"
        return _maybe_report(
            ctx, fp,
            FailureRecord(
                timestep=ctx.timestep,
                failure_type="queue_explosion",
                agents_involved=[],
                evidence_action_ids=[],
                summary=f"queue grew by {growth} over {QUEUE_EXPLOSION_WINDOW} steps",
                snapshot_timestep=window[-1].timestep,
            ),
        )
    return []


ALL_DETECTORS = [
    detect_contradictory_refunds,
    detect_escalation_loop,
    detect_policy_inconsistency,
    detect_sentiment_collapse,
    detect_hallucinated_reference,
    detect_stale_snapshot_reference,
    detect_queue_explosion,
]

# Detectors that work on any topology (not domain-specific).
GENERAL_DETECTORS = [
    detect_sentiment_collapse,
    detect_hallucinated_reference,
    detect_stale_snapshot_reference,
    detect_queue_explosion,
]

# Detectors specific to the customer-support topology.
SUPPORT_DETECTORS = [
    detect_contradictory_refunds,
    detect_escalation_loop,
    detect_policy_inconsistency,
]
