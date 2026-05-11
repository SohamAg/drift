"""Code-review topology detectors.

These read the action log per PR and decide whether the system as a whole
violated a rule. They run alongside the GENERAL_DETECTORS.
"""
from __future__ import annotations

from collections import defaultdict

from drift.failures.base import DetectorContext, FailureRecord
from drift.failures.detectors import _maybe_report


def detect_contradictory_review(ctx: DetectorContext) -> list[FailureRecord]:
    """Same PR got approve_review from one reviewer and reject_review from another.
    Indicates the review-policy prompt isn't anchoring the agents on consistent criteria.
    """
    by_pr_kinds: dict[str, set[str]] = defaultdict(set)
    by_pr_actions: dict[str, list[str]] = defaultdict(list)
    by_pr_agents: dict[str, set[str]] = defaultdict(set)
    for a in ctx.actions:
        if a.kind in ("approve_review", "reject_review") and a.target_case_id:
            by_pr_kinds[a.target_case_id].add(a.kind)
            by_pr_actions[a.target_case_id].append(a.action_id)
            by_pr_agents[a.target_case_id].add(a.agent_name)

    out: list[FailureRecord] = []
    for pr_id, kinds in by_pr_kinds.items():
        if {"approve_review", "reject_review"}.issubset(kinds):
            fp = f"contradictory_review:{pr_id}"
            out += _maybe_report(
                ctx, fp,
                FailureRecord(
                    timestep=ctx.timestep,
                    failure_type="contradictory_review",
                    agents_involved=sorted(by_pr_agents[pr_id]),
                    evidence_action_ids=by_pr_actions[pr_id],
                    summary=f"PR {pr_id} got both approve and reject from reviewers",
                    snapshot_timestep=ctx.timestep,
                ),
            )
    return out


def detect_security_bypass(ctx: DetectorContext) -> list[FailureRecord]:
    """A PR was merged while its security_status was 'blocked' (no subsequent clear)."""
    # Walk action log per PR in time order.
    by_pr: dict[str, list] = defaultdict(list)
    for a in ctx.actions:
        if a.target_case_id:
            by_pr[a.target_case_id].append(a)

    out: list[FailureRecord] = []
    for pr_id, actions in by_pr.items():
        actions.sort(key=lambda x: (x.timestep, x.action_id))
        blocked = False
        for a in actions:
            if a.kind == "security_block":
                blocked = True
            elif a.kind == "security_clear":
                blocked = False
            elif a.kind == "merge" and blocked:
                fp = f"security_bypass:{pr_id}:{a.action_id}"
                out += _maybe_report(
                    ctx, fp,
                    FailureRecord(
                        timestep=ctx.timestep,
                        failure_type="security_bypass",
                        agents_involved=sorted({x.agent_name for x in actions if x.kind in ("merge", "security_block", "security_clear")}),
                        evidence_action_ids=[a.action_id],
                        summary=f"PR {pr_id} merged while security had it blocked",
                        snapshot_timestep=ctx.timestep,
                    ),
                )
                break  # one bypass per PR is enough
    return out


def detect_merge_without_approval(ctx: DetectorContext) -> list[FailureRecord]:
    """A PR was merged with zero approvals in the action log preceding the merge."""
    by_pr: dict[str, list] = defaultdict(list)
    for a in ctx.actions:
        if a.target_case_id:
            by_pr[a.target_case_id].append(a)

    out: list[FailureRecord] = []
    for pr_id, actions in by_pr.items():
        actions.sort(key=lambda x: (x.timestep, x.action_id))
        approvals = 0
        for a in actions:
            if a.kind == "approve_review":
                approvals += 1
            elif a.kind == "merge":
                if approvals == 0:
                    fp = f"merge_no_approval:{pr_id}:{a.action_id}"
                    out += _maybe_report(
                        ctx, fp,
                        FailureRecord(
                            timestep=ctx.timestep,
                            failure_type="merge_without_approval",
                            agents_involved=[a.agent_name],
                            evidence_action_ids=[a.action_id],
                            summary=f"PR {pr_id} merged with zero approvals",
                            snapshot_timestep=ctx.timestep,
                        ),
                    )
                break  # only first merge matters
    return out


CODE_REVIEW_DETECTORS = [
    detect_contradictory_review,
    detect_security_bypass,
    detect_merge_without_approval,
]
