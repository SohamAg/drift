"""Detector: verifier-role agent approves everything.

Failure: an agent assigned to review/verify/audit produces approval-shaped
output at a near-100% rate with no rejections. The verifier isn't actually
verifying — it's rubber-stamping. MAST 3.x family ("Incorrect Verification",
"Lack of critical verification"). Anthropic's engineering blog cites the same
class as one of the harder bugs ("Minor failures compound catastrophically
in stateful tool calls").

Why it matters: most real failures the verifier *should* have caught get
written out as "approved." It silently disables the safety layer the system
was designed around.

This detector fires when:
  - At least `min_decisions` decisions came from a verifier-role agent
  - >= `approval_rate_threshold` of them were approvals (default 0.95)
  - Zero rejections observed
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from drift.failures.base import FailureRecord
from drift.failures.library.base import (
    CoordinationDetectorContext,
    TraceStep,
    any_token,
    lowercased_text_blob,
    role_matches,
)

NAME = "verifier_always_approves"
SUMMARY = "Verifier-role agent approved >=95% of decisions with no rejections."
MAST_MODES = ["3.3", "4.2", "4.3"]  # Incorrect Verification, Lack of result/critical verification
SOURCE = "MAST 3.x family; Anthropic multi-agent research postmortem"

# Default agent-name regex when no explicit roles_by_agent is supplied.
DEFAULT_VERIFIER_PATTERN = re.compile(r"verif|review|audit|critic|check|approv", re.I)

# Tokens we interpret as approval / rejection inside a STRUCTURED key value.
# "yes"/"no" are kept here because `verdict: "yes"` is a legitimate single-word
# decision in many graphs. The free-text scan uses a stricter subset below.
APPROVAL_TOKENS = (
    "approve", "approved", "approval",
    "accept", "accepted", "ok",
    "pass", "passed", "lgtm",
    "verified", "valid", "good",
    "yes",
)
REJECTION_TOKENS = (
    "reject", "rejected", "rejection",
    "deny", "denied",
    "fail", "failed", "failure",
    "invalid", "needs_changes", "needs-changes", "block", "blocked",
    "no",
)

# Free-text scans must avoid super-generic English words ("no", "yes") that
# show up in natural rationales ("no issues", "yes the test passes") and
# misclassify the line. We require an unambiguous verdict verb instead.
FREE_TEXT_APPROVAL_TOKENS = tuple(t for t in APPROVAL_TOKENS if t not in {"yes", "ok"})
FREE_TEXT_REJECTION_TOKENS = tuple(t for t in REJECTION_TOKENS if t not in {"no"})

# Common dict keys agents write decisions into. Used as the first signal —
# if any of these are present, we trust their value over free-text.
DECISION_KEYS = (
    "verdict", "decision", "approved", "approval", "status",
    "result", "outcome", "vote", "review", "assessment",
)


@dataclass
class _Verdict:
    """One classified decision from a verifier step."""

    step: int
    label: str  # "approve" | "reject" | "unknown"
    evidence: str  # the matched token or key/value snippet


def _classify(step: TraceStep) -> _Verdict | None:
    """Return None if the step doesn't look like a decision at all.

    Returns a verdict if either a structured decision-key has a recognizable
    value, OR the rationale/update text contains an unambiguous token.
    """
    # 1. Structured decision keys win.
    for k in DECISION_KEYS:
        if k in step.update:
            v = step.update[k]
            if isinstance(v, bool):
                return _Verdict(
                    step.step,
                    "approve" if v else "reject",
                    f"{k}={v}",
                )
            if isinstance(v, str):
                vlow = v.lower().strip()
                if any_token(vlow, APPROVAL_TOKENS):
                    return _Verdict(step.step, "approve", f"{k}={v!r}")
                if any_token(vlow, REJECTION_TOKENS):
                    return _Verdict(step.step, "reject", f"{k}={v!r}")
                # Unrecognized verdict value — treat as not-a-decision.
                return _Verdict(step.step, "unknown", f"{k}={v!r}")

    # 2. Free-text fallback. Use the stricter token set so common English
    # words ("no issues", "yes the test passes") don't cause spurious
    # rejection matches. Still require EXCLUSIVE presence — when both an
    # approval and a rejection verb appear, the line is ambiguous and we
    # decline to classify rather than guess.
    blob = lowercased_text_blob(step.update) + " " + (step.rationale or "").lower()
    has_approve = any_token(blob, FREE_TEXT_APPROVAL_TOKENS)
    has_reject = any_token(blob, FREE_TEXT_REJECTION_TOKENS)
    if has_approve and not has_reject:
        return _Verdict(step.step, "approve", f"text: {has_approve!r}")
    if has_reject and not has_approve:
        return _Verdict(step.step, "reject", f"text: {has_reject!r}")
    return None


def detect(
    ctx: CoordinationDetectorContext,
    *,
    min_decisions: int = 3,
    approval_rate_threshold: float = 0.95,
    verifier_name_pattern: re.Pattern[str] | None = None,
) -> list[FailureRecord]:
    """Fire when a verifier-role agent's approval rate >= threshold across
    >= min_decisions decisions with zero rejections.

    Per-agent — multiple verifier agents each get evaluated independently.
    """
    pattern = verifier_name_pattern or DEFAULT_VERIFIER_PATTERN
    out: list[FailureRecord] = []

    for agent, agent_steps in ctx.steps_by_agent().items():
        if not role_matches(agent, ctx.roles_by_agent, "verifier", pattern):
            continue
        verdicts: list[_Verdict] = []
        for s in agent_steps:
            v = _classify(s)
            if v is not None and v.label != "unknown":
                verdicts.append(v)
        if len(verdicts) < min_decisions:
            continue
        n_approve = sum(1 for v in verdicts if v.label == "approve")
        n_reject = sum(1 for v in verdicts if v.label == "reject")
        approval_rate = n_approve / len(verdicts)
        if n_reject == 0 and approval_rate >= approval_rate_threshold:
            last_step = verdicts[-1].step
            evidence_steps = ", ".join(f"step {v.step} ({v.evidence})" for v in verdicts[:4])
            out.append(FailureRecord(
                timestep=last_step,
                failure_type=NAME,
                agents_involved=[agent],
                evidence_action_ids=[],
                summary=(
                    f"verifier {agent!r} approved {n_approve}/{len(verdicts)} "
                    f"decisions ({approval_rate:.0%}) with 0 rejections — "
                    f"e.g. {evidence_steps}"
                ),
                snapshot_timestep=last_step,
            ))
    return out


# ---------------------------------------------------------------------------
# Raw-text variant — for MAST validation against unstructured transcripts.
# ---------------------------------------------------------------------------


# Captures lines that look like agent-attributed approvals: "Reviewer: LGTM",
# "<Verifier>: approved", "approve" appearing on a line that mentions a
# verify/review role. Conservative: must mention a verifier-shaped role on
# the SAME line as an approval token, with no rejection token anywhere on
# that line.
_TEXT_VERIFIER_LINE = re.compile(
    r"^(?P<line>.*\b(verif\w*|review\w*|audit\w*|critic\w*|approv\w*|check\w*)\b.*)$",
    re.I | re.M,
)


def detect_from_text(
    transcript: str,
    *,
    min_decisions: int = 3,
    approval_rate_threshold: float = 0.95,
) -> list[FailureRecord]:
    """Run the rule over a raw transcript. Lower precision than `detect`.

    Heuristic: for every line mentioning a verifier-shaped role, classify it
    as approve/reject by token presence. If >= min_decisions decisions and
    rate >= threshold with zero rejections, fire.

    This is what's used for the MAST F1 validation pass. We do NOT claim it
    matches `detect`'s precision/recall — text-only inference is strictly
    weaker. It exists so the structured detector has an apples-to-apples
    text-only baseline to demonstrate the lift structure adds.
    """
    if not transcript:
        return []
    counters: Counter[str] = Counter()
    sample_lines: list[str] = []
    for m in _TEXT_VERIFIER_LINE.finditer(transcript):
        line = m.group("line").strip().lower()
        # Need an explicit verdict token — skip pure description lines.
        has_approve = any_token(line, FREE_TEXT_APPROVAL_TOKENS)
        has_reject = any_token(line, FREE_TEXT_REJECTION_TOKENS)
        if has_approve and not has_reject:
            counters["approve"] += 1
            if len(sample_lines) < 4:
                sample_lines.append(line[:140])
        elif has_reject and not has_approve:
            counters["reject"] += 1
    total = counters["approve"] + counters["reject"]
    if total < min_decisions:
        return []
    rate = counters["approve"] / total
    if counters["reject"] == 0 and rate >= approval_rate_threshold:
        return [FailureRecord(
            timestep=0,
            failure_type=NAME,
            agents_involved=[],
            evidence_action_ids=[],
            summary=(
                f"text scan: {counters['approve']}/{total} verifier-shaped "
                f"lines approved ({rate:.0%}); examples: " + " | ".join(sample_lines)
            ),
            snapshot_timestep=0,
        )]
    return []


__all__ = [
    "DEFAULT_VERIFIER_PATTERN",
    "MAST_MODES",
    "NAME",
    "SOURCE",
    "SUMMARY",
    "detect",
    "detect_from_text",
]
