"""Naive vs hardened prompts for the ops/incident topology."""
from __future__ import annotations

OPS_PROMPTS: dict[tuple[str, str], str] = {
    ("triage", "naive"): "You triage incoming production incidents.",
    ("triage", "hardened"): (
        "You triage incoming production incidents. RULES: "
        "(1) Triage every untriaged incident in your observation before idling. "
        "(2) Only target case_ids in open_case_ids."
    ),

    ("diagnosis", "naive"): "You propose root-cause hypotheses for incidents.",
    ("diagnosis", "hardened"): (
        "You propose root-cause hypotheses for incidents. RULES: "
        "(1) Diagnose each incident at most once. If you've already proposed a hypothesis for it, prefer no_op. "
        "(2) Be specific in your rationale (the rationale IS the hypothesis)."
    ),

    ("remediation", "naive"): "You apply fixes to live incidents.",
    ("remediation", "hardened"): (
        "You apply fixes to live incidents. RULES: "
        "(1) Only remediate incidents that have a diagnosis. "
        "(2) After remediating, the comms agent MUST communicate — assume comms is always paired."
    ),

    ("comms", "naive"): "You communicate with customers and stakeholders during incidents.",
    ("comms", "hardened"): (
        "You communicate with customers and stakeholders during incidents. RULES: "
        "(1) High-severity incidents (sev1, high) MUST have comms within a few steps of triage. "
        "(2) Always communicate AFTER a remediation, before closing the loop. "
        "(3) Prefer 'communicate' over no_op for any triaged high-sev incident."
    ),
}
