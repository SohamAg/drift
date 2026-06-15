"""MAST evaluation helpers — shared between the offline runner and the web demo.

The offline runner (`examples/adapters/run_drift_judge_on_mast.py`) builds
the prompt, calls the judge, and compares predictions against the human
labels in MAD_human_labelled_dataset.json. The web demo at /api/mast-analyze
needs the same logic. These functions live here so both paths share a
single source of truth — drift the binding the judge to a MAST trace,
parsing the response, and labeling each mode as TP/FP/FN/TN.

The functions here intentionally don't touch the analyze.py replay pipeline.
MAST traces are unstructured text per MAS framework; we send the raw
trajectory directly to the judge rather than parsing into drift actions.
"""
from __future__ import annotations

import json
import time
from typing import Any

from drift.failures.judge import JudgeLLM

# gpt-4o-mini fits ~128k tokens; we cap at 100k chars (~25k tokens) so per-call
# cost is predictable. Long traces lose the tail — judge sees the prefix.
MAX_TRACE_CHARS = 100_000


JUDGE_SYSTEM_TEMPLATE = """You analyze a multi-agent system's execution trace and report which \
failure modes from the MAST taxonomy occurred.

Failure modes for this trace (with definitions):

{mode_descriptions}

Be CONSERVATIVE. Only report a mode as occurring if you can cite specific evidence \
from the trace. If you cannot cite evidence, do not report it. False positives \
destroy this tool's value.

Output strict JSON only — no prose:
{{"failures": [{{"mode_id": "<exact mode ID, e.g. 1.5>", "evidence": "<one-sentence quote or paraphrase>"}}, ...]}}

If no modes apply: {{"failures": []}}"""


def parse_mode_id(mode_text: str) -> str:
    """Extract just the numbered ID from a mode label (e.g. '1.5')."""
    head = mode_text.split("\n")[0].strip()
    if not head:
        return ""
    parts = head.split(maxsplit=1)
    return parts[0] if parts else head


def mode_name(mode_text: str) -> str:
    """Human-readable name without the description body."""
    return mode_text.split("\n")[0].strip()


def majority_vote(ann: dict) -> bool:
    """True if ≥2 of 3 annotators marked this mode as occurring."""
    n_yes = sum(1 for k in ("annotator_1", "annotator_2", "annotator_3") if ann.get(k) is True)
    return n_yes >= 2


def annotator_agreement(ann: dict) -> tuple[int, int]:
    """Returns (n_yes, n_total) for this annotation."""
    keys = [k for k in ("annotator_1", "annotator_2", "annotator_3") if k in ann]
    n_yes = sum(1 for k in keys if ann.get(k) is True)
    return n_yes, len(keys)


def _render_mast_guidelines_block(guidelines: list[str]) -> str:
    """Render user guidelines for the MAST per-mode schema.

    Unlike the generic judge prompt (judge.render_user_guidelines_block) which
    introduces a new `user_guideline` family + `guideline_id` schema, the MAST
    runner uses a fixed per-trace mode_id schema. Guidelines here are *hints
    to bias detection toward existing modes*, not new output types. So we
    inject them as additional "ALSO consider" bullets that point back into
    the trace's existing mode vocabulary. No schema change.
    """
    cleaned = [g.strip() for g in guidelines if g and g.strip()]
    if not cleaned:
        return ""
    bullets = "\n".join(f"  - {g}" for g in cleaned)
    return (
        "\n\nUser-supplied detection hints — apply these IN ADDITION to the mode list above. "
        "If a hint matches the trace, report under the closest matching mode_id from the list above; "
        "do not invent new mode_ids. Hints that are anti-examples (\"do NOT flag X\") should "
        "tighten precision against the matching mode:\n"
        + bullets
    )


def build_system_prompt(
    annotations: list[dict],
    user_guidelines: list[str] | None = None,
) -> str:
    """Build a judge system prompt with the mode list specific to this trace.

    user_guidelines, if supplied, are appended as detection HINTS that the
    judge maps back onto the existing per-trace mode_id vocabulary — they
    do NOT introduce a new family. This preserves the MAST per-mode F1
    accounting and lets users measure the lift their domain knowledge adds
    over the generic prompt.
    """
    lines = []
    seen_ids: set[str] = set()
    for ann in annotations:
        mode_text = ann["failure mode"]
        mid = parse_mode_id(mode_text)
        if mid in seen_ids:
            continue
        seen_ids.add(mid)
        head = mode_name(mode_text)
        rest = mode_text[len(head):].strip()
        first_sentence = rest.split(".")[0].strip()[:180] if rest else ""
        lines.append(f"  - {head}: {first_sentence}")
    base = JUDGE_SYSTEM_TEMPLATE.format(mode_descriptions="\n".join(lines))
    return base + _render_mast_guidelines_block(list(user_guidelines or []))


def parse_predictions(raw: str) -> list[dict]:
    """Parse the judge's JSON response into [{mode_id, evidence}, ...]."""
    if not raw:
        return []
    text = raw.strip()
    if not text.startswith("{"):
        i = text.find("{")
        if i == -1:
            return []
        text = text[i:]
    if not text.endswith("}"):
        j = text.rfind("}")
        if j != -1:
            text = text[: j + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    raw_failures = payload.get("failures") or []
    out = []
    for item in raw_failures:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("mode_id") or "").strip()
        if not mid:
            continue
        out.append({"mode_id": mid, "evidence": str(item.get("evidence") or "")})
    return out


async def judge_one_trace(
    record: dict,
    judge: JudgeLLM,
    *,
    user_guidelines: list[str] | None = None,
) -> dict:
    """Run the judge against one MAST record and compute per-mode TP/FP/FN/TN.

    Same return shape as the offline runner's per-trace results. Used both
    by the offline runner and by the /api/mast-analyze live mode.

    user_guidelines, if supplied, are appended to the judge's system prompt
    as additional patterns to watch for. They don't affect per-mode TP/FP/FN/TN
    accounting (MAST ground truth is fixed) but they can lift recall when the
    judge's blind spots are domain-specific. Use to measure F1 delta vs the
    no-guideline baseline.
    """
    trajectory = record["trace"][:MAX_TRACE_CHARS]
    truncated = len(record["trace"]) > MAX_TRACE_CHARS
    annotations = record["annotations"]
    system = build_system_prompt(annotations, user_guidelines=user_guidelines)
    user = (
        f"# Multi-agent trace ({record['mas_name']}, {record['benchmark_name']}, "
        f"trace_id={record['trace_id']}, {len(record['trace'])} chars"
        + (" — TRUNCATED to first 100k" if truncated else "")
        + ")\n\n"
        + trajectory
    )
    t0 = time.perf_counter()
    try:
        raw = await judge.judge(system=system, user=user)
        latency = round(time.perf_counter() - t0, 2)
    except Exception as e:
        return {
            "trace_id": record["trace_id"],
            "mas_name": record["mas_name"],
            "error": f"{type(e).__name__}: {e}",
        }
    predictions = parse_predictions(raw)

    truth_by_id: dict[str, dict] = {}
    for ann in annotations:
        mid = parse_mode_id(ann["failure mode"])
        if mid in truth_by_id:
            truth_by_id[mid]["truth"] = truth_by_id[mid]["truth"] or majority_vote(ann)
        else:
            truth_by_id[mid] = {
                "name": mode_name(ann["failure mode"]),
                "truth": majority_vote(ann),
                "annotator_agreement": annotator_agreement(ann),
            }
    predicted_ids = {p["mode_id"] for p in predictions}

    per_mode = []
    for mid, info in truth_by_id.items():
        predicted = mid in predicted_ids
        per_mode.append({
            "mode_id": mid,
            "name": info["name"],
            "ground_truth": info["truth"],
            "predicted": predicted,
            "outcome": (
                "TP" if (predicted and info["truth"]) else
                "FP" if (predicted and not info["truth"]) else
                "FN" if (not predicted and info["truth"]) else
                "TN"
            ),
            "annotator_agreement": info["annotator_agreement"],
            "evidence": next(
                (p["evidence"] for p in predictions if p["mode_id"] == mid), ""
            ),
        })

    return {
        "trace_id": record["trace_id"],
        "mas_name": record["mas_name"],
        "benchmark_name": record["benchmark_name"],
        "n_chars": len(record["trace"]),
        "truncated": truncated,
        "latency_s": latency,
        "raw_response": raw,
        "predictions": predictions,
        "per_mode": per_mode,
        "n_modes_evaluated": len(per_mode),
        "n_ground_truth_positives": sum(1 for m in per_mode if m["ground_truth"]),
        "n_predicted_positives": sum(1 for m in per_mode if m["predicted"]),
        "n_tp": sum(1 for m in per_mode if m["outcome"] == "TP"),
        "n_fp": sum(1 for m in per_mode if m["outcome"] == "FP"),
        "n_fn": sum(1 for m in per_mode if m["outcome"] == "FN"),
        "n_tn": sum(1 for m in per_mode if m["outcome"] == "TN"),
    }


__all__ = [
    "MAX_TRACE_CHARS",
    "JUDGE_SYSTEM_TEMPLATE",
    "annotator_agreement",
    "build_system_prompt",
    "judge_one_trace",
    "majority_vote",
    "mode_name",
    "parse_mode_id",
    "parse_predictions",
]
