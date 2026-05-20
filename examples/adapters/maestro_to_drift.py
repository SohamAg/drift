"""Adapter: MAESTRO OTEL traces -> drift JSONL (v2, rich-payload).

Reads MAESTRO's traces.parquet (one row per OTEL span), groups by run_id,
and emits one drift-format JSONL per run under data/external/maestro/drift_traces/.
Also emits one ground-truth JSON per run under data/external/maestro/ground_truth/
containing MAESTRO's own run.outcome / run.judgement annotations — used by
the runner script to compute agreement metrics after drift's judge runs.

This is v2 of the adapter. v1 stripped each span down to (agent_name,
operation_name) which gave the judge nothing to work with. v2 packs the
meaningful payload into the action's `rationale` field so the judge sees:
  - tool name + arguments
  - LLM completion excerpt (response text, tool calls)
  - retry attempt number + reason
  - explicit agent.failure.* annotations
  - useless-output flags
  - exceptions

Snapshot strategy: start + every ~10 actions + end, so the judge fires at
multiple points on long traces (each snapshot tick triggers a detector
pass). Ground truth is segregated into a separate file so it doesn't leak
into the judge's input.

Run: python examples/adapters/maestro_to_drift.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


PARQUET = Path("data/external/maestro/traces.parquet")
OUT_DIR = Path("data/external/maestro/drift_traces")
GT_DIR = Path("data/external/maestro/ground_truth")

# Span operations we keep as drift actions. Wider than v1 — we now keep
# embedding and invocation spans too so the judge sees the full agent
# choreography, not just the LLM-call backbone.
OPERATIONS_OF_INTEREST = {
    "invoke_agent",
    "call_llm",
    "execute_tool",
    "invocation",
    "create_agent",
    "embed_documents",
    "embed_query",
}

# Snapshot every N actions in addition to start + end. More snapshots = more
# judge ticks on long traces, but each snapshot is small so storage cost
# stays low. The judge runs with every=1, so each snapshot is one judge call.
SNAPSHOT_EVERY = 10

# Token budget per action rationale. Keeps prompt size sane on long traces
# (median 35 actions × 600 chars = ~5k tokens, well under model limits).
MAX_RATIONALE_CHARS = 600
MAX_COMPLETION_EXCERPT = 350
MAX_TOOL_ARGS_EXCERPT = 200


# --------------------------------------------------------------- helpers --


def _coerce_agent_name(span: dict) -> str:
    attrs = span.get("attributes") or {}
    name = (
        attrs.get("gen_ai.agent.name")
        or attrs.get("agent.name")
        or span.get("agent_name")
        or attrs.get("sender_agent_type")
        or "unknown"
    )
    if not name or name == "None":
        name = "unknown"
    name = str(name)
    if "_" in name and len(name.split("_")[-1]) >= 8:
        name = name.split("_")[0]
    return name


def _excerpt_completion(raw_completion: object) -> str:
    """Pull the salient bits out of gen_ai_completion.

    The field is usually a list of message dicts; the response text is in
    `content`, tool calls in `tool_calls`. We extract whichever exists and
    truncate to MAX_COMPLETION_EXCERPT chars.
    """
    if raw_completion is None:
        return ""
    msgs = raw_completion if isinstance(raw_completion, list) else [raw_completion]
    pieces: list[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, str) and content.strip():
            pieces.append(content.strip())
        tcs = m.get("tool_calls") or []
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            tn = tc.get("name") or "?"
            ta = str(tc.get("arguments") or "")[:MAX_TOOL_ARGS_EXCERPT]
            pieces.append(f"tool:{tn}({ta})")
        finish = m.get("finish_reason")
        if finish and finish != "stop":
            pieces.append(f"finish={finish}")
    if not pieces:
        return ""
    out = " | ".join(pieces)
    return out[:MAX_COMPLETION_EXCERPT].replace("\n", " ").strip()


def _build_rationale(attrs: dict, span_name: str) -> str:
    """Pack the rich span payload into a single-line rationale.

    Composes: operation | tool=name(args) | resp=excerpt | retry=N/reason
              | failure=cat/reason | useless | exc=type
    Only includes pieces that are populated.
    """
    parts: list[str] = []

    op = attrs.get("gen_ai.operation.name") or span_name
    if op:
        parts.append(str(op))

    tool_name = attrs.get("gen_ai.tool.name") or attrs.get("tool.name")
    if tool_name:
        # Try to find tool arguments in a few shapes MAESTRO uses
        tool_args = (
            attrs.get("arguments")
            or attrs.get("gcp.vertex.agent.tool_call_args")
            or ""
        )
        args_str = str(tool_args)[:MAX_TOOL_ARGS_EXCERPT] if tool_args else ""
        parts.append(f"tool={tool_name}({args_str})" if args_str else f"tool={tool_name}")

    resp = _excerpt_completion(attrs.get("gen_ai_completion"))
    if resp:
        parts.append(f"resp={resp}")

    retry_n = attrs.get("agent.retry.attempt_number")
    if retry_n not in (None, "", "None"):
        retry_reason = attrs.get("agent.retry.reason") or attrs.get("agent.retry.trigger") or ""
        parts.append(f"retry=#{retry_n}({retry_reason})" if retry_reason else f"retry=#{retry_n}")

    fail_cat = attrs.get("agent.failure.category")
    if fail_cat not in (None, "", "None"):
        fail_reason = attrs.get("agent.failure.reason") or ""
        parts.append(f"FAILURE={fail_cat}:{fail_reason}")

    useless = attrs.get("agent.output.useless")
    if useless in (True, "true", "True"):
        ureason = attrs.get("agent.output.useless_reason") or ""
        parts.append(f"USELESS={ureason}")

    exc = attrs.get("exception.type") or attrs.get("error.type")
    if exc:
        msg = attrs.get("exception.message") or ""
        parts.append(f"EXC={exc}:{str(msg)[:120]}")

    rationale = " | ".join(parts)
    return rationale[:MAX_RATIONALE_CHARS]


def _extract_ground_truth(spans: list[dict]) -> dict:
    """Pull MAESTRO's own annotations out of the run's spans.

    These appear on whichever span emitted them (often the root run-level
    span). We aggregate everything into a single per-run dict. The judge
    does NOT see this — it's used after the fact for agreement metrics.
    """
    gt = {
        "run_outcome": None,             # success / failure (MAESTRO's verdict)
        "run_judgement": None,           # additional judgment field
        "run_judgement_reason": None,
        "agent_failures": [],            # list of {agent, category, reason}
        "useless_outputs": [],           # list of {agent, reason}
        "retries": [],                   # list of {agent, attempt, reason}
        "exceptions": [],                # list of {type, message}
    }
    for span in spans:
        attrs = span.get("attributes") or {}
        agent = _coerce_agent_name(span)
        if attrs.get("run.outcome"):
            gt["run_outcome"] = attrs["run.outcome"]
        if attrs.get("run.judgement"):
            gt["run_judgement"] = attrs["run.judgement"]
        if attrs.get("run.judgement_reason"):
            gt["run_judgement_reason"] = attrs["run.judgement_reason"]
        if attrs.get("agent.failure.category"):
            gt["agent_failures"].append({
                "agent": agent,
                "category": attrs.get("agent.failure.category"),
                "reason": attrs.get("agent.failure.reason"),
            })
        if attrs.get("agent.output.useless") in (True, "true", "True"):
            gt["useless_outputs"].append({
                "agent": agent,
                "reason": attrs.get("agent.output.useless_reason"),
            })
        if attrs.get("agent.retry.attempt_number"):
            gt["retries"].append({
                "agent": agent,
                "attempt": attrs.get("agent.retry.attempt_number"),
                "reason": attrs.get("agent.retry.reason"),
                "trigger": attrs.get("agent.retry.trigger"),
            })
        if attrs.get("exception.type"):
            gt["exceptions"].append({
                "type": attrs.get("exception.type"),
                "message": attrs.get("exception.message"),
            })
    return gt


# --------------------------------------------------------------- adapt --


def adapt() -> None:
    if not PARQUET.exists():
        sys.exit(f"missing {PARQUET}; download MAESTRO traces first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(PARQUET)
    runs: dict[str, list[dict]] = defaultdict(list)
    all_spans_for_gt: dict[str, list[dict]] = defaultdict(list)

    # We read more attributes than v1 because the judge needs to see them.
    cols = [
        "run_id", "example_name", "trace_id", "span_id", "parent_span_id",
        "name", "agent_name", "start_time", "end_time", "status", "attributes",
    ]

    print(f"Reading {pf.metadata.num_rows:,} spans from {PARQUET}")
    for rg_idx in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg_idx, columns=cols)
        for r in tbl.to_pylist():
            attrs = r.get("attributes") or {}
            op = attrs.get("gen_ai.operation.name")
            # Always keep the span for ground-truth extraction — the
            # annotations often live on parent/root spans that aren't in
            # OPERATIONS_OF_INTEREST.
            all_spans_for_gt[r["run_id"]].append(r)
            if op in OPERATIONS_OF_INTEREST:
                runs[r["run_id"]].append(r)
        print(f"  row group {rg_idx + 1}/{pf.num_row_groups}: {sum(len(v) for v in runs.values()):,} action spans so far")

    print(f"\n{len(runs):,} runs with action spans")
    written = 0
    skipped = 0
    gt_written = 0

    for run_id, spans in runs.items():
        spans.sort(key=lambda s: s["start_time"])
        if len(spans) < 2:
            skipped += 1
            continue

        case_id = f"task_{run_id}"
        example_name = spans[0].get("example_name", "unknown")
        records: list[dict] = []

        # Initial snapshot.
        records.append({
            "type": "snapshot",
            "timestep": 1,
            "open_cases": {
                case_id: {
                    "case_id": case_id,
                    "issue": example_name,
                    "opened_at_step": 1,
                    "escalation_count": 0,
                },
            },
            "escalation_queue": [],
        })

        # Actions — one per span, with rich rationale.
        total_steps = len(spans)
        for i, span in enumerate(spans, start=1):
            attrs = span.get("attributes") or {}
            rationale = _build_rationale(attrs, span.get("name") or "")
            op = attrs.get("gen_ai.operation.name") or "unknown_op"
            records.append({
                "type": "action",
                "action_id": f"mae_{run_id}_{i:06d}",
                "timestep": i,
                "agent_name": _coerce_agent_name(span),
                "kind": op,
                "target_case_id": case_id,
                "rationale": rationale,
            })

            # Intermediate snapshot every SNAPSHOT_EVERY actions (skip the
            # last, which gets its own final snapshot below).
            if i % SNAPSHOT_EVERY == 0 and i < total_steps:
                records.append({
                    "type": "snapshot",
                    "timestep": i,
                    "open_cases": {
                        case_id: {
                            "case_id": case_id,
                            "issue": example_name,
                            "opened_at_step": 1,
                            "escalation_count": 0,
                        },
                    },
                    "escalation_queue": [],
                })

        # Final snapshot.
        records.append({
            "type": "snapshot",
            "timestep": total_steps,
            "open_cases": {
                case_id: {
                    "case_id": case_id,
                    "issue": example_name,
                    "opened_at_step": 1,
                    "escalation_count": 0,
                },
            },
            "escalation_queue": [],
        })

        out_path = OUT_DIR / f"{run_id}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        written += 1

        # Ground truth from ALL spans (annotations often live on non-action spans).
        gt = _extract_ground_truth(all_spans_for_gt[run_id])
        gt["run_id"] = run_id
        gt["example_name"] = example_name
        gt["n_action_spans"] = len(spans)
        (GT_DIR / f"{run_id}.json").write_text(
            json.dumps(gt, indent=2, default=str), encoding="utf-8"
        )
        gt_written += 1

    print(f"\nWrote {written} drift traces to {OUT_DIR}")
    print(f"Wrote {gt_written} ground-truth files to {GT_DIR}")
    if skipped:
        print(f"Skipped {skipped} runs with <2 action spans")

    # Audit: how many ground-truth files actually have something useful?
    gt_with_outcome = 0
    gt_with_failure = 0
    gt_with_useless = 0
    for gt_path in GT_DIR.glob("*.json"):
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        if gt.get("run_outcome"):
            gt_with_outcome += 1
        if gt.get("agent_failures"):
            gt_with_failure += 1
        if gt.get("useless_outputs"):
            gt_with_useless += 1
    print(f"\nGround-truth coverage:")
    print(f"  runs with run.outcome:        {gt_with_outcome}")
    print(f"  runs with agent.failure.*:    {gt_with_failure}")
    print(f"  runs with useless_outputs:    {gt_with_useless}")


if __name__ == "__main__":
    adapt()
