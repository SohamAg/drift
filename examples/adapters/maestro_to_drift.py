"""Adapter: MAESTRO OTEL traces -> drift JSONL.

Reads MAESTRO's traces.parquet (one row per OTEL span), groups by run_id,
and emits one drift-format JSONL per run under data/external/maestro/drift_traces/.

The mapping is deliberately faithful (not interpretive) — we preserve
MAESTRO's operation names as drift action `kind` values rather than trying
to guess "what failure mode does this represent." Domain-aware mappings can
be layered on later; first we want to see what drift's existing detectors
fire on the raw data.

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

# Only these operations are "interesting" for drift's purposes. Plumbing
# spans (autogen process/publish/create_agent) are dropped — they describe
# framework infrastructure, not agent decisions.
OPERATIONS_OF_INTEREST = {"invoke_agent", "call_llm", "execute_tool"}


def _coerce_agent_name(span: dict) -> str:
    attrs = span.get("attributes") or {}
    name = (
        attrs.get("gen_ai.agent.name")
        or attrs.get("agent.name")
        or span.get("agent_name")
        or "unknown"
    )
    # MAESTRO sometimes appends a UUID suffix; strip if present.
    if "_" in name and len(name.split("_")[-1]) >= 8:
        name = name.split("_")[0]
    return str(name)


def adapt() -> None:
    if not PARQUET.exists():
        sys.exit(f"missing {PARQUET}; download MAESTRO traces first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(PARQUET)
    runs: dict[str, list[dict]] = defaultdict(list)

    print(f"Reading {pf.metadata.num_rows:,} spans from {PARQUET}")
    for rg_idx in range(pf.num_row_groups):
        tbl = pf.read_row_group(
            rg_idx,
            columns=[
                "run_id", "example_name", "trace_id", "span_id", "parent_span_id",
                "name", "agent_name", "start_time", "end_time", "status", "attributes",
            ],
        )
        for r in tbl.to_pylist():
            op = (r.get("attributes") or {}).get("gen_ai.operation.name")
            if op in OPERATIONS_OF_INTEREST:
                runs[r["run_id"]].append(r)
        print(f"  row group {rg_idx + 1}/{pf.num_row_groups}: {sum(len(v) for v in runs.values()):,} relevant spans so far")

    print(f"\n{len(runs):,} runs with relevant spans")
    written = 0
    skipped = 0
    for run_id, spans in runs.items():
        spans.sort(key=lambda s: s["start_time"])
        if len(spans) < 2:
            skipped += 1
            continue

        # Use the run's first span's trace_id (== the run-level task ID).
        case_id = f"task_{run_id}"
        records: list[dict] = []

        # Initial snapshot at t=1 — the case exists, world is otherwise default.
        records.append({
            "type": "snapshot",
            "timestep": 1,
            "open_cases": {
                case_id: {
                    "case_id": case_id,
                    "issue": spans[0].get("example_name", "unknown"),
                    "opened_at_step": 1,
                    "escalation_count": 0,
                },
            },
            "escalation_queue": [],
        })

        # One action per relevant span, monotonic timestep.
        for i, span in enumerate(spans, start=1):
            attrs = span.get("attributes") or {}
            op = attrs.get("gen_ai.operation.name")
            rationale = span.get("name", "")[:200]
            error_type = attrs.get("error.type")
            if error_type:
                rationale = f"[ERROR {error_type}] {rationale}"

            records.append({
                "type": "action",
                "action_id": f"mae_{run_id}_{i:06d}",
                "timestep": i,
                "agent_name": _coerce_agent_name(span),
                "kind": op,
                "target_case_id": case_id,
                "rationale": rationale,
            })

        # Final snapshot at last step — case still open (we don't model resolution).
        records.append({
            "type": "snapshot",
            "timestep": len(spans),
            "open_cases": {
                case_id: {
                    "case_id": case_id,
                    "issue": spans[0].get("example_name", "unknown"),
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

    print(f"\nWrote {written} JSONL files to {OUT_DIR}")
    if skipped:
        print(f"Skipped {skipped} runs with <2 relevant spans")


if __name__ == "__main__":
    adapt()
