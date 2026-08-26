"""Verify every registered target imports and compiles — no LLM calls.

Building a graph instantiates the model client and wires the nodes, but does
not invoke anything, so this is free and fast. It catches the failure mode
that actually bites: a target that looks fine in the registry but blows up on
`build()` because of an API change in the underlying library.

Usage:
    PYTHONPATH=src python examples/targets/verify_targets.py
    PYTHONPATH=src python examples/targets/verify_targets.py --json
    PYTHONPATH=src python examples/targets/verify_targets.py --no-build   # registry only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from targets import describe, all_targets  # noqa: E402


def _graph_shape(graph) -> dict:
    """Pull node names + edge count out of a compiled graph, defensively."""
    info: dict = {"nodes": [], "n_nodes": 0}
    try:
        g = graph.get_graph()
        nodes = [n for n in g.nodes if n not in ("__start__", "__end__")]
        info["nodes"] = sorted(nodes)
        info["n_nodes"] = len(nodes)
        info["n_edges"] = len(list(g.edges))
    except Exception as e:
        info["shape_error"] = f"{type(e).__name__}: {e}"
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument("--no-build", action="store_true",
                    help="check imports/deps only; skip graph construction")
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    rows = []

    for mod in all_targets():
        row = describe(mod)
        row["built"] = False
        row["build_error"] = ""

        if args.no_build:
            rows.append(row)
            continue
        if row["missing_deps"]:
            row["build_error"] = f"missing deps: {', '.join(row['missing_deps'])}"
            rows.append(row)
            continue
        if row["needs_api_key"] and not has_key:
            row["build_error"] = "OPENAI_API_KEY not set"
            rows.append(row)
            continue

        try:
            graph = mod.build(model=args.model)
            row["built"] = True
            row.update(_graph_shape(graph))
            # initial_state must be constructible too — it's half the contract.
            st = mod.initial_state()
            row["state_keys"] = sorted(st.keys())
        except Exception as e:
            row["build_error"] = f"{type(e).__name__}: {e}"
            if not args.json:
                print(f"\n--- traceback for {row['name']} ---", file=sys.stderr)
                traceback.print_exc()
        rows.append(row)

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0

    ok = sum(1 for r in rows if r["built"])
    print()
    print("=" * 92)
    print(f"drift target registry — {len(rows)} targets, {ok} built OK"
          + ("" if has_key else "   (no OPENAI_API_KEY — build skipped)"))
    print("=" * 92)
    hdr = f"{'target':26s} {'topology':14s} {'vis':4s} {'built':6s} {'nodes':6s} notes"
    print(hdr)
    print("-" * 92)
    for r in rows:
        vis = "yes" if r["visible_to_drift"] else "NO"
        built = "ok" if r["built"] else "--"
        nodes = str(r.get("n_nodes", "")) if r["built"] else ""
        note = r["build_error"] or (", ".join(r.get("nodes", []))[:40])
        print(f"{r['name']:26s} {r['topology']:14s} {vis:4s} {built:6s} {nodes:6s} {note}")
    print()

    by_topo: dict[str, int] = {}
    for r in rows:
        by_topo[r["topology"]] = by_topo.get(r["topology"], 0) + 1
    print("topology coverage:", ", ".join(f"{k}={v}" for k, v in sorted(by_topo.items())))

    blind = [r["name"] for r in rows if not r["visible_to_drift"]]
    if blind:
        print(f"drift-blind targets (negative controls): {', '.join(blind)}")

    failed = [r for r in rows if not r["built"] and not args.no_build]
    if failed:
        print(f"\n{len(failed)} target(s) did not build:")
        for r in failed:
            print(f"  {r['name']}: {r['build_error']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
