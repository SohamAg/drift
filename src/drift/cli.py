"""drift CLI — `python -m drift`.

After the 2026-06-29 cleanup the native simulator subcommands (`run`,
`fork`, `compare`) were removed alongside the simulator. The CLI now
exposes one subcommand: `serve`, which launches the web UI for the
LangGraph adapter.

Future subcommands planned (see FUTURE_DIRECTIONS.md): `drift test`
to invoke the adapter from a config file, `drift ci` for CI-shaped
output, etc. None of those are built yet.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal .env loader.

    Looks in (in order, first match wins):
      1. ./drift.env          — CWD-local override.
      2. ./.env                — CWD-local default.
      3. <project_root>/.env   — the drift checkout itself, regardless of CWD.

    Doesn't override values already set in the environment.
    """
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        Path.cwd() / "drift.env",
        Path.cwd() / ".env",
        project_root / ".env",
    ]
    for path in candidates:
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
        break
    if "OPENAI_API_KEY" not in os.environ and os.environ.get("OPEN_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPEN_API_KEY"]


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        from drift.server import serve
    except ImportError as e:
        raise SystemExit(
            f"web dependencies not installed ({e}). Install with: pip install drift[web]"
        )
    print(f"drift web UI on http://{args.host}:{args.port}/")
    serve(host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from drift.report import write_report

    out = write_report(
        args.results,
        fork_path=args.fork,
        out_path=args.out,
        title=args.title,
        target_name=args.target,
    )
    print(f"evidence report written to {out}")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    """Print the regression-tracking trend for one graph, or list every
    graph that has tracked history. Reads what `drift_test(... track_history
    ...)` / the web UI's "Track for regression" checkbox already wrote —
    this command never runs anything itself."""
    import datetime

    from drift.regression import DEFAULT_HISTORY_DIR, compare, load_history

    history_dir = Path(args.history_dir) if args.history_dir else DEFAULT_HISTORY_DIR

    if args.list or not args.graph_signature:
        if not history_dir.exists():
            print("no tracked graphs yet — run with track_history=True first")
            return 0
        names = sorted(p.stem for p in history_dir.glob("*.jsonl"))
        if not names:
            print("no tracked graphs yet — run with track_history=True first")
            return 0
        print("tracked graphs:")
        for name in names:
            print(f"  {name}")
        print("\nShow one graph's trend: drift history <graph_signature>")
        return 0

    snapshots = load_history(args.graph_signature, history_dir)
    if not snapshots:
        print(f"no tracked history for {args.graph_signature!r}", file=sys.stderr)
        print("run `drift history --list` to see tracked graph signatures", file=sys.stderr)
        return 1

    prior: list = []
    for snap in snapshots:
        ts = datetime.datetime.fromtimestamp(snap.timestamp).strftime("%Y-%m-%d %H:%M")
        report = compare(snap, prior)
        print(
            f"{ts}  intensity={snap.intensity:<10} "
            f"crashed={snap.n_crashed}/{snap.patterns_total} "
            f"diverged={snap.n_diverged}/{snap.patterns_total} "
            f"coord={snap.n_coordination_findings}"
        )
        for flag in report.flags:
            marker = "!" if flag.severity in ("crash", "regression") else "i"
            print(f"    [{marker}] {flag.summary}")
        prior.append(snap)
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(
        prog="drift",
        description="Pre-deploy chaos testing for LangGraph multi-agent systems.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    srv_p = sub.add_parser("serve", help="launch the local web UI")
    srv_p.add_argument("--host", default="127.0.0.1")
    srv_p.add_argument("--port", type=int, default=8765)
    srv_p.add_argument("--reload", action="store_true", help="auto-reload on code changes (dev)")

    rep_p = sub.add_parser(
        "report",
        help="render a saved results JSON into a self-contained HTML evidence report",
    )
    rep_p.add_argument("results", help="path to a runner results JSON")
    rep_p.add_argument("--fork", default=None,
                       help="optional fork-edit-replay proof JSON (fork_edit_demo --save-json)")
    rep_p.add_argument("--out", default=None,
                       help="output HTML path (default: <results>.report.html)")
    rep_p.add_argument("--title", default=None, help="report title")
    rep_p.add_argument("--target", default=None, help="target system name shown in the header")

    hist_p = sub.add_parser(
        "history",
        help="print the regression-tracking trend for a graph (see drift.regression)",
    )
    hist_p.add_argument(
        "graph_signature", nargs="?", default=None,
        help="graph_name (bundled) or module.build_fn (imported) — as passed to track_history",
    )
    hist_p.add_argument("--list", action="store_true", help="list every tracked graph signature")
    hist_p.add_argument(
        "--history-dir", default=None,
        help="override the history directory (default: <cwd>/results/_history)",
    )

    args = parser.parse_args(argv)
    if args.cmd == "serve":
        return cmd_serve(args)
    if args.cmd == "report":
        return cmd_report(args)
    if args.cmd == "history":
        return cmd_history(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
