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

    args = parser.parse_args(argv)
    if args.cmd == "serve":
        return cmd_serve(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
