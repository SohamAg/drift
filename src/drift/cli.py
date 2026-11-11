"""drift CLI — `python -m drift` / `drift`.

    drift run  <module[:build_fn]>   run drift_test against an importable graph,
                                     print the verdicts, optionally write the
                                     JSON the web UI reads and fail the build
    drift gate <results.json>        apply pass/fail rules to a saved result
    drift targets                    list the registry of real third-party MAS
                                     usable with `drift run --target NAME`
    drift serve                      the local web UI
    drift report <results.json>      self-contained HTML evidence report
    drift history [<graph>]          regression trend from tracked runs

`run` + `gate` are the CI path: run on a box with the API key, commit the
JSON as an artifact, gate anywhere. See docs/ci.md.
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


def _parse_graph_spec(spec: str) -> tuple[str, str]:
    """`pkg.mod` → (pkg.mod, "build"); `pkg.mod:make_graph` → (pkg.mod, make_graph)."""
    module, _, fn = spec.partition(":")
    module = module.strip()
    if not module:
        raise SystemExit("graph spec must be 'module' or 'module:build_fn'")
    return module, (fn.strip() or "build")


def _import_graph(module: str, build_fn: str, initial_state_fn: str, query: str | None, model: str | None):
    """Import `module`, call build()/initial_state(). sys.path gets the CWD
    so `drift run my_graph` works on a file in the current directory."""
    import importlib

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        mod = importlib.import_module(module)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"could not import {module!r}: {type(e).__name__}: {e}")
    fn = getattr(mod, build_fn, None)
    if not callable(fn):
        raise SystemExit(f"{module}.{build_fn} is not callable")
    try:
        graph = fn(model=model) if model else fn()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"{build_fn}() failed: {type(e).__name__}: {e}")
    state_fn = getattr(mod, initial_state_fn, None)
    if not callable(state_fn):
        raise SystemExit(
            f"{module}.{initial_state_fn} is not callable — drift needs the initial state "
            f"your graph is invoked with (use --initial-state-fn to name it)"
        )
    try:
        state = state_fn(query) if query else state_fn()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"{initial_state_fn}() failed: {type(e).__name__}: {e}")
    if not isinstance(state, dict):
        try:
            state = dict(state)
        except Exception:  # noqa: BLE001
            raise SystemExit(f"{initial_state_fn}() must return a dict, got {type(state).__name__}")
    agents = []
    try:
        nodes = getattr(graph.get_graph(), "nodes", {})
        agents = [n for n in nodes if n not in ("__start__", "__end__")]
    except Exception:  # noqa: BLE001
        pass
    return graph, state, {"name": f"{module}.{build_fn}", "description": f"imported — {module}", "agents": agents}


def _resolve_target(name: str, query: str | None, model: str | None):
    """A registry target from examples/targets (real third-party MAS)."""
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from examples.targets import describe, get
    except ImportError as e:
        raise SystemExit(f"target registry unavailable ({e}); run from a drift checkout or pass a module")
    try:
        mod = get(name)
    except KeyError as e:
        raise SystemExit(str(e))
    info = describe(mod)
    if info["missing_deps"]:
        raise SystemExit(f"target {name!r} needs {info['missing_deps']} — see docs/targets_setup.md")
    graph = mod.build(model=model) if model else mod.build()
    state = dict(mod.initial_state(query or mod.DEFAULT_QUERY))
    return graph, state, {"name": name, "description": mod.DESCRIPTION, "agents": []}


def cmd_run(args: argparse.Namespace) -> int:
    import json
    import time

    from drift.adapters.langgraph import drift_test
    from drift.adapters.serialize import shape_adapter_result
    from drift.gate import PRESETS, evaluate, parse_rules

    try:
        rules = parse_rules(args.fail_on)
    except ValueError as e:
        raise SystemExit(f"--fail-on: {e}")

    knobs = dict(PRESETS[args.preset])
    for key in ("intensity", "max_perturbations", "judge", "divergence_mode",
                "baseline_rollouts", "perturbation_rollouts", "max_judge_calls"):
        v = getattr(args, key, None)
        if v is not None:
            knobs[key] = v

    if args.target:
        graph, state, meta = _resolve_target(args.target, args.query, args.model)
    else:
        module, build_fn = _parse_graph_spec(args.graph)
        graph, state, meta = _import_graph(module, build_fn, args.initial_state_fn, args.query, args.model)

    judge_llm = None
    if knobs["judge"] != "off":
        from drift.failures.judge import build_judge

        try:
            judge_llm = build_judge(knobs["judge"], model=args.judge_model)
        except ValueError as e:
            raise SystemExit(f"judge: {e}")

    from drift import guidelines as _g

    spec = None
    try:
        if args.guidelines:
            spec = _g.load(args.guidelines)
        elif not args.no_guidelines:
            spec = _g.load_default()
    except (OSError, _g.GuidelineError) as e:
        raise SystemExit(f"guidelines: {e}")
    guidelines = list(spec.judge) if spec else []
    rule_list = list(spec.rules) if spec else []
    if spec and not args.quiet:
        print(f"guidelines: {spec.source} — {len(spec.judge)} judge string(s), {len(spec.rules)} rule(s)",
              file=sys.stderr)

    if not args.quiet:
        print(
            f"drift run · {meta['name']} · preset={args.preset} intensity={knobs['intensity']} "
            f"max_perturbations={knobs['max_perturbations']} rollouts={knobs['perturbation_rollouts']} "
            f"divergence={knobs['divergence_mode']} judge={knobs['judge']}",
            file=sys.stderr,
        )
    t0 = time.perf_counter()
    try:
        result = drift_test(
            graph=graph,
            initial_state=state,
            intensity=knobs["intensity"],
            seed=args.seed,
            max_perturbations=knobs["max_perturbations"],
            judge_llm=judge_llm,
            user_guidelines=guidelines or None,
            divergence_mode=knobs["divergence_mode"],
            baseline_rollouts=knobs["baseline_rollouts"],
            perturbation_rollouts=knobs["perturbation_rollouts"],
            max_judge_calls=knobs["max_judge_calls"],
            auto_chaos_exclude=args.exclude or None,
            rules=spec,
            tool_chaos=not args.no_tool_chaos,
            chaos_timing=args.chaos_timing,
        )
    except (TypeError, ValueError) as e:
        raise SystemExit(f"drift_test: {e}")
    took = time.perf_counter() - t0

    shaped = shape_adapter_result(
        result, graph_meta=meta, seed=args.seed, judge=knobs["judge"], judge_model=args.judge_model,
        user_guidelines=guidelines, track_history=args.track_history,
        history_dir=Path(args.history_dir) if args.history_dir else None,
        guideline_rules=[r.to_dict() for r in rule_list],
    )

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(shaped, indent=1, default=str), encoding="utf-8")

    report = evaluate(shaped, rules)
    if not args.quiet:
        _print_run(shaped, took)
        for line in report.lines():
            print(line)
        if args.json:
            print(f"json written to {args.json}  (open it in the web UI's Results tab, or `drift report {args.json}`)")
    elif not report.passed:
        for line in report.lines():
            print(line)
    return 0 if report.passed else 1


def _print_run(shaped: dict, took: float) -> None:
    for line in shaped.get("summary_lines") or []:
        print(line)
    b = shaped.get("baseline") or {}
    if b.get("crashed"):
        print(f"  ! baseline crashed: {b.get('error_type')}: {b.get('error')}")
    for f in b.get("coordination_findings") or []:
        print(f"  baseline  {f.get('failure_type')}: {f.get('summary')}")
    rank = {True: 0}
    perts = sorted(
        shaped.get("perturbations") or [],
        key=lambda p: (0 if p.get("crashed") else 1 if p.get("diverged") else 2, p.get("event_name", "")),
    )
    for p in perts:
        if p.get("crashed"):
            verdict = "CRASH   "
        elif p.get("diverged"):
            verdict = "DIVERGE "
        else:
            verdict = "ok      "
        extra = ""
        if (p.get("n_rollouts") or 1) > 1:
            extra = f"  [{p.get('outcome_label')}]"
            if p.get("is_intermittent"):
                extra += " intermittent"
        if p.get("fault_layer") == "tool":
            extra += "  (tool fault" + ("" if p.get("fault_fired") else " — did not fire") + ")"
        if p.get("chaos_timing") == "mid_run":
            extra += f"  (mid-run @ step {p.get('fork_step')})"
        elif p.get("timing_note"):
            extra += "  (initial — " + str(p["timing_note"]).replace("fell back to initial-state timing: ", "") + ")"
        print(f"{verdict} {p.get('event_name')}{extra}")
        if p.get("crashed"):
            print(f"           {p.get('error_type')}: {str(p.get('error', ''))[:160]}")
        elif p.get("diverged") and p.get("divergence_summary"):
            print(f"           {str(p['divergence_summary'])[:160]}")
        for f in p.get("coordination_findings") or []:
            print(f"           ! {f.get('failure_type')}: {str(f.get('summary', ''))[:200]}")
        for f in p.get("judge_findings") or []:
            print(f"           ? judge {f.get('failure_type', '')}: {str(f.get('summary', ''))[:160]}")
    print(f"({took:.1f}s)")


def cmd_targets(args: argparse.Namespace) -> int:
    """List examples/targets — real third-party multi-agent systems drift is
    validated against — with whether each is runnable here."""
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from examples.targets import registry
    except ImportError as e:
        raise SystemExit(f"target registry unavailable ({e}); run from a drift checkout")
    rows = registry()
    if args.json:
        import json

        print(json.dumps(rows, indent=1))
        return 0
    width = max(len(r["name"]) for r in rows) if rows else 10
    for r in rows:
        status = "ready" if r["importable"] else f"missing {', '.join(r['missing_deps'])}"
        key = " · needs OPENAI_API_KEY" if r["needs_api_key"] else ""
        print(f"{r['name']:<{width}}  {r['topology']:<13} {status}{key}")
        print(f"{'':<{width}}  {r['description']}")
        print(f"{'':<{width}}  source: {r['source']}")
    print()
    print("run one:  drift run --target <name> [--query ...] [--preset quick]")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    import json

    from drift.gate import evaluate, parse_rules

    try:
        rules = parse_rules(args.fail_on)
    except ValueError as e:
        raise SystemExit(f"--fail-on: {e}")
    if not rules:
        raise SystemExit("gate: give at least one --fail-on rule")
    path = Path(args.results)
    try:
        shaped = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SystemExit(f"could not read {path}: {e}")
    if not isinstance(shaped, dict) or "perturbations" not in shaped:
        raise SystemExit(f"{path} is not a drift result (no 'perturbations' key) — pass `drift run --json` output")
    report = evaluate(shaped, rules)
    for line in report.lines():
        print(line)
    return 0 if report.passed else 1


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        from drift.server import serve
    except ImportError as e:
        raise SystemExit(
            f"web dependencies not installed ({e}). Install with: pip install drift[web]"
        )
    browser_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{browser_host}:{args.port}/"
    print(f"drift web UI on {url}")
    if not args.no_browser:
        import threading
        import webbrowser

        def _open_browser() -> None:
            try:
                webbrowser.open(url)
            except Exception:
                pass  # headless/no-display environment — the printed URL still works

        threading.Timer(1.0, _open_browser).start()
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
    # Verdict lines use "×" and "—"; a cp1252 Windows console would raise on
    # them. Replace rather than crash — the words are what matter.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(
        prog="drift",
        description="Pre-deploy chaos testing for LangGraph multi-agent systems.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser(
        "run",
        help="run drift_test against an importable graph; print verdicts; optionally gate",
        description=(
            "Import MODULE (optionally MODULE:build_fn), call build() for the compiled graph and "
            "initial_state() for the input, run drift's auto-chaos, print the verdicts. "
            "--json writes the same JSON the web UI renders; --fail-on makes the exit code 1 on a violation."
        ),
    )
    run_p.add_argument("graph", nargs="?", default=None,
                       help="importable module, or module:build_fn (omit when using --target)")
    run_p.add_argument("--target", default=None,
                       help="a registry target from examples/targets instead of a module (e.g. supervisor_basic)")
    run_p.add_argument("--initial-state-fn", default="initial_state",
                       help="name of the function returning the initial state (default: initial_state)")
    run_p.add_argument("--query", default=None, help="passed to initial_state(query) when given")
    run_p.add_argument("--model", default=None, help="passed to build(model=...) when given")
    run_p.add_argument("--preset", choices=("quick", "balanced", "thorough", "exhaustive"), default="quick",
                       help="chaos/judge preset (default: quick — no LLM calls). Any knob below overrides it.")
    run_p.add_argument("--intensity", choices=("off", "light", "moderate", "aggressive", "exhaustive"), default=None)
    run_p.add_argument("--max-perturbations", dest="max_perturbations", type=int, default=None)
    run_p.add_argument("--rollouts", dest="perturbation_rollouts", type=int, default=None,
                       help="run each perturbed input N times and report k/N")
    run_p.add_argument("--baseline-rollouts", dest="baseline_rollouts", type=int, default=None)
    run_p.add_argument("--divergence", dest="divergence_mode", choices=("exact", "tiered", "off"), default=None)
    run_p.add_argument("--judge", choices=("off", "openai", "mock"), default=None)
    run_p.add_argument("--judge-model", default=None)
    run_p.add_argument("--max-judge-calls", dest="max_judge_calls", type=int, default=None)
    run_p.add_argument("--exclude", action="append", default=None,
                       help="chaos pattern to skip (repeatable), e.g. --exclude inject_tool_result_instruction")
    run_p.add_argument("--guidelines", default=None,
                       help="guideline file: drift.yml (judge strings + deterministic rules, see docs/guidelines.md) "
                            "or a text file with one judge guideline per line. Defaults to ./drift.yml or "
                            "./.drift/guidelines.yml when present.")
    run_p.add_argument("--no-guidelines", action="store_true", help="ignore a default drift.yml")
    run_p.add_argument("--no-tool-chaos", action="store_true",
                       help="skip tool-layer patterns (runtime faults on the graph's ToolNode tools)")
    run_p.add_argument("--timing", dest="chaos_timing", choices=("initial", "mid_run"), default="initial",
                       help="where state patterns land: perturb the initial state (default) or fork the "
                            "baseline at the scheduled step and apply the pattern mid-run")
    run_p.add_argument("--seed", type=int, default=7)
    run_p.add_argument("--json", default=None, help="write the shaped result JSON here")
    run_p.add_argument("--fail-on", action="append", default=None,
                       help="gate rule(s), comma-separable and repeatable: crash, recursion_limit, diverge, "
                            "intermittent, finding, finding:<type>, baseline_finding, guideline, judge, "
                            "max_crash_rate=<0..1>, max_diverge_rate=<0..1>")
    run_p.add_argument("--track-history", action="store_true",
                       help="append a regression snapshot (see `drift history`)")
    run_p.add_argument("--history-dir", default=None)
    run_p.add_argument("--quiet", "-q", action="store_true", help="print only gate failures")

    gate_p = sub.add_parser("gate", help="apply --fail-on rules to a saved `drift run --json` result")
    gate_p.add_argument("results", help="path to a drift result JSON")
    gate_p.add_argument("--fail-on", action="append", required=True,
                        help="same rules as `drift run --fail-on`")

    tgt_p = sub.add_parser("targets", help="list the registry of real third-party MAS for --target")
    tgt_p.add_argument("--json", action="store_true", help="machine-readable output")

    srv_p = sub.add_parser("serve", help="launch the local web UI")
    srv_p.add_argument("--host", default="127.0.0.1")
    srv_p.add_argument("--port", type=int, default=8765)
    srv_p.add_argument("--reload", action="store_true", help="auto-reload on code changes (dev)")
    srv_p.add_argument(
        "--no-browser", action="store_true",
        help="don't automatically open the dashboard in a browser on startup",
    )

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
    if args.cmd == "run":
        if not args.graph and not args.target:
            run_p.error("give a module (module[:build_fn]) or --target NAME")
        return cmd_run(args)
    if args.cmd == "gate":
        return cmd_gate(args)
    if args.cmd == "targets":
        return cmd_targets(args)
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
