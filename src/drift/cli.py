"""drift CLI — `python -m drift`.

Subcommands:
  run     — execute a simulation (default if no subcommand given, for back-compat)
  compare — diff two existing runs by failure counts + per-agent behavior
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal .env loader.

    Looks in (in order, first match wins):
      1. ./drift.env          — CWD-local override (rare; lets you run the
                                 same project with multiple credentials).
      2. ./.env                — CWD-local default.
      3. <project_root>/.env   — the drift checkout itself, regardless of CWD.

    Doesn't override values already set in the environment.
    """
    project_root = Path(__file__).resolve().parents[2]   # e:\drift
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
        break  # first found wins
    # Tolerate the common OPEN_API_KEY typo by aliasing to OPENAI_API_KEY.
    if "OPENAI_API_KEY" not in os.environ and os.environ.get("OPEN_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPEN_API_KEY"]

from drift.events.scheduler import EventScheduler
from drift.llm import ScriptedMockLLM
from drift.llm.base import LLMClient
from drift.observability.logger import RunLogger
from drift.simulation import RunResult, SimulationRunner
from drift.topologies import Topology, get_topology, list_topologies


# ---------------------------------------------------------------- run --

def build_llm(
    name: str,
    seed: int,
    model: str | None = None,
    topology: Topology | None = None,
) -> LLMClient:
    if name == "mock":
        handlers = topology.mock_handlers if topology else None
        return ScriptedMockLLM(seed=seed, role_handlers=handlers)
    if name == "openai":
        from drift.llm.openai_adapter import OpenAILLM
        return OpenAILLM(model=model or "gpt-4o-mini")
    if name == "anthropic":
        from drift.llm.anthropic_adapter import AnthropicLLM
        return AnthropicLLM(model=model or "claude-haiku-4-5")
    raise SystemExit(f"unknown --llm {name!r}; use 'mock', 'openai', or 'anthropic'")


def build_runner(args: argparse.Namespace) -> SimulationRunner:
    topology = get_topology(args.topology)
    llm = build_llm(args.llm, args.seed, model=getattr(args, "model", None), topology=topology)
    variant = args.prompt_variant
    agents = topology.agent_factory(llm)
    # Apply per-role prompt variants from the topology bundle.
    for a in agents:
        prompt = topology.prompts.get((a.role, variant))
        if prompt is None:
            prompt = topology.prompts.get((a.role, "naive"), a.system_prompt)
        a.system_prompt = prompt

    if args.scenario:
        scheduler = EventScheduler.from_yaml(args.scenario, seed=args.seed,
                                             event_registry=topology.event_registry)
    else:
        scheduler = EventScheduler.empty(seed=args.seed,
                                         event_registry=topology.event_registry)

    logger = None
    if not args.no_log:
        logger = RunLogger(base_dir=args.runs_dir, run_id=args.run_id)
    initial = topology.initial_world()
    return SimulationRunner(
        agents=agents,
        scheduler=scheduler,
        steps=args.steps,
        detectors=topology.detectors,
        logger=logger,
        initial_world=initial,
    )


def print_report(result: RunResult) -> None:
    print()
    print("=" * 64)
    print(" DRIFT — Simulation Report")
    print("=" * 64)
    if result.run_dir:
        print(f"Run dir: {result.run_dir}")
    print(f"Final timestep: {result.final_state.timestep}")
    print()

    print("--- Event timeline ---")
    if not result.events:
        print("  (no events fired)")
    else:
        by_step: dict[int, list[str]] = defaultdict(list)
        for e in result.events:
            by_step[e.timestep].append(f"{e.name} ({e.summary})")
        for t in sorted(by_step):
            for line in by_step[t]:
                print(f"  t={t:3d}  {line}")

    print()
    print("--- Detected failures ---")
    if not result.failures:
        print("  (no failures detected)")
    else:
        by_type: dict[str, list] = defaultdict(list)
        for f in result.failures:
            by_type[f.failure_type].append(f)
        for ftype, items in sorted(by_type.items()):
            print(f"  [{ftype}]  count={len(items)}")
            for f in items[:5]:
                agents = ",".join(f.agents_involved) or "-"
                print(f"    t={f.timestep:3d}  agents={agents}  {f.summary}")
            if len(items) > 5:
                print(f"    ... +{len(items) - 5} more")

    print()
    print("--- Final world state ---")
    s = result.final_state
    print(f"  customer_sentiment    = {s.customer_sentiment:.3f}")
    print(f"  refund_policy_version = {s.refund_policy_version}")
    print(f"  inventory_delay_min   = {s.inventory_delay_minutes}")
    print(f"  system_load           = {s.system_load:.3f}")
    print(f"  open_cases            = {len(s.open_cases)}")
    print(f"  escalation_queue      = {len(s.escalation_queue)}")

    print()
    print("--- Agent behavior summary ---")
    for agent_name in sorted(result.metrics.actions_by_agent):
        print(f"  {result.metrics.agent_summary(agent_name)}")
    print()


def cmd_run(args: argparse.Namespace) -> int:
    runner = build_runner(args)
    # Persist run metadata so this run is forkable / comparable later.
    if runner.logger:
        import datetime as _dt
        meta = {
            "topology": args.topology,
            "scenario": args.scenario.name if args.scenario else None,
            "seed": args.seed,
            "llm": args.llm,
            "model": args.model,
            "prompt_variant": args.prompt_variant,
            "steps": args.steps,
            "started_at": _dt.datetime.now().isoformat(timespec="seconds"),
        }
        (runner.logger.run_dir / "run_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
    try:
        result = asyncio.run(runner.run())
    finally:
        if runner.logger:
            runner.logger.close()
    print_report(result)
    return 0


# ------------------------------------------------------------ compare --

def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summarize_run(run_dir: Path) -> dict:
    failures = _load_jsonl(run_dir / "failures.jsonl")
    actions = _load_jsonl(run_dir / "actions.jsonl")
    snapshots = _load_jsonl(run_dir / "snapshots.jsonl")

    failure_counts: Counter = Counter(f["failure_type"] for f in failures)
    action_kinds_by_agent: dict[str, Counter] = defaultdict(Counter)
    for a in actions:
        action_kinds_by_agent[a["agent_name"]][a["kind"]] += 1
    final = snapshots[-1] if snapshots else {}
    return {
        "failure_counts": failure_counts,
        "action_kinds_by_agent": action_kinds_by_agent,
        "final": final,
        "n_actions": len(actions),
        "n_failures": len(failures),
        "n_snapshots": len(snapshots),
    }


def _delta(a: int, b: int) -> str:
    d = b - a
    if d == 0:
        return f"  ={a}"
    sign = "+" if d > 0 else ""
    return f"  {a} -> {b} ({sign}{d})"


def cmd_compare(args: argparse.Namespace) -> int:
    a_dir, b_dir = Path(args.run_a), Path(args.run_b)
    if not a_dir.is_dir() or not b_dir.is_dir():
        print(f"error: both run dirs must exist. got {a_dir} {b_dir}", file=sys.stderr)
        return 2

    a, b = _summarize_run(a_dir), _summarize_run(b_dir)

    print()
    print("=" * 72)
    print(f" DRIFT — Run Comparison")
    print("=" * 72)
    print(f"  A: {a_dir.name}")
    print(f"  B: {b_dir.name}")
    print()

    print(f"--- Headline ---")
    print(f"  total failures      {_delta(a['n_failures'], b['n_failures'])}")
    print(f"  total actions       {_delta(a['n_actions'], b['n_actions'])}")
    print()

    print("--- Failures by type ---")
    all_types = sorted(set(a["failure_counts"]) | set(b["failure_counts"]))
    if not all_types:
        print("  (none in either run)")
    else:
        print(f"  {'failure_type':28s}  {'A':>6s}  {'B':>6s}  delta")
        for t in all_types:
            ca, cb = a["failure_counts"].get(t, 0), b["failure_counts"].get(t, 0)
            d = cb - ca
            marker = "  ==" if d == 0 else ("  DOWN" if d < 0 else "  UP")
            print(f"  {t:28s}  {ca:>6d}  {cb:>6d}  {d:+d}{marker}")
    print()

    print("--- Action mix per agent ---")
    all_agents = sorted(set(a["action_kinds_by_agent"]) | set(b["action_kinds_by_agent"]))
    for ag in all_agents:
        ka = a["action_kinds_by_agent"].get(ag, Counter())
        kb = b["action_kinds_by_agent"].get(ag, Counter())
        kinds = sorted(set(ka) | set(kb))
        print(f"  [{ag}]")
        for k in kinds:
            print(f"     {k:18s}  A={ka.get(k,0):>3d}   B={kb.get(k,0):>3d}   delta={kb.get(k,0)-ka.get(k,0):+d}")
    print()

    print("--- Final world state ---")
    fa, fb = a.get("final", {}), b.get("final", {})
    keys = ("customer_sentiment", "refund_policy_version", "inventory_delay_minutes", "system_load")
    for k in keys:
        va, vb = fa.get(k), fb.get(k)
        print(f"  {k:24s}  A={va}   B={vb}")
    print()

    return 0


# ---------------------------------------------------------------- main --

def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--topology", default="support",
                   help=f"topology bundle (one of {list_topologies()}; default: support)")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scenario", type=Path, default=None)
    p.add_argument("--llm", choices=["mock", "openai", "anthropic"], default="mock")
    p.add_argument("--model", default=None)
    p.add_argument("--prompt-variant", choices=["naive", "hardened"], default="naive")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--run-id", default=None, help="explicit run dir name (default: timestamp)")
    p.add_argument("--no-log", action="store_true")


def _parse_kv_pairs(items: list[str] | None) -> dict[str, str]:
    """Parse --variant role=value occurrences into a dict. Empty list → {}."""
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--variant must be role=value, got {item!r}")
        role, _, value = item.partition("=")
        out[role.strip()] = value.strip()
    return out


def cmd_fork(args: argparse.Namespace) -> int:
    from drift.fork import ForkConfig, ForkOverrides, build_fork
    overrides = ForkOverrides(
        seed=args.seed,
        prompt_variants=_parse_kv_pairs(args.variant),
        disabled_agents=set(args.disable or []),
    )
    cfg = ForkConfig(
        parent_run_id=args.parent,
        branch_at_step=args.at,
        overrides=overrides,
        new_run_id=args.run_id,
        extend_by=args.extend,
    )
    try:
        from drift.testing import reset_all_counters
        reset_all_counters()
        runner, meta = build_fork(cfg, runs_dir=args.runs_dir)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        result = asyncio.run(runner.run())
    finally:
        if runner.logger:
            runner.logger.close()
    print(f"forked {cfg.parent_run_id} at t={cfg.branch_at_step}")
    print(f"new run: {runner.logger.run_dir.name if runner.logger else '(unlogged)'}")
    if overrides.seed is not None:           print(f"  seed override            : {overrides.seed}")
    if overrides.prompt_variants:            print(f"  prompt variant overrides : {overrides.prompt_variants}")
    if overrides.disabled_agents:            print(f"  disabled agents          : {sorted(overrides.disabled_agents)}")
    print()
    print_report(result)
    return 0


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
    parser = argparse.ArgumentParser(prog="drift", description="Multi-agent stress-test simulator.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="execute a simulation")
    _add_run_args(run_p)

    cmp_p = sub.add_parser("compare", help="diff two existing run directories")
    cmp_p.add_argument("run_a", help="path to run A directory")
    cmp_p.add_argument("run_b", help="path to run B directory")

    srv_p = sub.add_parser("serve", help="launch the local web UI")
    srv_p.add_argument("--host", default="127.0.0.1")
    srv_p.add_argument("--port", type=int, default=8765)
    srv_p.add_argument("--reload", action="store_true", help="auto-reload on code changes (dev)")

    fork_p = sub.add_parser("fork", help="counterfactual replay: re-run a parent run from a chosen step with overrides")
    fork_p.add_argument("--parent", required=True, help="parent run_id to fork from")
    fork_p.add_argument("--at", type=int, required=True, help="branch at this timestep (0 = from start)")
    fork_p.add_argument("--seed", type=int, default=None, help="override RNG seed (defaults to parent's seed)")
    fork_p.add_argument("--variant", action="append", default=None,
                        help="prompt-variant override: role=naive|hardened (repeatable)")
    fork_p.add_argument("--disable", action="append", default=None,
                        help="disable an agent by name (repeatable)")
    fork_p.add_argument("--extend", type=int, default=None,
                        help="extra steps to run from the branch point (defaults to parent's total - at)")
    fork_p.add_argument("--run-id", default=None, help="explicit run dir name for the fork")
    fork_p.add_argument("--runs-dir", type=Path, default=Path("runs"))

    args = parser.parse_args(argv)
    if args.cmd == "compare":
        return cmd_compare(args)
    if args.cmd == "serve":
        return cmd_serve(args)
    if args.cmd == "fork":
        return cmd_fork(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
