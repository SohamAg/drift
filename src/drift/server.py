"""drift web server — FastAPI backend for the local UI.

After the 2026-06-29 cleanup all native-simulator endpoints (/api/runs,
/api/compare, /api/byoa, /api/topologies, /api/scenarios) were removed
along with the simulator. The server now exposes only adapter-shaped
endpoints:

  GET  /api/adapter-graphs           — list bundled graphs the Test tab can run
  POST /api/adapter-demo             — run drift_test against a chosen bundled graph
  POST /api/custom-graph-run         — run drift_test against a user-imported module
  POST /api/adapter-fork             — fork-edit-replay a completed run at one step
  POST /api/trace-ask                — ask a free-text question about one captured trace
  POST /api/save-run                 — persist a run's full JSON so it's loadable later
  GET  /api/results                  — list saved experiment JSON under results/
  GET  /api/results/{relpath:path}   — fetch one saved result by path
  GET  /api/mast-demos               — list curated MAST demo traces
  POST /api/mast-analyze             — judge one MAST trace (cached or live)
  GET  /api/targets                  — list the examples/targets/ registry
  POST /api/suite-run                — run drift against several targets in one pass
  POST /api/propose-target             — ask an LLM to draft a topology + agent roster from a scenario
  GET  /api/historical-events          — the real seed dataset for grounding/calibration
  POST /api/simulate-scenario          — propose + ground + generate + run a verdict for a scenario
  POST /api/calibrate                  — score a verdict-producing target against real held-out history
  POST /api/build-target              — generate a LangGraph target from topology + agents
  POST /api/save-target               — write a generated target to examples/targets/user_generated/

The HTML/CSS/JS frontend is served from /web/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# Resolved at import so endpoints find paths regardless of CWD.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Every other script in this repo loads .env before touching OPENAI_API_KEY;
# the server never did, silently requiring the key to already be exported
# in the launching shell. Load it here too so `python -m drift serve` works
# the same way `python examples/...` scripts do.
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# examples/targets/ is imported as the top-level `targets` package (not
# `examples.targets` — examples/ has no __init__.py) by every script that
# already touches the registry (tests/test_target_registry.py,
# examples/targets/verify_targets.py). Match that convention here so
# /api/targets and /api/suite-run can import it the same way.
_EXAMPLES_DIR = str(PROJECT_ROOT / "examples")
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

WEB_DIR: Path = PROJECT_ROOT / "web"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
# Per-graph regression history (see drift.regression). A module-level
# constant rather than relying on drift.regression's own default so tests
# can monkeypatch `drift.server.HISTORY_DIR` — regression.track()'s default
# argument is bound once at import time and wouldn't see a patched module
# constant on drift.regression itself.
HISTORY_DIR: Path = RESULTS_DIR / "_history"
# Build tab's generated targets (see drift.target_builder). Gitignored
# except its own __init__.py — these are arbitrary user experiments, not
# the curated third-party-only registry under examples/targets/.
USER_GENERATED_DIR: Path = PROJECT_ROOT / "examples" / "targets" / "user_generated"
MAST_DATASET: Path = PROJECT_ROOT / "data" / "external" / "mast" / "MAD_human_labelled_dataset.json"
MAST_CACHED_RESULTS_DIR: Path = PROJECT_ROOT / "results" / "mast_judge" / "full"


# Curated MAST traces for the "MAST" demo. Each entry pairs one MAST
# trace_id with a short human-readable narrative so the demo can show
# "real published multi-agent run X had these human-flagged failures —
# here is what drift's judge sees." Picked to span frameworks + outcomes
# (clean win, mixed, hard miss) so the demo is honest about scope.
MAST_DEMO_TRACES: list[dict[str, Any]] = [
    {
        "id": 2,
        "title": "AG2 — Christmas ribbon math",
        "mas_name": "AG2",
        "task_brief": "Math word problem with insufficient data. Agents debate how much ribbon to use per gift bow.",
        "story": "WIN",
        "story_blurb": "Drift caught both failure modes the human annotators flagged, with zero false alarms.",
    },
    {
        "id": 11,
        "title": "AppWorld — Bucket-list manager",
        "mas_name": "AppWorld",
        "task_brief": "Supervisor + Spotify/Notes/etc. agents coordinate to mark items off a bucket list.",
        "story": "MIXED",
        "story_blurb": "Drift caught 2 of 4 real failures (task drift and constraint violation) but missed both verification-related ones.",
    },
    {
        "id": 9,
        "title": "MetaGPT — Budget tracker app",
        "mas_name": "MetaGPT",
        "task_brief": "Coder + Tester agents build a budget tracker. Tests don't actually match the code's logic.",
        "story": "MIXED",
        "story_blurb": "Drift correctly flagged the reasoning-action mismatch but missed the backtracking failure.",
    },
    {
        "id": 3,
        "title": "ChatDev — Sudoku project (hard case)",
        "mas_name": "ChatDev",
        "task_brief": "Software dev pipeline (CEO/CTO/coder/tester) builds a Sudoku app over hundreds of agent turns.",
        "story": "MISS",
        "story_blurb": "Long-trace honest limit: 9 human-flagged failures, drift's current generic prompt caught none. Shows the prompt-iteration headroom.",
    },
]


# ---- request models -------------------------------------------------------

class MastAnalyzeRequest(BaseModel):
    """Judge one curated MAST trace, either via cached result or a live call."""
    trace_id: int
    mode: str = "cached"  # cached | live
    judge_model: str = "gpt-4o-mini"
    user_guidelines: list[str] = Field(default_factory=list)


class AdapterDemoRequest(BaseModel):
    """Run drift's langgraph adapter against a bundled graph.

    Backs the Adapter tab. Server-side graph build keeps user code out of
    the request surface — safe to expose. For arbitrary graphs the user
    installs drift locally and calls drift_test from a notebook.
    """
    # Which graph to run.
    # "ticket_triage" — bundled 3-node demo, always available.
    # "langgraph_supervisor" — requires langgraph-supervisor + OPENAI_API_KEY.
    graph_name: str = "ticket_triage"
    # Per-graph overrides; currently used to pass the user's question.
    state_overrides: dict[str, Any] = Field(default_factory=dict)

    # Chaos knobs.
    # off | light | moderate | aggressive | exhaustive.
    intensity: str = "aggressive"
    seed: int = 7
    max_perturbations: int = Field(default=25, ge=1, le=500)
    auto_chaos_exclude: list[str] = Field(default_factory=list)

    # Judge config (optional LLM over traces).
    judge: str = "off"
    judge_model: str | None = None
    user_guidelines: list[str] = Field(default_factory=list)

    # Agent/node name -> role label (e.g. "verifier", "executor"). Powers
    # role-aware library detectors (role_drift); falls back to name-pattern
    # matching when empty. Same shape as AdapterForkRequest's field below.
    coordination_roles: dict[str, str] = Field(default_factory=dict)

    # Trigger agent/node name -> required follow-up agent/node name. Powers
    # premature_termination; provisional single-hint config, not a general
    # rule DSL (see drift_test's docstring). Empty leaves that detector silent.
    required_followups: dict[str, str] = Field(default_factory=dict)
    # State field name holding a policy/config version (e.g. "policy_version").
    # Powers policy_inconsistency_across_steps. Empty leaves it silent.
    policy_field: str = ""

    # Divergence cascade.
    divergence_mode: str = "exact"
    baseline_rollouts: int = Field(default=1, ge=1, le=10)
    max_judge_calls: int = Field(default=10, ge=0, le=100)
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    # Regression tracking — compare this run against the same graph's
    # history (see drift.regression) and persist this run to that history.
    track_history: bool = False


class AdapterForkRequest(BaseModel):
    """Fork-edit-replay a completed adapter run at a specific step.

    Design spec: docs/design/fork_edit_replay_v1.md. Deferred features live
    in memory/feature_ideas.md; ship v1 (state edit + top-vs-bottom compare)
    only.

    The client passes back the parent's baseline trace + initial state (data
    it already received from /api/adapter-demo). The server rebuilds the
    graph (using the same graph_name + state_overrides) and calls the fork
    API. This avoids a server-side graph cache — trade a redundant graph
    build for statelessness. Cheap.
    """
    # Same fields as AdapterDemoRequest for graph rebuild.
    graph_name: str = "ticket_triage"
    state_overrides: dict[str, Any] = Field(default_factory=dict)

    # Set `module` to fork a run that came from /api/custom-graph-run instead
    # of a bundled graph — takes precedence over graph_name when non-empty.
    module: str = ""
    build_fn: str = "build"
    initial_state_fn: str = "initial_state"
    query: str = ""
    model: str | None = None

    # Parent baseline the client received from /api/adapter-demo.
    parent_baseline_trace: list[dict[str, Any]] = Field(default_factory=list)
    parent_initial_state: dict[str, Any] = Field(default_factory=dict)

    # Fork params.
    fork_step: int = Field(..., ge=1, le=500)
    edits: dict[str, Any] = Field(default_factory=dict)
    also_apply_at_initial: bool = False

    # Optional coordination role overrides for the coord detectors on the
    # forked trace.
    coordination_roles: dict[str, str] = Field(default_factory=dict)


class CustomGraphRequest(BaseModel):
    """Run drift_test against a user-supplied, importable LangGraph module.

    SECURITY: this dynamically imports `module` and calls a function on
    it — arbitrary code execution by design. drift's web server carries no
    auth and is meant for the machine's own user, same trust model as a
    Jupyter kernel or `python -c`. Never expose this server on a network
    you don't fully control.

    `module` must already be importable (on PYTHONPATH / installed) —
    e.g. a target you wrote under `examples/targets/` yourself, or your
    own package. `build_fn` is called as `build_fn(model=...)` if `model`
    is set, else `build_fn()`; must return a compiled graph. `initial_state_fn`
    is called as `initial_state_fn(query)` if `query` is set, else
    `initial_state_fn()`; must return a dict.
    """
    module: str
    build_fn: str = "build"
    initial_state_fn: str = "initial_state"
    query: str = ""
    model: str | None = None

    intensity: str = "light"
    seed: int = 7
    max_perturbations: int = Field(default=6, ge=1, le=500)
    auto_chaos_exclude: list[str] = Field(default_factory=list)

    judge: str = "off"
    judge_model: str | None = None
    user_guidelines: list[str] = Field(default_factory=list)

    coordination_roles: dict[str, str] = Field(default_factory=dict)
    required_followups: dict[str, str] = Field(default_factory=dict)
    policy_field: str = ""

    divergence_mode: str = "exact"
    baseline_rollouts: int = Field(default=1, ge=1, le=10)
    max_judge_calls: int = Field(default=10, ge=0, le=100)
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    track_history: bool = False


class SuiteRunRequest(BaseModel):
    """Run drift against several examples/targets/ registry entries in
    one pass. Backs the Test Suite tab.

    Each target runs independently with the SAME chaos/judge config —
    this is a breadth pass across topologies, not a per-target tuning
    surface. One target failing to build/run doesn't stop the others;
    per-target failures are reported in the response, not raised.
    """
    target_names: list[str] = Field(..., min_length=1)
    # Optional per-target query override (target NAME -> query string);
    # targets not listed here use their own DEFAULT_QUERY.
    query_overrides: dict[str, str] = Field(default_factory=dict)

    intensity: str = "light"
    seed: int = 7
    max_perturbations: int = Field(default=6, ge=1, le=500)

    judge: str = "off"
    judge_model: str | None = None

    track_history: bool = False


class ProposeTargetRequest(BaseModel):
    """Ask an LLM to draft a topology + agent roster for a free-text
    scenario. Backs the Build tab's "Suggest agents" step — the result
    is a DRAFT the user reviews/edits before /api/build-target ever
    runs; this endpoint never generates or saves anything."""
    scenario: str
    model: str = "gpt-4o-mini"


class SimulateScenarioRequest(BaseModel):
    """Propose a roster, optionally ground it in real historical
    precedent, generate a verdict-producing target, and actually run it
    — one call, ephemeral (nothing saved to disk; pass the returned
    `code` to /api/save-target if you want to keep it). Backs the
    Historical Grounding / "Simulate" panel."""
    scenario: str
    ground: bool = True
    model: str = "gpt-4o-mini"


class CalibrateRequest(BaseModel):
    """Score a verdict-producing target against the real held-out
    historical event dataset (drift.historical_grounding). Defaults to
    a fixed stock-reaction roster so repeated runs are comparable; pass
    your own topology/agents to calibrate something else, as long as it
    makes sense against this dataset's up/down/flat vocabulary."""
    topology: str = "fanout"
    agents: list[dict[str, str]] | None = None
    model: str = "gpt-4o-mini"
    ground: bool = True


class BuildTargetRequest(BaseModel):
    """Generate a runnable LangGraph target from a topology + agent list.
    Backs the Build tab. See drift.target_builder for what's actually
    generated and why routing is fixed except for the loop topology."""
    topology: str
    agents: list[dict[str, str]] = Field(..., min_length=1)  # [{"name": ..., "role": ...}, ...]
    model: str = "gpt-4o-mini"


class SaveTargetRequest(BaseModel):
    """Write a Build-tab-generated (or hand-edited) target to disk under
    examples/targets/user_generated/ so it's importable via the
    Import-graph tab. `code` is saved verbatim — the Build tab's code
    preview is editable, so this is not necessarily what /api/build-target
    returned."""
    name: str
    code: str


class SaveRunRequest(BaseModel):
    """Persist a run's full JSON to disk under results/ so it can be
    reloaded later by Fork & Replay / Ask / Compare.

    `data` is the exact object the client got back from /api/adapter-demo
    or /api/custom-graph-run — saved verbatim, not re-shaped server-side.
    """
    data: dict[str, Any]
    group: str = "web_saved"
    name: str = ""


class TraceAskRequest(BaseModel):
    """Ask a free-text question about one already-loaded run.

    The data comes from the client (a run it already has loaded — from the
    Test tab or the Results browser), not re-fetched server-side, so this
    endpoint has no notion of "current run."

    Most saved runs under results/ predate full-trace capture and only
    recorded summary stats (step counts, agent names, finding counts) —
    not the actual per-step trace. Rather than refuse those files, `trace`
    is optional: when empty, `summary` (the raw loaded JSON, whatever
    shape it is) is used instead, and the judge is told explicitly it's
    working from summary data only so it doesn't fabricate step-level
    detail it was never shown.
    """
    trace: list[dict[str, Any]] = Field(default_factory=list)
    initial_state: dict[str, Any] = Field(default_factory=dict)
    final_state: dict[str, Any] | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    question: str
    judge: str = "openai"
    judge_model: str | None = None


class ReportRequest(BaseModel):
    """Render a saved run under results/ into the HTML evidence pack.

    `path` and `fork_path` are relative to results/; both are validated
    server-side against directory escape.
    """
    path: str
    fork_path: str = ""
    title: str = ""
    target: str = ""


# ---- adapter graph registry -----------------------------------------------

def _list_adapter_graphs() -> list[dict[str, Any]]:
    """Return user-facing metadata for every graph the Adapter tab can run.

    Includes an `available` flag so the UI can grey out options whose
    deps / env vars aren't satisfied.
    """
    import os

    sup_pkg_ok = True
    try:
        import langgraph_supervisor  # noqa: F401
        import langchain_openai  # noqa: F401
    except ImportError:
        sup_pkg_ok = False
    sup_key_ok = bool(os.environ.get("OPENAI_API_KEY"))

    return [
        {
            "name": "ticket_triage",
            "label": "Ticket Triage (3-node demo)",
            "description": (
                "Bundled 3-node demo: classify -> (escalate | respond). "
                "Both terminal nodes look up open_tickets[ticket_id] without "
                "a defensive check. Free, deterministic, always available."
            ),
            "agents": ["classify", "escalate", "respond"],
            "available": True,
            "unavailable_reason": "",
            "supports_query_override": True,
            "query_field_label": "Ticket text",
            "query_default": "site is down can someone help",
            "needs_openai": False,
        },
        {
            "name": "langgraph_supervisor",
            "label": "LangGraph Supervisor (math + research)",
            "description": (
                "Canonical math+research supervisor from langchain-ai/"
                "langgraph-supervisor-py's README. Supervisor delegates to "
                "math_expert (add/multiply) or research_expert (mocked search). "
                "Each run = ~5-15 OpenAI calls per perturbation."
            ),
            "agents": ["supervisor", "math_expert", "research_expert"],
            "available": sup_pkg_ok and sup_key_ok,
            "unavailable_reason": (
                "" if (sup_pkg_ok and sup_key_ok)
                else (
                    "missing OPENAI_API_KEY" if sup_pkg_ok
                    else "install with: pip install drift[validation]"
                )
            ),
            "supports_query_override": True,
            "query_field_label": "User message",
            "query_default": "What is 7 times 8?",
            "needs_openai": True,
        },
    ]


def _build_demo_graph(name: str, overrides: dict[str, Any]) -> tuple[Any, dict, dict[str, Any]]:
    if name == "ticket_triage":
        return _build_ticket_triage_demo(overrides)
    if name == "langgraph_supervisor":
        return _build_langgraph_supervisor_demo(overrides)
    raise ValueError(f"unknown graph_name {name!r}; see /api/adapter-graphs")


def _load_generated_module(code: str, label: str) -> Any:
    """Load freshly-generated target source (from target_builder) as a
    real importable module — never a bare exec(code, {}) into a
    throwaway dict.

    Bare exec() intermittently breaks: the generated module's State
    TypedDict uses Annotated[...] field reducers, and LangGraph's
    get_type_hints() call resolves those against the class's registered
    module in sys.modules. A throwaway dict exec'd repeatedly within one
    long-running process (this server, across many requests) doesn't
    reliably give each generated "State" class a distinct, correctly
    resolvable module identity — confirmed by reproducing a real
    NameError: name 'Annotated' is not defined that didn't reproduce in
    a single fresh interpreter process, only under repeated exec() calls
    in one running server. Writing to a real file and importing via
    importlib (same mechanism _build_custom_graph already uses for saved
    targets) sidesteps this entirely: each module gets a real,
    sys.modules-registered identity.

    `label` only affects the ephemeral module's generated name/filename
    (for debugging a traceback) — these files accumulate under the OS
    temp dir and are not meant to be curated; nothing reads them back by
    name.
    """
    import importlib.util
    import tempfile
    import uuid

    tmp_dir = Path(tempfile.gettempdir()) / "drift_ephemeral_targets"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mod_name = f"drift_ephemeral_{label}_{uuid.uuid4().hex}"
    path = tmp_dir / f"{mod_name}.py"
    path.write_text(code, encoding="utf-8")

    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _build_custom_graph(
    module: str, build_fn_name: str, initial_state_fn_name: str,
    query: str, model: str | None,
) -> tuple[Any, dict]:
    """Dynamically import `module` and call its build()/initial_state().

    Shared by /api/custom-graph-run and /api/adapter-fork (when forking a
    run that originated from a custom import rather than a bundled graph).
    See CustomGraphRequest's docstring for the arbitrary-code-execution note.
    """
    import importlib

    try:
        mod = importlib.import_module(module)
    except Exception as e:
        raise ImportError(f"could not import {module!r}: {type(e).__name__}: {e}") from e

    build_fn = getattr(mod, build_fn_name, None)
    if not callable(build_fn):
        raise AttributeError(f"{module}.{build_fn_name} is not callable")
    state_fn = getattr(mod, initial_state_fn_name, None)

    try:
        graph = build_fn(model=model) if model else build_fn()
    except Exception as e:
        raise RuntimeError(f"{build_fn_name}() failed: {type(e).__name__}: {e}") from e

    try:
        initial_state = state_fn(query) if (callable(state_fn) and query) else (
            state_fn() if callable(state_fn) else {}
        )
        initial_state = dict(initial_state)
    except Exception as e:
        raise RuntimeError(f"{initial_state_fn_name}() failed: {type(e).__name__}: {e}") from e

    return graph, initial_state


def _build_ticket_triage_demo(overrides: dict[str, Any]) -> tuple[Any, dict, dict[str, Any]]:
    """The bundled 3-node demo."""

    def _classify(state: dict) -> dict:
        text = (state.get("text") or "").lower()
        if "urgent" in text or "down" in text:
            priority = "high"
        elif "question" in text:
            priority = "low"
        else:
            priority = "normal"
        return {"priority": priority}

    def _escalate(state: dict) -> dict:
        tid = state["ticket_id"]
        ticket = state["open_tickets"][tid]
        return {
            "reply": f"escalated ticket {tid} ({ticket['issue']}) to on-call",
            "escalated": True,
        }

    def _respond(state: dict) -> dict:
        tid = state["ticket_id"]
        ticket = state["open_tickets"][tid]
        return {
            "reply": f"resolved ticket {tid}: {ticket['issue']}",
            "escalated": False,
        }

    class _DemoGraph:
        def stream(self, state: dict):
            merged = dict(state)
            u1 = _classify(merged)
            merged.update(u1)
            yield {"classify": u1}
            if merged.get("priority") == "high" or merged.get("is_premium"):
                u2 = _escalate(merged)
                merged.update(u2)
                yield {"escalate": u2}
            else:
                u2 = _respond(merged)
                merged.update(u2)
                yield {"respond": u2}

        def invoke(self, state: dict) -> dict:
            out = dict(state)
            for chunk in self.stream(state):
                for upd in chunk.values():
                    out.update(upd)
            return out

    text = str(overrides.get("query") or "site is down can someone help")
    initial_state = {
        "ticket_id": "TKT-42",
        "text": text,
        "is_premium": True,
        "open_tickets": {
            "TKT-42": {"issue": "checkout 500s", "customer": "acme"},
        },
        "reply": "",
        "escalated": False,
        "priority": "",
    }
    meta = {
        "name": "ticket_triage",
        "description": (
            "3-node ticket triage: classify -> (escalate | respond). "
            "Both terminal nodes look up open_tickets[ticket_id] without "
            "a defensive check — production-realistic and brittle."
        ),
        "agents": ["classify", "escalate", "respond"],
    }
    return _DemoGraph(), initial_state, meta


def _build_langgraph_supervisor_demo(overrides: dict[str, Any]) -> tuple[Any, dict, dict[str, Any]]:
    """Canonical langgraph-supervisor README example."""
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "langgraph_supervisor graph requires OPENAI_API_KEY in environment "
            "or .env. Pick a different graph or set the key and restart the server."
        )
    try:
        from langchain_openai import ChatOpenAI  # noqa: F401
        from langgraph.prebuilt import create_react_agent  # noqa: F401
        from langgraph_supervisor import create_supervisor  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"langgraph_supervisor graph requires extra packages; install with "
            f"`pip install drift[validation]`. Underlying error: {e}"
        )

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*langgraph.*")
        from pathlib import Path
        import sys as _sys
        _harness_dir = PROJECT_ROOT / "examples" / "adapters"
        if str(_harness_dir) not in _sys.path:
            _sys.path.insert(0, str(_harness_dir))
        from run_drift_on_langgraph_supervisor import (  # type: ignore
            _build_supervisor_mas,
            _initial_state,
        )
        graph = _build_supervisor_mas(model_name="gpt-4o-mini")

    question = str(overrides.get("query") or "What is 7 times 8?")
    initial_state = _initial_state(question)
    meta = {
        "name": "langgraph_supervisor",
        "description": (
            "Canonical math+research supervisor from langgraph-supervisor-py's "
            "README. Supervisor delegates to math_expert (add/multiply) or "
            "research_expert (mocked search). Real OpenAI calls each step."
        ),
        "agents": ["supervisor", "math_expert", "research_expert"],
    }
    return graph, initial_state, meta


def _file_has_trace(path: Path) -> bool:
    """Cheap check: does this saved result carry a full per-step trace?

    Most historical CLI-script runs saved only `baseline_trace_steps`
    (an int) rather than `baseline.trace` (the actual list) — see
    docs note in results_index(). Returns False on any parse error rather
    than raising, since this backs a UI filter, not a hard guarantee.
    """
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    baseline = d.get("baseline") if isinstance(d, dict) else None
    if not isinstance(baseline, dict):
        return False
    trace = baseline.get("trace")
    return isinstance(trace, list) and len(trace) > 0


# ---- MAST helpers ---------------------------------------------------------

def _shape_mast_response(entry: dict[str, Any], result: dict[str, Any], *, mode: str) -> dict[str, Any]:
    """Format a MAST per-trace result for the UI."""
    per_mode = result.get("per_mode", [])
    n_tp = sum(1 for m in per_mode if m["outcome"] == "TP")
    n_fp = sum(1 for m in per_mode if m["outcome"] == "FP")
    n_fn = sum(1 for m in per_mode if m["outcome"] == "FN")
    n_tn = sum(1 for m in per_mode if m["outcome"] == "TN")
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else None
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None

    return {
        "demo_meta": entry,
        "mode": mode,
        "trace_id": result.get("trace_id"),
        "mas_name": result.get("mas_name"),
        "benchmark_name": result.get("benchmark_name"),
        "n_chars": result.get("n_chars"),
        "truncated": result.get("truncated", False),
        "latency_s": result.get("latency_s"),
        "predictions": result.get("predictions", []),
        "per_mode": per_mode,
        "summary": {
            "n_tp": n_tp, "n_fp": n_fp, "n_fn": n_fn, "n_tn": n_tn,
            "n_ground_truth_positives": n_tp + n_fn,
            "n_predicted_positives": n_tp + n_fp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
    }


# ---- shared adapter-result shaping ----------------------------------------
# Used by both /api/adapter-demo (bundled graphs) and /api/custom-graph-run
# (user-imported graphs) so the two response shapes never drift apart —
# the frontend renders both with the same code.


def _shape_adapter_result(
    result: Any,
    *,
    graph_meta: dict[str, Any],
    seed: int,
    judge: str,
    judge_model: str | None,
    user_guidelines: list[str],
    track_history: bool = False,
) -> dict[str, Any]:
    def _divergence_dict(d: Any) -> dict[str, Any]:
        return {
            "name": d.name,
            "tier": d.tier,
            "baseline_value": d.baseline_value,
            "perturbed_value": d.perturbed_value,
            "summary": d.summary,
            "similarity_score": d.similarity_score,
            "within_noise_band": d.within_noise_band,
            "judge_equivalent": d.judge_equivalent,
            "judge_reasoning": d.judge_reasoning,
        }

    def _baseline_dict(b: Any) -> dict[str, Any]:
        return {
            "initial_state": b.initial_state,
            "final_state": b.final_state,
            "crashed": b.crashed,
            "error": b.error,
            "error_type": b.error_type,
            "duration_s": round(b.duration_s, 4),
            "trace": b.trace,
            "judge_findings": b.judge_findings,
            "coordination_findings": b.coordination_findings,
        }

    def _pert_dict(p: Any) -> dict[str, Any]:
        return {
            "event_name": p.event_name,
            "event_summary": p.event_summary,
            "perturbed_field": p.perturbed_field,
            "pattern_type": p.pattern_type,
            "perturbed_initial_state": p.perturbed_initial_state,
            "final_state": p.final_state,
            "crashed": p.crashed,
            "error": p.error,
            "error_type": p.error_type,
            "diverged": p.diverged,
            "divergence_summary": p.divergence_summary,
            "duration_s": round(p.duration_s, 4),
            "trace": p.trace,
            "judge_findings": p.judge_findings,
            "coordination_findings": p.coordination_findings,
            "divergence_details": [_divergence_dict(d) for d in p.divergence_details],
            "filtered_divergences": [_divergence_dict(d) for d in p.filtered_divergences],
        }

    noise_band_dump = {
        name: {
            "name": band.name,
            "sample_count": band.sample_count,
            "distinct_values": band.distinct_values,
            "value_frequencies": band.value_frequencies,
            "text_min_similarity": band.text_min_similarity,
            "text_mean_similarity": band.text_mean_similarity,
            "numeric_min": band.numeric_min,
            "numeric_max": band.numeric_max,
        }
        for name, band in result.noise_band.items()
    }

    regression_report = None
    if track_history:
        from drift import regression

        regression_report = regression.track(
            result, graph_meta["name"], history_dir=HISTORY_DIR
        ).to_json()

    return {
        "graph_name": graph_meta["name"],
        "graph_description": graph_meta["description"],
        "graph_agents": graph_meta.get("agents", []),
        "intensity": result.intensity,
        "seed": seed,
        "judge": judge,
        "judge_model": judge_model,
        "n_user_guidelines": len(user_guidelines or []),
        "divergence_mode": result.divergence_mode,
        "baseline_rollouts": result.baseline_rollouts,
        "judge_calls_used": result.judge_calls_used,
        "judge_calls_budget": result.judge_calls_budget,
        "noise_band": noise_band_dump,
        "patterns_total": result.patterns_total,
        "n_crashed": result.n_crashed,
        "n_diverged": result.n_diverged,
        "n_unchanged": result.n_unchanged,
        "n_judge_findings": result.n_judge_findings,
        "n_coordination_findings": result.n_coordination_findings,
        "n_filtered_divergences": result.n_filtered_divergences,
        "summary_lines": result.summary_lines(),
        "baseline": _baseline_dict(result.baseline),
        "perturbations": [_pert_dict(p) for p in result.perturbations],
        "regression": regression_report,
    }


# ---- trace Q&A helpers ------------------------------------------------------


def _render_trace_for_ask(
    trace: list[dict[str, Any]],
    initial_state: dict[str, Any],
    final_state: dict[str, Any] | None,
    max_chars: int = 12_000,
) -> str:
    """Compact text rendering of one trace for an LLM Q&A prompt.

    Not the same renderer the coordination judge uses (`_render_window` in
    failures/judge.py) — that one windows by timestep for repeated calls
    during a run; this renders the WHOLE trace once for a single ad hoc
    question, so truncation (not windowing) is the right tradeoff here.
    """
    lines = ["# Initial state", json.dumps(initial_state, default=str)[:2000], "", "# Trace"]
    for step in trace:
        node = step.get("node", "?")
        upd = step.get("update") or {}
        lines.append(f"[step {step.get('step')}] {node}: {json.dumps(upd, default=str)[:800]}")
    if final_state:
        lines.append("")
        lines.append("# Final state")
        lines.append(json.dumps(final_state, default=str)[:2000])
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated — trace too long to show in full]"
    return text


def _render_summary_for_ask(summary: dict[str, Any], max_chars: int = 12_000) -> str:
    """Render a whole loaded result file for Q&A when it has no per-step
    trace — most of results/ predates trace capture and only saved counts,
    agent names, and finding lists. Dumps the raw JSON rather than trying
    to normalize every historical shape; the system prompt tells the judge
    plainly this is summary data, not a trace, so it answers accordingly.
    """
    text = "# Run summary (no per-step trace was saved for this run)\n" + json.dumps(
        summary, default=str, indent=2
    )
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated — summary too large to show in full]"
    return text


def _extract_json_answer(raw: str) -> str:
    """Pull `answer` out of the judge's forced-JSON response; fall back to
    the raw text if it isn't valid JSON (degrade gracefully, never 500)."""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "answer" in obj:
            return str(obj["answer"])
    except (json.JSONDecodeError, TypeError):
        pass
    return raw


# ---- app ------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(title="drift", version="0.2.0")

    # ---- LangGraph adapter demo (Adapter tab) -----------------------------

    @app.get("/api/adapter-graphs")
    def adapter_graphs() -> dict[str, Any]:
        """List which bundled graphs the Adapter tab can run, with their
        descriptions, default state shape, and availability (deps + API key)."""
        return {"graphs": _list_adapter_graphs()}

    @app.get("/api/targets")
    def list_targets() -> dict[str, Any]:
        """List the examples/targets/ registry — real third-party
        multi-agent systems drift can run against, with live
        dependency/availability status. Backs the Test Suite tab."""
        from targets import registry as target_registry

        return {"targets": target_registry()}

    @app.post("/api/propose-target")
    async def propose_target_endpoint(req: ProposeTargetRequest) -> dict[str, Any]:
        """Ask an LLM to draft a topology + agent roster for a free-text
        scenario. Backs the Build tab's "Suggest agents" step. Returns a
        draft only — writes nothing, generates no code."""
        from drift.target_builder import ScenarioProposalError, propose_target

        try:
            proposal = await propose_target(req.scenario, model=req.model)
        except ScenarioProposalError as e:
            raise HTTPException(400, str(e))
        return proposal.to_json()

    @app.get("/api/historical-events")
    def list_historical_events() -> dict[str, Any]:
        """The real, verified seed dataset (drift.historical_grounding) —
        what retrieval-grounding draws analogues from and what
        /api/calibrate scores against."""
        from drift.historical_grounding import load_events

        return {"events": [e.to_json() for e in load_events()]}

    @app.post("/api/simulate-scenario")
    async def simulate_scenario(req: SimulateScenarioRequest) -> dict[str, Any]:
        """Propose a roster for a scenario, optionally ground it in real
        historical precedent, generate a verdict-producing target, and
        run it once. Ephemeral — nothing is saved to disk; the returned
        `code` can be passed to /api/save-target if the user wants to
        keep it.
        """
        from drift.historical_grounding import (
            STOCK_VERDICT_SCHEMA,
            GroundingError,
            build_grounded_query,
            retrieve_similar_events,
        )
        from drift.target_builder import (
            ScenarioProposalError,
            TargetBuilderError,
            generate_target_source,
            propose_target,
        )

        try:
            proposal = await propose_target(req.scenario, model=req.model)
        except ScenarioProposalError as e:
            raise HTTPException(400, str(e))

        analogues = []
        if req.ground:
            try:
                analogues = await retrieve_similar_events(req.scenario, model=req.model)
            except GroundingError as e:
                raise HTTPException(400, f"grounding failed: {e}")

        query = build_grounded_query(req.scenario, analogues) if req.ground else req.scenario

        try:
            code = generate_target_source(
                proposal.topology, proposal.agents, model=req.model,
                verdict_schema=STOCK_VERDICT_SCHEMA,
            )
        except TargetBuilderError as e:
            raise HTTPException(400, str(e))

        try:
            module = _load_generated_module(code, "simulate")
            graph = module.build(req.model)
            state = module.initial_state(query)
            final_state = await graph.ainvoke(state)
        except Exception as e:
            raise HTTPException(502, f"simulation run failed: {type(e).__name__}: {e}")

        return {
            "topology": proposal.topology,
            "rationale": proposal.rationale,
            "verdict": final_state.get("verdict", {}),
            "notes": final_state.get("notes", {}),
            "grounded_on": [a.to_json() for a in analogues],
            "code": code,
        }

    @app.post("/api/calibrate")
    async def calibrate_endpoint(req: CalibrateRequest) -> dict[str, Any]:
        """Score a verdict-producing target against the real held-out
        historical event dataset. Makes real LLM calls — one full
        simulation run per event (several calls each) plus one
        retrieval call per event if grounding is on; realistically
        20-35 calls for the full 7-event seed dataset. Not fast, not free.
        """
        from drift.calibration import CalibrationError, run_calibration
        from drift.historical_grounding import (
            DEFAULT_STOCK_REACTION_AGENTS,
            DEFAULT_STOCK_REACTION_TOPOLOGY,
            STOCK_VERDICT_SCHEMA,
        )
        from drift.target_builder import TargetBuilderError, generate_target_source

        agents = req.agents if req.agents is not None else list(DEFAULT_STOCK_REACTION_AGENTS)
        topology = req.topology or DEFAULT_STOCK_REACTION_TOPOLOGY

        try:
            code = generate_target_source(
                topology, agents, model=req.model, verdict_schema=STOCK_VERDICT_SCHEMA,
            )
        except TargetBuilderError as e:
            raise HTTPException(400, str(e))

        try:
            module = _load_generated_module(code, "calibrate")
        except Exception as e:
            raise HTTPException(502, f"generated target failed to load: {type(e).__name__}: {e}")

        try:
            report = await run_calibration(
                module.build, module.initial_state, model=req.model, ground=req.ground,
            )
        except CalibrationError as e:
            raise HTTPException(400, str(e))

        result = report.to_json()
        result["code"] = code
        return result

    @app.post("/api/build-target")
    def build_target(req: BuildTargetRequest) -> dict[str, Any]:
        """Generate a runnable LangGraph target from a topology + agent
        list. Backs the Build tab's "Generate" step — returns source code
        only, writes nothing to disk (see /api/save-target for that)."""
        from drift.target_builder import TargetBuilderError, generate_target_source

        try:
            code = generate_target_source(req.topology, req.agents, model=req.model)
        except TargetBuilderError as e:
            raise HTTPException(400, str(e))
        return {"code": code}

    @app.post("/api/save-target")
    def save_target(req: SaveTargetRequest) -> dict[str, Any]:
        """Write a (possibly hand-edited) generated target to disk under
        examples/targets/user_generated/, so it becomes importable via the
        Import-graph tab. Refuses to save code that doesn't even parse."""
        from drift.target_builder import TargetBuilderError, validate_target_name

        try:
            safe_name = validate_target_name(req.name)
        except TargetBuilderError as e:
            raise HTTPException(400, str(e))

        try:
            compile(req.code, f"<{safe_name}>", "exec")
        except SyntaxError as e:
            raise HTTPException(400, f"generated code has a syntax error: {e}")

        USER_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        init_file = USER_GENERATED_DIR / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")

        out_path = USER_GENERATED_DIR / f"{safe_name}.py"
        out_path.write_text(req.code, encoding="utf-8")

        try:
            display_path = str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            # USER_GENERATED_DIR isn't under PROJECT_ROOT — never true in
            # production (it's defined as PROJECT_ROOT / ...), but tests
            # monkeypatch it to an isolated tmp dir, so degrade gracefully
            # rather than let a display-only field crash the request.
            display_path = str(out_path)

        return {
            "module": f"targets.user_generated.{safe_name}",
            "path": display_path,
        }

    @app.get("/api/results")
    def results_index(with_trace_flag: bool = False) -> dict[str, Any]:
        """List every saved experiment JSON under results/, grouped by
        experiment subdirectory. Powers the Results browser in the UI.

        `with_trace_flag=true` additionally opens each file to check whether
        it saved a full per-step trace (`baseline.trace` as a non-empty
        list) — most older CLI-script runs only saved a step COUNT, so this
        lets the Fork & Replay / Ask tabs filter to files that are actually
        usable instead of offering hundreds of dead ends. Off by default
        because it means parsing every file, not just stat()-ing it.
        """
        groups: dict[str, list[dict[str, Any]]] = {}
        if RESULTS_DIR.exists():
            for path in sorted(RESULTS_DIR.rglob("*.json")):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                rel = path.relative_to(RESULTS_DIR)
                group = rel.parts[0] if len(rel.parts) > 1 else "(root)"
                entry = {
                    "name": rel.name,
                    "path": str(rel).replace("\\", "/"),
                    "size_bytes": stat.st_size,
                    "modified_ts": stat.st_mtime,
                }
                if with_trace_flag:
                    entry["has_trace"] = _file_has_trace(path)
                groups.setdefault(group, []).append(entry)
        for entries in groups.values():
            entries.sort(key=lambda e: e["modified_ts"], reverse=True)
        return {"groups": groups}

    @app.post("/api/save-run")
    def save_run(req: SaveRunRequest) -> dict[str, Any]:
        """Persist a run's full JSON (as returned by /api/adapter-demo or
        /api/custom-graph-run) under results/, so it becomes loadable later
        by Fork & Replay / Ask / Compare — those need the full per-step
        trace this endpoint's caller already has in the browser, which
        nothing else currently writes to disk.
        """
        import re
        import time

        safe_group = re.sub(r"[^A-Za-z0-9_-]+", "_", req.group or "web_saved").strip("_") or "web_saved"
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", req.name or "").strip("_")
        if not safe_name:
            safe_name = f"run_{int(time.time())}"
        if not safe_name.endswith(".json"):
            safe_name += ".json"

        out_dir = (RESULTS_DIR / safe_group).resolve()
        if not str(out_dir).startswith(str(RESULTS_DIR.resolve())):
            raise HTTPException(400, "invalid group")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / safe_name

        out_path.write_text(json.dumps(req.data, indent=2, default=str), encoding="utf-8")
        rel = out_path.relative_to(RESULTS_DIR)
        return {"path": str(rel).replace("\\", "/")}

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        """Aggregate every saved run under results/ into one dashboard payload.

        Walks each experiment JSON, normalizes it through report.load_run
        (which understands both runner shapes), and rolls the per-run
        summaries up into portfolio-level counts: systems tested, runs
        executed, perturbations applied, and every finding surfaced with a
        pointer back to its source file.

        Files that don't parse as a drift run (fork proofs, ad-hoc dumps)
        are skipped silently rather than failing the whole view.
        """
        from drift.report import NotADriftRun, load_run

        runs: list[dict[str, Any]] = []
        findings: list[dict[str, Any]] = []
        totals = {
            "runs": 0, "perturbations": 0, "crashed": 0,
            "diverged": 0, "unchanged": 0, "findings": 0,
        }
        agents: set[str] = set()

        if RESULTS_DIR.exists():
            for path in sorted(RESULTS_DIR.rglob("*.json")):
                rel = str(path.relative_to(RESULTS_DIR)).replace("\\", "/")
                try:
                    run = load_run(path)
                except (NotADriftRun, json.JSONDecodeError, OSError):
                    continue
                s = run["summary"]
                group = rel.split("/")[0] if "/" in rel else "(root)"
                n_find = s["n_judge_findings"] + s["n_coord_findings"]
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    mtime = 0.0

                runs.append({
                    "path": rel,
                    "group": group,
                    "name": path.name,
                    "modified_ts": mtime,
                    "model": run["config"].get("model", ""),
                    "intensity": run["config"].get("intensity", ""),
                    "elapsed_seconds": run.get("elapsed_seconds"),
                    "n_scenarios": s["n_scenarios"],
                    "n_runs": s["n_runs"],
                    "n_perturbations": s["n_perturbations"],
                    "n_crashed": s["n_crashed"],
                    "n_diverged": s["n_diverged"],
                    "n_findings": n_find,
                    "agents": s["agents"],
                })
                totals["runs"] += s["n_runs"]
                totals["perturbations"] += s["n_perturbations"]
                totals["crashed"] += s["n_crashed"]
                totals["diverged"] += s["n_diverged"]
                totals["unchanged"] += s["n_unchanged"]
                totals["findings"] += n_find
                agents.update(s["agents"])

                for f in run["judge_findings"] + run["coord_findings"]:
                    findings.append({
                        "failure_type": f.get("failure_type") or f.get("type") or "finding",
                        "summary": f.get("summary", ""),
                        "where": f.get("where", ""),
                        "source_path": rel,
                        "source_group": group,
                        "modified_ts": mtime,
                    })

        runs.sort(key=lambda r: r["modified_ts"], reverse=True)
        findings.sort(key=lambda f: f["modified_ts"], reverse=True)

        by_group: dict[str, dict[str, Any]] = {}
        for r in runs:
            g = by_group.setdefault(r["group"], {
                "group": r["group"], "n_files": 0, "n_runs": 0,
                "n_perturbations": 0, "n_findings": 0, "agents": set(),
                "last_run_ts": 0.0,
            })
            g["n_files"] += 1
            g["n_runs"] += r["n_runs"]
            g["n_perturbations"] += r["n_perturbations"]
            g["n_findings"] += r["n_findings"]
            g["agents"].update(r["agents"])
            g["last_run_ts"] = max(g["last_run_ts"], r["modified_ts"])
        systems = sorted(
            (
                {**g, "agents": sorted(g["agents"])}
                for g in by_group.values()
            ),
            key=lambda g: g["last_run_ts"],
            reverse=True,
        )

        type_counts: dict[str, int] = {}
        for f in findings:
            type_counts[f["failure_type"]] = type_counts.get(f["failure_type"], 0) + 1

        return {
            "totals": {**totals, "systems": len(systems), "agents": len(agents)},
            "systems": systems,
            "runs": runs,
            "findings": findings,
            "finding_types": sorted(
                ({"type": k, "count": v} for k, v in type_counts.items()),
                key=lambda d: d["count"], reverse=True,
            ),
        }

    @app.post("/api/report")
    def build_report(req: ReportRequest) -> Any:
        """Render a saved run into the self-contained HTML evidence pack.

        Same renderer the `drift report` CLI uses. Returns the HTML inline so
        the UI can preview it in an iframe or hand it to the user as a
        download without writing anything to disk.
        """
        from drift.report import load_fork_proof, load_run, render_report

        results_root = RESULTS_DIR.resolve()

        def _safe(relpath: str) -> Path:
            try:
                full = (results_root / relpath).resolve()
            except (ValueError, OSError) as e:
                raise HTTPException(400, f"bad path: {e}")
            if not str(full).startswith(str(results_root)):
                raise HTTPException(403, "path escapes results/")
            if not full.exists() or not full.is_file() or full.suffix != ".json":
                raise HTTPException(404, "not found")
            return full

        run_path = _safe(req.path)
        try:
            run = load_run(run_path)
        except Exception as e:
            raise HTTPException(400, f"not a drift run: {e}")

        fork = None
        if req.fork_path:
            try:
                fork = load_fork_proof(_safe(req.fork_path))
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(400, f"bad fork proof: {e}")

        html_text = render_report(
            run, fork=fork, title=req.title or None, target_name=req.target or None,
        )
        return {
            "html": html_text,
            "filename": Path(req.path).stem + ".report.html",
            "summary": run["summary"],
        }

    @app.get("/api/results/{relpath:path}")
    def results_file(relpath: str) -> Any:
        """Serve a single result JSON file by relative path. Restricted to
        results/ and bare *.json to keep the surface boring."""
        results_root = RESULTS_DIR.resolve()
        try:
            full = (results_root / relpath).resolve()
        except (ValueError, OSError) as e:
            raise HTTPException(400, f"bad path: {e}")
        if not str(full).startswith(str(results_root)):
            raise HTTPException(403, "path escapes results/")
        if not full.exists() or not full.is_file():
            raise HTTPException(404, "not found")
        if full.suffix != ".json":
            raise HTTPException(400, "only .json supported")
        try:
            return json.loads(full.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise HTTPException(500, f"could not read: {e}")

    @app.post("/api/adapter-demo")
    async def adapter_demo(req: AdapterDemoRequest) -> dict[str, Any]:
        """Run drift's auto-chaos against a bundled langgraph-shaped graph."""
        from drift.adapters.langgraph import drift_test_async
        from drift.failures.judge import build_judge

        try:
            graph, initial_state, graph_meta = _build_demo_graph(
                req.graph_name, req.state_overrides
            )
        except (ValueError, ImportError, RuntimeError) as e:
            raise HTTPException(400, str(e))

        try:
            judge_llm = build_judge(req.judge, model=req.judge_model)
        except ValueError as e:
            raise HTTPException(400, str(e))

        try:
            result = await drift_test_async(
                graph=graph,
                initial_state=initial_state,
                intensity=req.intensity,
                seed=req.seed,
                auto_chaos_exclude=req.auto_chaos_exclude or None,
                max_perturbations=req.max_perturbations,
                judge_llm=judge_llm,
                user_guidelines=req.user_guidelines or None,
                divergence_mode=req.divergence_mode,
                baseline_rollouts=req.baseline_rollouts,
                max_judge_calls=req.max_judge_calls,
                similarity_threshold=req.similarity_threshold,
                coordination_roles=req.coordination_roles or None,
                required_followups=req.required_followups or None,
                policy_field=req.policy_field or None,
            )
        except (TypeError, ValueError) as e:
            raise HTTPException(400, str(e))

        return _shape_adapter_result(
            result, graph_meta=graph_meta, seed=req.seed,
            judge=req.judge, judge_model=req.judge_model,
            user_guidelines=req.user_guidelines,
            track_history=req.track_history,
        )

    @app.post("/api/custom-graph-run")
    async def custom_graph_run(req: CustomGraphRequest) -> dict[str, Any]:
        """Run drift_test against a user-supplied importable module.

        Backs the "Import graph" tab. See CustomGraphRequest docstring for
        the security note — this executes arbitrary local code by design.
        """
        from drift.adapters.langgraph import drift_test_async
        from drift.failures.judge import build_judge

        try:
            graph, initial_state = _build_custom_graph(
                req.module, req.build_fn, req.initial_state_fn, req.query, req.model,
            )
        except (ImportError, AttributeError, RuntimeError) as e:
            raise HTTPException(400, str(e))

        try:
            judge_llm = build_judge(req.judge, model=req.judge_model)
        except ValueError as e:
            raise HTTPException(400, str(e))

        try:
            result = await drift_test_async(
                graph=graph,
                initial_state=initial_state,
                intensity=req.intensity,
                seed=req.seed,
                auto_chaos_exclude=req.auto_chaos_exclude or None,
                max_perturbations=req.max_perturbations,
                judge_llm=judge_llm,
                user_guidelines=req.user_guidelines or None,
                divergence_mode=req.divergence_mode,
                baseline_rollouts=req.baseline_rollouts,
                max_judge_calls=req.max_judge_calls,
                similarity_threshold=req.similarity_threshold,
                coordination_roles=req.coordination_roles or None,
                required_followups=req.required_followups or None,
                policy_field=req.policy_field or None,
            )
        except (TypeError, ValueError) as e:
            raise HTTPException(400, str(e))

        graph_meta = {
            "name": f"{req.module}.{req.build_fn}",
            "description": f"user-imported graph — {req.module}",
            "agents": [],
        }
        return _shape_adapter_result(
            result, graph_meta=graph_meta, seed=req.seed,
            judge=req.judge, judge_model=req.judge_model,
            user_guidelines=req.user_guidelines,
            track_history=req.track_history,
        )

    @app.post("/api/suite-run")
    async def suite_run(req: SuiteRunRequest) -> dict[str, Any]:
        """Run drift against several examples/targets/ registry entries in
        one pass. Backs the Test Suite tab. Each target is built and run
        independently; a target that's missing dependencies is skipped
        (not attempted), and a target that fails to build or run reports
        its error without stopping the rest of the suite.
        """
        from targets import describe as describe_target, get as get_target

        from drift.adapters.langgraph import drift_test_async
        from drift.failures.judge import build_judge

        try:
            judge_llm = build_judge(req.judge, model=req.judge_model)
        except ValueError as e:
            raise HTTPException(400, str(e))

        results: list[dict[str, Any]] = []
        for name in req.target_names:
            try:
                mod = get_target(name)
            except KeyError as e:
                results.append({
                    "target_name": name, "status": "error", "error": str(e), "result": None,
                })
                continue

            info = describe_target(mod)
            if not info["importable"]:
                results.append({
                    "target_name": name, "status": "skipped",
                    "error": f"missing dependencies: {', '.join(info['missing_deps'])}",
                    "result": None,
                })
                continue

            query = req.query_overrides.get(name) or mod.DEFAULT_QUERY
            try:
                graph, initial_state = _build_custom_graph(
                    info["module"], "build", "initial_state", query, None,
                )
            except (ImportError, AttributeError, RuntimeError) as e:
                results.append({
                    "target_name": name, "status": "error", "error": str(e), "result": None,
                })
                continue

            try:
                result = await drift_test_async(
                    graph=graph,
                    initial_state=initial_state,
                    intensity=req.intensity,
                    seed=req.seed,
                    max_perturbations=req.max_perturbations,
                    judge_llm=judge_llm,
                )
            except (TypeError, ValueError) as e:
                results.append({
                    "target_name": name, "status": "error", "error": str(e), "result": None,
                })
                continue

            graph_meta = {"name": mod.NAME, "description": mod.DESCRIPTION, "agents": []}
            shaped = _shape_adapter_result(
                result, graph_meta=graph_meta, seed=req.seed,
                judge=req.judge, judge_model=req.judge_model,
                user_guidelines=[],
                track_history=req.track_history,
            )
            results.append({
                "target_name": name, "status": "ok", "error": None, "result": shaped,
            })

        return {"results": results}

    @app.post("/api/trace-ask")
    async def trace_ask(req: TraceAskRequest) -> dict[str, Any]:
        """Answer a free-text question about one already-loaded run.

        Works two ways depending on what the loaded file actually saved:
        full per-step trace (rich, step-cited answers) or summary-only
        (most historical results/ files — coarser answers, judge is told
        plainly that's all it has so it doesn't invent step detail).
        """
        from drift.failures.judge import build_judge

        if not req.question.strip():
            raise HTTPException(400, "question is required")
        if not req.trace and not req.summary:
            raise HTTPException(400, "no data to ask about — run or load a run first")

        try:
            judge_llm = build_judge(req.judge, model=req.judge_model)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if judge_llm is None:
            raise HTTPException(400, "judge must be 'openai' or 'mock' to answer questions")

        if req.trace:
            rendered = _render_trace_for_ask(req.trace, req.initial_state, req.final_state)
            system = (
                "You are analyzing ONE real execution trace of a multi-agent system. "
                "Answer the user's question using only evidence in the trace below — "
                "do not speculate about behavior the trace doesn't show. Cite step "
                "numbers and node/agent names where relevant. If the trace doesn't "
                "contain enough information to answer, say so plainly instead of "
                "guessing.\n\n"
                'Respond with strict JSON: {"answer": "<your answer, plain text>"}'
            )
        else:
            rendered = _render_summary_for_ask(req.summary)
            system = (
                "You are analyzing SUMMARY STATISTICS from one multi-agent test run — "
                "not the actual per-step trace, which this file never saved (only "
                "counts, agent names, and finding lists). Answer using only what's "
                "in the summary below. You CANNOT see individual steps, so never "
                "cite a specific step number or claim to know what happened at a "
                "particular point in the run — if the question needs that level of "
                "detail, say plainly that only summary data is available for this "
                "run, not a full trace.\n\n"
                'Respond with strict JSON: {"answer": "<your answer, plain text>"}'
            )
        user = f"{rendered}\n\n# Question\n{req.question.strip()}"
        try:
            raw = await judge_llm.judge(system=system, user=user)
        except RuntimeError as e:
            raise HTTPException(502, str(e))

        return {"answer": _extract_json_answer(raw), "question": req.question}

    @app.post("/api/adapter-fork")
    async def adapter_fork(req: AdapterForkRequest) -> dict[str, Any]:
        """Fork-edit-replay a completed adapter run.

        Design spec: docs/design/fork_edit_replay_v1.md.

        Client sends: graph_name + state_overrides (to rebuild the same
        graph), parent baseline trace + initial state (to fork from),
        fork_step + edits (what to change), and the top-vs-bottom opt-in.
        Server rebuilds the graph, wraps the parent data into the shape
        drift_test_fork_async expects, and runs the fork.
        """
        from drift.adapters.langgraph import (
            AdapterResult,
            BaselineResult,
            drift_test_fork_async,
        )

        try:
            if req.module:
                graph, _initial_ignored = _build_custom_graph(
                    req.module, req.build_fn, req.initial_state_fn, req.query, req.model,
                )
                graph_meta = {"name": f"{req.module}.{req.build_fn}", "description": ""}
            else:
                graph, _initial_ignored, graph_meta = _build_demo_graph(
                    req.graph_name, req.state_overrides
                )
        except (ValueError, ImportError, AttributeError, RuntimeError) as e:
            raise HTTPException(400, str(e))

        # Wrap the parent-baseline snapshot into an AdapterResult shape so
        # drift_test_fork_async can validate + read from it. Only baseline
        # is populated — we don't need the parent's perturbations here.
        parent_shim = AdapterResult(
            baseline=BaselineResult(
                initial_state=dict(req.parent_initial_state),
                trace=list(req.parent_baseline_trace),
            ),
        )

        try:
            fork = await drift_test_fork_async(
                graph=graph,
                parent_result=parent_shim,
                fork_step=req.fork_step,
                edits=req.edits,
                also_apply_at_initial=req.also_apply_at_initial,
                coordination_roles=req.coordination_roles or None,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))

        def _branch_dict(b: Any) -> dict[str, Any]:
            return {
                "initial_state": b.initial_state,
                "trace": b.trace,
                "final_state": b.final_state,
                "crashed": b.crashed,
                "error": b.error,
                "error_type": b.error_type,
                "duration_s": round(b.duration_s, 4),
                "coordination_findings": b.coordination_findings,
            }

        return {
            "graph_name": graph_meta["name"],
            "graph_description": graph_meta["description"],
            "fork_step": fork.fork_step,
            "edits": fork.edits,
            "fork_point_state": fork.fork_point_state,
            "edited_state_at_fork": fork.edited_state_at_fork,
            "fork_branch": _branch_dict(fork.fork_branch),
            "top_edited_branch": (
                _branch_dict(fork.top_edited_branch)
                if fork.top_edited_branch else None
            ),
            "n_coordination_findings": fork.n_coordination_findings,
            "duration_s": round(fork.duration_s, 4),
        }

    # ---- MAST demo --------------------------------------------------------

    @app.get("/api/mast-demos")
    def mast_demos() -> dict[str, Any]:
        """List the curated MAST demo traces with task + outcome metadata."""
        if not MAST_DATASET.exists():
            raise HTTPException(404, f"MAST dataset not bundled at {MAST_DATASET}")
        mast = json.loads(MAST_DATASET.read_text(encoding="utf-8"))
        mast_by_id = {r["trace_id"]: r for r in mast}

        out = []
        for entry in MAST_DEMO_TRACES:
            rec = mast_by_id.get(entry["id"])
            if not rec:
                continue
            cached_path = MAST_CACHED_RESULTS_DIR / f"trace_{entry['id']:03d}.json"
            cached = json.loads(cached_path.read_text(encoding="utf-8")) if cached_path.exists() else None
            gt_modes = []
            if cached:
                gt_modes = [m["name"] for m in cached.get("per_mode", []) if m.get("ground_truth")]
            out.append({
                **entry,
                "trace_chars": len(rec["trace"]),
                "trace_truncated": len(rec["trace"]) > 100_000,
                "benchmark_name": rec.get("benchmark_name"),
                "ground_truth_modes": gt_modes,
                "n_ground_truth_positives": len(gt_modes),
                "has_cached_result": cached is not None,
            })
        return {"traces": out, "judge_model_default": "gpt-4o-mini"}

    @app.post("/api/mast-analyze")
    async def mast_analyze(req: MastAnalyzeRequest) -> dict[str, Any]:
        """Run (or replay cached) drift's judge against one curated MAST trace."""
        entry = next((e for e in MAST_DEMO_TRACES if e["id"] == req.trace_id), None)
        if entry is None:
            raise HTTPException(400, f"trace_id {req.trace_id} is not a curated MAST demo trace")

        if req.mode == "cached":
            cached_path = MAST_CACHED_RESULTS_DIR / f"trace_{req.trace_id:03d}.json"
            if not cached_path.exists():
                raise HTTPException(404, f"no cached result for trace_id {req.trace_id}")
            cached = json.loads(cached_path.read_text(encoding="utf-8"))
            return _shape_mast_response(entry, cached, mode="cached")

        if req.mode == "live":
            from drift.failures.judge import OpenAIJudge
            from drift.failures.mast_eval import judge_one_trace

            mast = json.loads(MAST_DATASET.read_text(encoding="utf-8"))
            rec = next((r for r in mast if r["trace_id"] == req.trace_id), None)
            if rec is None:
                raise HTTPException(404, f"trace_id {req.trace_id} missing from MAST dataset")

            try:
                judge = OpenAIJudge(model=req.judge_model)
            except Exception as e:
                raise HTTPException(400, f"could not build judge: {type(e).__name__}: {e}")

            result = await judge_one_trace(
                rec, judge,
                user_guidelines=req.user_guidelines or None,
            )
            if "error" in result:
                raise HTTPException(500, f"judge call failed: {result['error']}")
            return _shape_mast_response(entry, result, mode="live")

        raise HTTPException(400, f"unknown mode {req.mode!r}; expected 'cached' or 'live'")

    # Static frontend.
    if WEB_DIR.exists():
        app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

        @app.get("/")
        def root() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")
    else:
        @app.get("/")
        def root() -> JSONResponse:
            return JSONResponse({"detail": "web/ directory not found", "api": "/api/adapter-graphs"})

    return app


def serve(host: str = "127.0.0.1", port: int = 8765, reload: bool = False) -> None:
    import uvicorn
    if reload:
        uvicorn.run("drift.server:create_app", host=host, port=port, factory=True, reload=True)
    else:
        uvicorn.run(create_app(), host=host, port=port)
