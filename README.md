# drift

[![tests](https://github.com/SohamAg/drift/actions/workflows/tests.yml/badge.svg)](https://github.com/SohamAg/drift/actions/workflows/tests.yml)

> Pre-deploy chaos testing for LangGraph multi-agent systems. Drift takes a
> compiled graph, perturbs its initial state, captures the per-super-step
> trace, and reports the crashes + silent divergences + named coordination
> failures you wouldn't find with a happy-path test.

What changed and when: [`CHANGELOG.md`](CHANGELOG.md).

---

## Quickstart — run it against a LangGraph app

```python
from drift.adapters.langgraph import drift_test

result = drift_test(
    graph=my_compiled_graph,
    initial_state={"messages": [...], "session_id": "abc", "is_premium": True},
    intensity="aggressive",   # or "exhaustive" for full schema coverage
    divergence_mode="tiered",
    baseline_rollouts=3,
    perturbation_rollouts=3,  # run each perturbed input 3× → "crashed 1/3"
)

for p in result.perturbations:
    if p.crashed:
        print(f"CRASH   {p.event_name}: {p.error_type}: {p.error}")
    elif p.diverged:
        print(f"DIVERGE {p.event_name}: {p.divergence_summary}")
    for f in p.coordination_findings:
        print(f"  COORD [{f['failure_type']}] {f['summary']}")
```

Works against any object exposing `.invoke / .ainvoke / .stream / .astream` —
no hard dependency on `langgraph` itself. Plain callables work too (no trace).

Worked end-to-end demo: [`examples/adapters/langgraph_demo.py`](examples/adapters/langgraph_demo.py)
Same flow wired into the web UI under the **Adapter** tab.

---

### From the command line — and as a CI gate

```bash
pip install drift[langgraph]
drift run my_app.graph:build --preset quick --fail-on crash,finding:injected_instruction_followed
```

`my_app.graph` exposes `build()` → compiled graph and `initial_state()` →
the dict you invoke it with. Add `--json drift-results.json` to write the
same JSON the web UI renders, `--rollouts 3` to run every perturbed input
three times (`crashed 1/3`), and `drift gate drift-results.json --fail-on …`
to decide from the artifact on a box without the graph or the API key.
Exit code 1 on a violated rule. Rules: `crash`, `recursion_limit`,
`diverge`, `intermittent`, `finding`, `finding:<type>`, `baseline_finding`,
`guideline`, `judge`, `max_crash_rate=`, `max_diverge_rate=`. Full guide and
a GitHub Actions workflow in [`docs/ci.md`](docs/ci.md).

### Your own rules — `drift.yml`

```yaml
rules:
  - id: approve-first
    tool_order: {before: approve_refund, after: issue_refund}
  - id: researcher-is-read-only
    forbid_tool: send_*
    agent: researcher
  - id: budget
    state: "budget_remaining >= 0"
  - id: cite-ticket
    final_answer: {must_match: "#\d+"}
```

Deterministic, free, evaluated on the baseline and every perturbation;
`drift run` picks the file up from the repo root and `--fail-on guideline`
gates on it. Nine rule kinds plus `python: pkg.mod:func` /
`@drift.guideline` for the rest. Reference: [`docs/guidelines.md`](docs/guidelines.md).

## What drift actually does

Three concentric layers, each cheaper than the last:

### 1. Chaos engine — schema-walked auto-perturbation
Reads the runtime types of your `initial_state`. For each field, picks
type-appropriate perturbations: `flip_bool`, `corrupt_string`, `clear_list`,
`duplicate_list_entry`, `remove_dict_key`, `inject_fake_dict_key`,
`boundary_numeric`, etc. Dict fields are walked two levels down, so
`config.max_retries` gets its own perturbation. No configuration. Runs each
perturbed state through your graph and buckets the outcomes into
`crashed | diverged | unchanged`; a crash that is LangGraph's recursion
limit firing is labelled `recursion_limit` and keeps its partial trace, so
the loop is visible.

A `messages` channel (langchain objects, role dicts, tuples — anything
`add_messages` accepts) gets six message-aware patterns instead of the
generic list ones, each tagged with its fault class (AgentChaos's
crash / omission / value split):

| Pattern | Fault class | Models |
|---|---|---|
| `drop_last_user_message` | omission | the instruction never arrived |
| `truncate_last_user_message` | omission | the instruction was cut to 30% |
| `garble_last_user_message` | value | 20% of characters corrupted in transit |
| `inject_tool_error_result` | crash | the agent's last tool call failed upstream |
| `inject_tool_result_instruction` | value | indirect prompt injection via a tool result, canary-verified |
| `inject_tool_result_tool_hijack` | value | injection that demands a tool call with a canary argument — the AgentDojo success criterion |
| `inject_fabricated_prior_turn` | value | history claims the task is already done |

**Tool-layer patterns** go further than any initial-state edit: they
fault one call to one tool *while the graph runs*. drift finds every
`ToolNode` in the compiled graph (subgraphs included) and installs
LangGraph's own `wrap_tool_call` interceptor for the duration of one run,
so the second `web_search` call times out, or comes back empty,
truncated, stale, or carrying an injected instruction — and the model has
to cope mid-task. Nine faults (`tool_error`, `tool_timeout`,
`tool_rate_limit`, `tool_exception`, `tool_empty`, `tool_truncated`,
`tool_stale`, `tool_inject_instruction`, `tool_hijack`), named
`tool_timeout[web_search]`, scheduled alongside the state patterns, or
asked for by name in `drift.yml`:

```yaml
chaos:
  - tool: web_search
    fault: tool_timeout
    on_call: 2
```

Reference: [`docs/tool_chaos.md`](docs/tool_chaos.md).

**Mid-run timing.** `drift_test(chaos_timing="mid_run")` (CLI `--timing
mid_run`) applies the state and message patterns to a *checkpoint* of the
baseline run instead of the initial state: drift runs the baseline on a
checkpointer, forks it at the step the planner scheduled each pattern
for, writes the perturbation through `update_state` and resumes. The
               '(injected tool results arrive after the agent&#39;s own tool round-trip), resume. Falls back to '
the dropped user message vanishes from a conversation already under way,
and the trace continues the baseline's step numbering. Anything that
cannot be forked falls back to initial timing and says so. Live, on a
react agent with a cold conversation, the seeded tool hijack landed 0/3
at the initial state and 3/3 mid-run (`results/midrun_injection/`).

Perturbed conversations are built in the same dialect your state uses, so
the graph still accepts them — a drift-induced API 400 would be a fake
crash, not a finding.

Intensity ladder: `off` / `light` (~8%) / `moderate` (~18%) / `aggressive`
(~35%) / `exhaustive`. Exhaustive ignores sampling and walks every
applicable pattern in the schema exactly once — meant for pre-deploy
gates where you want full schema coverage at the cost of one graph run
per fuzzable pattern.

### 2. Tiered divergence cascade
LLM outputs are non-deterministic — naive `baseline ≠ perturbed` produces
all false positives. Drift filters through 4 stages:

- **t0 structural** — key added/removed/type changed (always real)
- **t1 exact** — canonical JSON equality on remaining fields
- **t2 noise band** — does perturbed value fall within natural baseline
  variance (measured over N baseline rollouts)?
- **t3 judge** — LLM equivalence check on survivors (budget-capped)

Trades a few cents per run for ~zero false positives on LLM-driven graphs.
Every drop at t2/t3 is preserved in `PerturbationResult.filtered_divergences`
with its reasoning intact, so UNCHANGED verdicts are auditable — you can
see exactly which diffs the noise band or the judge cleared and why.

**One perturbed run is a sample, not a verdict.** On a live 2-agent
supervisor the planted prompt injection was followed in 1 of 4 runs.
`perturbation_rollouts=N` runs every perturbed input N times: each
`PerturbationResult` is the most severe of its rollouts (so `crashed` reads
"in at least one run"), carries the others in `.rollouts`, and reports
`.outcome_counts` (`crashed 1/3 · unchanged 2/3`), `.finding_rollout_counts`
(detector → how many rollouts it fired in) and `.is_intermittent` when the
rollouts disagree — which is itself the finding. The Thorough and Exhaustive
UI presets use 2 and 3.

### 3. Coordination-failure detector library
Curated, source-cited detectors for documented multi-agent failure modes.
Currently shipped:

| Detector | Source |
|---|---|
| `verifier_always_approves` | MAST 3.x family + Anthropic engineering blog |
| `infinite_handoff` | MAST 1.3 + Cognition open problem #2 |
| `repeated_identical_action` | MAST 1.3 (single-agent) + Anthropic stateful tool-call postmortem |
| `subagent_fanout_excess` | Anthropic 50-subagent incident |
| `hallucinated_reference` | MAST 2.x grounding failures + Anthropic subagent hallucination |
| `contradictory_decisions` | MAST 3.2 + Cognition principle 2 (conflicting decisions carry bad results) |
| `stale_state_reference` | MAST 1.5 parallel-agent state race + Cognition open problem #2 |
| `role_drift` | MAST 1.2 (Disobey Role Specification) |
| `context_loss_on_handoff` | MAST 1.4 + 2.5 (lost history / ignored input) |
| `unacknowledged_error` | Anthropic engineering blog (silently swallowed tool/agent errors) |
| `premature_termination`¹ | MAST 4.1 (termination errors) |
| `policy_inconsistency_across_steps`¹ | MAST 1.5 (configuration drift) |
| `tool_call_loop`² | MAST 1.3 at the tool layer + tool-repetition-rate metric (arXiv:2601.17915) |
| `silent_tool_failure`² | AgentChaos "Silent" outcome class (arXiv:2608.06790) + Anthropic postmortem |
| `injected_instruction_followed`² | AgentDojo indirect-injection protocol + OWASP LLM01; canary-verified |
| `injected_tool_call_followed`² | AgentDojo success criterion (the attacker's tool call executed); canary in tool-call arguments |
| `retry_amplification`² | AgentChaos cost amplification (LLM calls/task up to 4.24× under faults, arXiv:2608.06790) + MAST 1.3 |
| `novel_tool_call`² | AgentDojo (injection success = unauthorised tool call) + ToolEmu + OWASP LLM06; side-effect tools flagged |
| `tainted_side_effect_call`² | CaMeL taint tracking (arXiv:2503.18813) + AgentDojo exfiltration; tool-result text lifted verbatim into a side-effect tool's arguments, no canary needed |

¹ requires a small config hint (`required_followups=` / `policy_field=`
on `drift_test`, or the matching fields in the Test/Import tabs) —
silent without it, by design. See each detector's doc under
`docs/detectors/` for the full config shape.

² message-aware: reads the chat messages nodes write (AI tool calls, tool
results, final answers) rather than top-level state fields, so these see
the tool layer the others can't. `injected_instruction_followed` is the
second half of the `inject_tool_result_instruction` chaos pattern — the
pattern plants a tool result demanding a `DRIFT-CANARY-xxxxxx` phrase, the
detector fires only if a later AI message emits it. Deterministic verdict,
no LLM. `injected_tool_call_followed` is the same contract at the action
layer: `inject_tool_result_tool_hijack` demands a tool call with the canary
as its argument, and the detector fires when a later tool call carries it.

Each detector ships a structured `detect()` (over the adapter trace) plus a
text-only `detect_from_text()` (for MAST-style transcripts) plus a curated
explainer doc under [`docs/detectors/`](docs/detectors/) with source
citations and architecture-level fix guidance — no generated suggestion
text. Free, deterministic, runs alongside every drift_test. **Empirically
validated** on real LangGraph code
(`examples/adapters/run_drift_on_adversarial_mas.py`), not just synthetic
fixtures.

### 4. Fork-edit-replay — counterfactual state edits at any trace step
Once you have a coord finding, `drift_test_fork()` lets you edit the state
at any super-step and re-run the graph forward from there. The forked
branch's trace + coord findings render side-by-side with the parent's, so
you can verify a hypothesized fix without changing your graph code.

```python
from drift.adapters.langgraph import drift_test, drift_test_fork
parent = drift_test(graph=graph, initial_state={...})
fork = drift_test_fork(
    graph=graph,
    parent_result=parent,
    fork_step=1,
    edits={"ticket_id": "TICKET-42"},     # sparse deep-merge into state_after
    also_apply_at_initial=True,            # top-vs-bottom compare (design diagnostic)
)
```

Diagnostic, not therapeutic. If both branches close the finding, the fix is
legitimate at both initial-design AND local level. If only the fork closes
it, there's path-dependence — the fix works locally but wouldn't survive a
redesign upstream. See [`docs/design/fork_edit_replay_v1.md`](docs/design/fork_edit_replay_v1.md)
for the design spec and [`examples/adapters/fork_edit_demo.py`](examples/adapters/fork_edit_demo.py)
for a runnable end-to-end demo.

If the graph was compiled with a checkpointer, the fork is real LangGraph
time travel: drift runs every baseline / perturbation on its own thread,
finds the checkpoint after the fork step, applies the edit with
`update_state` (through the channel reducers — a `messages` edit appends)
and resumes with input `None`, so only the continuation executes.
`ForkResult.execution_note` says which path ran and why; without a
checkpointer, or when the fork step is inside a subgraph, it falls back to
rerunning from the edited state. The web server attaches one process-wide
in-memory saver to every compiled graph that has none and the UI sends the
parent run's thread id back with the fork, so forks clicked in the Test and
Import tabs take the checkpoint path too — each branch says
"resumed a real checkpoint" or "rerun from edited state".

### Plus — optional LLM judge over the full trace
Six-category taxonomy (`coordination_contradiction`, `grounding_failure`,
`state_drift`, `emergent_decay`, `gate_bypass`, `user_guideline`) applied to
the per-super-step trace. Catches semantic coordination issues a single-step
detector can't see. Add custom rules in plain English via `user_guidelines=`.

### 5. Regression tracking — over time, not just one-shot
Everything above answers "is this graph robust right now." Regression
tracking answers the question that actually catches a bad prompt edit
before it ships: "did the LAST change make it worse?"

```python
from drift.adapters.langgraph import drift_test
from drift import regression

result = drift_test(graph=my_compiled_graph, initial_state={...})
report = regression.track(result, graph_signature="my_graph")

if report.is_regression:
    for flag in report.flags:
        print(f"[{flag.severity}] {flag.summary}")
```

`track()` condenses the run into a small snapshot (never the full trace),
appends it to a per-graph history file under `results/_history/`, and diffs
it against the most recent prior run for that same `graph_signature`: new
coordination-finding types, finding types that stopped firing, crash/
divergence rate jumps, and perturbation runtime jumps. The web UI exposes
this as a **"Track for regression"** checkbox on the Test and Import-graph
tabs; `drift history <graph_signature>` replays the same comparison from
the terminal without re-running anything.

### 6. Evidence packs — `drift report`
A run's raw JSON is not what a reviewer, auditor, or insurer consumes. `drift
report` renders it into a single self-contained HTML document: what was tested,
what chaos was applied, what diverged, every finding with its evidence, the
counterfactual fork proof, and a sha256 of the source run for integrity.

```powershell
python -m drift report results/<run>.json `
    --fork results/fork_demo/<fork>.json `
    --target "my-agent-system" `
    --out evidence.html
```

No external fonts, scripts, or network fetches — archivable and print-to-PDF
clean. This is the shape emerging agent-certification schemes ask for: AIUC-1
(the standard behind the first insured AI-agent policies) requires evidence of
thousands of adversarial simulations on a **quarterly re-test cadence**. Drift
generates the simulations and the document in one pass.

[**→ See a real generated report**](https://claude.ai/code/artifact/3c12d718-8604-4062-a7f9-7a0680b59f34)
(langgraph-supervisor, LangChain's own multi-agent demo — 12 runs, 2 real
coordination bugs, fix verified by fork-replay).

---

## Web UI

```powershell
$env:PYTHONPATH = "e:\drift\src"
python -m drift serve
```

Opens at **http://127.0.0.1:8765**.

Eleven tabs:

- **Overview** — rolls up every saved drift_test-shaped run under
  `results/` into one dashboard: systems tested, perturbations applied, a
  findings-by-type chart, and every finding newest-first with a one-click
  CSV export. Click a finding to open its source run; click a chart bar to
  jump straight to that failure type's explainer on the Detectors tab.
- **Detectors** — the full detection surface, in one reference view: all
  12 deterministic library detectors (name, MAST citation, source, which
  ones need a config hint to fire) plus the LLM judge's 6 failure
  families. Generated live from the same metadata the detectors and judge
  actually ship with, so it can't drift out of sync with the code.
- **Test** — pick a graph (bundled ticket-triage demo or the
  langgraph-supervisor math+research demo), type a query, pick a preset
  (Quick / Balanced / Thorough / Exhaustive), run. Get every super-step of
  baseline + every per-perturbation trace side-by-side, every finding with
  a click-to-expand explanation, raw JSON download. Exhaustive is the
  "every applicable pattern in the schema, no sampling" pre-deploy gate.
  Advanced settings expose agent roles (`coordination_roles`, powers
  `role_drift`), the two provisional detectors' config hints
  (`required_followups=` / `policy_field=`), and "Track for regression."
- **Fork & Replay** — load a run (fresh or from Results), click a trace
  step, edit the state, re-run forward from there. Three augmentations on
  top of the basic replay: an optional top-vs-bottom comparison against
  applying the same edit at the initial state instead; bounded
  counterfactual replay (stop each branch after N more steps instead of
  running to completion, for cost control and focus); and an automatic
  path-dependence verdict when top-vs-bottom is used — does the same edit
  at two different points converge to the same outcome, or does the
  system carry real path-dependent state the initial design can't
  express? See "Fork-edit-replay" above.
- **Compare** — two saved runs side by side: headline stat diff, findings,
  trace.
- **Ask** — free-text Q&A over a loaded run's trace (or summary-only data
  for older saved runs that predate full-trace capture).
- **Import graph** — point drift at any importable Python module exposing
  `build(model=...)` / `initial_state(query)`, run the same pipeline
  against it. Executes arbitrary local code by design — same trust model
  as a Jupyter kernel; never expose this server on a network you don't
  control.
- **Build** — pick a coordination topology (supervisor, swarm, pipeline,
  loop, fanout), name a few agents, give each a short role description,
  get back a real runnable LangGraph file (every agent is a real LLM
  call) in an editable preview. Routing is fixed by position for every
  topology except loop's generator/critic stop condition — this is a
  starting point, not a finished system; edit the generated code before
  saving if you want to change anything. Save it and it's immediately
  testable via Import graph. Or skip naming agents by hand: describe a
  scenario in plain English ("how would a new sci-fi action movie do in
  theaters?") and **Suggest agents** asks an LLM to draft the topology +
  roster for you — same review-then-generate flow, AI just writes the
  first draft.
- **Test Suite** — run drift against several `examples/targets/` registry
  entries (real third-party multi-agent systems, not synthetic fixtures)
  in one pass with shared chaos/judge settings, plus an aggregate summary
  (crashed/diverged/findings totals across every target) above the
  per-target breakdown. A target missing a dependency is skipped, not
  attempted; one target failing doesn't stop the rest.
- **Simulate** — describe a scenario ("a streaming company loses a major
  content licensing deal") and get a structured prediction (direction /
  magnitude / confidence) from a multi-agent debate, grounded in real
  historical stock-reaction data (11 verified events, fetched live from
  Yahoo Finance — not fabricated) instead of pure LLM guessing. A
  **calibration** panel runs the same simulation against all held-out
  historical events and reports whether its predicted direction actually
  matched what happened — the only way "grounded in real data" means
  something checkable instead of just sounding more credible. Research
  tool, not financial advice: 11 events is a real but still small seed,
  not a statistically powerful one.
- **Results** — browse every saved experiment JSON from `results/`,
  filter by file name or group, generate an evidence pack from any of
  them, delete ones you no longer need.

---

## Empirical validation

Live case study against [`langchain-ai/langgraph-supervisor-py`](https://github.com/langchain-ai/langgraph-supervisor-py)
(official LangChain supervisor library, 1.6k stars) using the canonical
math + research demo from its README.

Four experiments, ~25 minutes wall clock, ~$0.50 OpenAI cost, 100+ perturbations:

1. **Question-diversity sweep** (12 queries) → surfaced 3 baseline-level
   coordination contradictions in the unperturbed supervisor + the universal
   silent-failure pattern on `clear_list[messages]` (36/36 cases).
2. **5-specialist extended MAS** → confirmed detectors stay silent on
   well-functioning real MAS (no false positives).
3. **3 adversarial graphs** → **all 3 structured detectors empirically fire**
   on real LangGraph code, not just fixtures. 1.6 seconds total.
4. **6 state-shape sweep** → drift's chaos auto-adapts (`flip_bool` for flags,
   `corrupt_string` for IDs, `clear_dict` for nested context, etc.).

Reproducers: [`examples/adapters/`](examples/adapters/) (each script
auto-saves to `results/`). Full writeup in
[`CASE_STUDY_LANGGRAPH_SUPERVISOR.md`](CASE_STUDY_LANGGRAPH_SUPERVISOR.md) (gitignored
local doc) and the plain-language version in
[`REPORT_LANGGRAPH_SUPERVISOR.md`](REPORT_LANGGRAPH_SUPERVISOR.md).

---

## Installation

```powershell
pip install -e .

# Optional extras:
pip install -e .[openai]      # real OpenAI judge + adapter
pip install -e .[web]         # FastAPI server for the web UI
pip install -e .[langgraph]   # langgraph package itself
pip install -e .[validation]  # langgraph + langgraph-supervisor + langchain-openai
pip install -e .[dev]         # pytest + pytest-asyncio
```

Or all at once: `pip install -e .[openai,web,langgraph,validation,dev]`.

### `.env`

The CLI loads `.env` automatically. Looks in (first match wins):
1. `./drift.env` (CWD)
2. `./.env` (CWD)
3. `<project-root>/.env`

For real LLM calls:
```
OPENAI_API_KEY=sk-...
```

---

## Project layout

```
e:\drift\
├── src/drift/
│   ├── adapters/
│   │   ├── langgraph.py   — drift_test() / drift_test_fork(): native streaming, rollouts,
│   │   │                    tiered cascade, checkpoint forks, judge + detector wiring
│   │   └── serialize.py   — the one JSON shape the UI, CLI, gate and report all read
│   ├── chaos/             — schema-walked auto-chaos: patterns, message-aware patterns, planner,
│   │                        tools.py (runtime tool faults via ToolNode wrap_tool_call)
│   ├── failures/
│   │   ├── library/       — coordination-failure detector library (19 detectors)
│   │   └── judge.py       — LLM judge over traces
│   ├── gate.py            — pass/fail rules for `drift run --fail-on` / `drift gate`
│   ├── guidelines.py      — drift.yml rule language (+ @drift.guideline) — docs/guidelines.md
│   ├── regression.py      — run-over-run trend tracking
│   ├── report.py          — self-contained HTML evidence report
│   ├── server.py          — FastAPI app behind the web UI
│   └── cli.py             — `drift run | gate | targets | serve | report | history`
├── web/                   — vanilla HTML/CSS/JS UI (Test, Import, Build, Fork, Results, …)
├── examples/adapters/     — validation harnesses against real third-party graphs
├── examples/targets/      — registry of real third-party MAS (see docs/targets_setup.md)
├── examples/ci/           — GitHub Actions workflow for the gate
├── docs/detectors/        — one explainer per detector; docs/ci.md for the CLI gate
├── results/               — saved run JSON (gitignored)
└── tests/                 — pytest suite (580+ tests)
```

> **2026-06-29 cleanup:** the native per-tick simulator (topologies / scenarios
> / fork+replay / run history) was removed alongside its UI tabs. The
> LangGraph adapter is the single supported path. The coordination-failure
> detector library + the LLM judge are the value layer running over adapter
> traces.

### Tests

```powershell
$env:PYTHONPATH = "e:\drift\src"
python -m pytest -q
```

198 tests cover: chaos engine (schema-walked perturbations + intensities
including exhaustive), tiered divergence cascade + UNCHANGED audit,
LLM judge (6-family taxonomy + user guidelines + dedup), all 6 coordination
detectors with synthetic positive + negative + cross-specificity fixtures,
LangGraph adapter integration paths, fork-edit-replay (state edit +
top-vs-bottom compare).

---

## Limitations to be honest about

- **Initial-state chaos only.** Mid-execution perturbation (via langgraph's
  checkpointer) is in `FUTURE_DIRECTIONS.md` as phase 4 — not built. Within
  the initial-state target, `intensity="exhaustive"` covers every applicable
  schema pattern; what's missing is *time-axis* coverage, not schema coverage.
- **Tier 2 noise filtering is weak on rich message structures.** When state
  contains LangChain `AIMessage` objects with metadata (IDs, timestamps,
  token counts), text-similarity scoring can't filter well — everything
  escalates to tier 3, burning judge budget. Real product limit.
- **Tier-2 noise filtering is rough on text fields with high natural
  variance** (template-y prose, similar but not identical responses).
  The noise floor measurement helps, but the threshold itself is one
  number applied across all fields — a per-field comparator override
  would be more correct.
- **19 coordination detectors is a starting kit, not a complete library.**
  Twelve read node-level state writes; seven read the chat messages (tool
  calls, tool results, final answers, work and tools used vs the baseline,
  data flowing from tool results into privileged tool arguments). Domain-specific failures need the
  user-guideline mechanism — which is currently a free-text textarea, not a
  structured DSL.
- **Drift now streams real LangGraph graphs with `subgraphs=True`**, so
  sub-agents that run as subgraphs (every `create_react_agent` under a
  supervisor) show up as `math_expert/agent`, `math_expert/tools`, … steps,
  and a sub-agent started from inside a tool call — the
  [deepagents](https://github.com/langchain-ai/deepagents) pattern that was a
  documented blind spot — appears as `tools/model`, `tools/…` steps, each
  tagged with the sub-agent's name recovered from the dispatching tool call
  (`task(subagent_type="critic")` → `critic`; parallel dispatches are told
  apart by task text). On a live deepagents run all 15 of 15 sub-agent steps
  were named `researcher` / `critic` / `summarizer`. What is still true: a
  sub-agent implemented as a plain function call inside a tool (no LangGraph
  graph) stays invisible, and a dispatcher that names the agent nowhere in
  its arguments or tool name stays anonymous. Hand-rolled stubs without
  `stream_mode` keep the old flat view.
- **Empirical evidence is still thin, and now measured as rates.** On
  LangChain's own 2-agent supervisor example with gpt-4o-mini, every
  message perturbation run 3 times (`examples/adapters/run_drift_on_supervisor_message_chaos.py
  --rollouts 3 --save-json`, 2026-11-09): the planted indirect prompt
  injection was followed in **1 of 3** rollouts (the model answered with
  only the canary) and flagged intermittent; the planted failed tool call
  was papered over in **3 of 3**; a fabricated "already done" turn and the
  injection each produced a `novel_tool_call` (`web_search`, never called at
  baseline) and `retry_amplification` (2.0×) in **2 of 3**; the tool-hijack
  injection reached a tool call in **0 of 3** (this graph has no tool the
  supervisor can call directly, so the demand had nowhere to land). On a
  plain `create_react_agent` with one mocked `web_search` tool
  (`examples/adapters/run_drift_on_react_hijack.py`, same day), the phrase
  injection was followed **3 of 3** times and the tool-hijack injection
  reached a real `web_search(query="DRIFT-CANARY-…")` call **3 of 3** times
  when the payload named the tool — while the final answer stayed "The
  capital of France is Paris." every time. When the payload could only ask
  for "one of your tools" (no prior call in the history) it was ignored
  **3 of 3**. Give the same agent a mocked `send_email` tool and the
  injected demand produced `send_email(to=…, body="DRIFT-CANARY-…")`
  **3 of 3** — `injected_tool_call_followed`, `novel_tool_call`
  (SIDE-EFFECT) and `tainted_side_effect_call` all fired, and the final
  answer was still correct. On the 5-specialist variant a real `tool_call_loop` fired
  (`_web_search` ×4). The
  earlier 5-specialist judge findings (2 real `coordination_contradiction`s)
  still stand. Three rollouts is a rate, not a verdict; drift hasn't been
  shown to catch things engineers couldn't find by reading their own code,
  only that it surfaces them systematically and with lower friction.

See `FUTURE_DIRECTIONS.md` and the post-compaction kickoff memory file for
the next builds.
