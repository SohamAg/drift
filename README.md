# drift

> Pre-deploy chaos testing for LangGraph multi-agent systems. Drift takes a
> compiled graph, perturbs its initial state, captures the per-super-step
> trace, and reports the crashes + silent divergences + named coordination
> failures you wouldn't find with a happy-path test.

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

## What drift actually does

Three concentric layers, each cheaper than the last:

### 1. Chaos engine — schema-walked auto-perturbation
Reads the runtime types of your `initial_state`. For each field, picks
type-appropriate perturbations: `flip_bool`, `corrupt_string`, `clear_list`,
`duplicate_list_entry`, `remove_dict_key`, `inject_fake_dict_key`,
`boundary_numeric`, etc. No configuration. Runs each perturbed state through
your graph and bucket the outcomes into `crashed | diverged | unchanged`.

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

### 3. Coordination-failure detector library
Curated, source-cited detectors for documented multi-agent failure modes.
Currently shipped:

| Detector | Source |
|---|---|
| `verifier_always_approves` | MAST 3.x family + Anthropic engineering blog |
| `infinite_handoff` | MAST 1.3 + Cognition open problem #2 |
| `subagent_fanout_excess` | Anthropic 50-subagent incident |

Each detector ships a structured `detect()` (over the adapter trace) plus a
text-only `detect_from_text()` (for MAST-style transcripts). Free,
deterministic, runs alongside every drift_test. **Empirically validated** on
real LangGraph code (`examples/adapters/run_drift_on_adversarial_mas.py`),
not just synthetic fixtures.

### Plus — optional LLM judge over the full trace
Six-category taxonomy (`coordination_contradiction`, `grounding_failure`,
`state_drift`, `emergent_decay`, `gate_bypass`, `user_guideline`) applied to
the per-super-step trace. Catches semantic coordination issues a single-step
detector can't see. Add custom rules in plain English via `user_guidelines=`.

---

## Web UI

```powershell
$env:PYTHONPATH = "e:\drift\src"
python -m drift serve
```

Opens at **http://127.0.0.1:8765**.

Three tabs:

- **Adapter** — pick a graph (bundled ticket-triage demo or the
  langgraph-supervisor math+research demo), type a query, pick a preset
  (Quick / Balanced / Thorough / Exhaustive), run. Get every super-step of
  baseline + every per-perturbation trace side-by-side, every finding with
  a click-to-expand explanation, raw JSON download. Exhaustive is the
  "every applicable pattern in the schema, no sampling" pre-deploy gate.
- **Results** — browse every saved experiment JSON from `results/`.
- **Custom** — currently paused while we re-wire `@drift.agent` to the
  adapter path (was previously powered by the now-removed native simulator).

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
│   │   └── langgraph.py   — drift_test() + tiered cascade + judge wiring
│   ├── agents/base.py     — Action / Agent data classes (used by adapter + library)
│   ├── chaos/             — Schema-walked auto-chaos engine
│   ├── events/base.py     — Event / EventRecord (used by chaos patterns)
│   ├── failures/
│   │   ├── library/       — Coordination-failure detector library (3 detectors)
│   │   ├── judge.py       — LLM judge over agent traces (6-family taxonomy)
│   │   └── mast_eval.py   — MAST dataset evaluation helpers
│   ├── llm/               — Protocol + ScriptedMockLLM + OpenAI adapter
│   ├── world.py           — WorldState data primitive
│   ├── server.py          — FastAPI app (adapter + results + MAST endpoints)
│   ├── cli.py             — `drift serve`
│   └── sdk.py             — @drift.agent decorator (data shape; runtime pending re-wire)
├── web/                   — Vanilla HTML/CSS/JS frontend (3 tabs: Adapter / Results / Custom-stub)
├── examples/adapters/     — drift_test runners + validation harnesses
├── data/external/mast/    — MAST dataset (gitignored)
├── results/               — Saved experiment JSON (gitignored)
└── tests/                 — Pytest suite (140 tests)
```

> **2026-06-29 cleanup:** the native per-tick simulator (topologies / scenarios
> / fork+replay / run history) was removed alongside its UI tabs. The
> LangGraph adapter is the single supported path. The 3-detector coordination
> library + the LLM judge are the value layer running over adapter traces.

### Tests

```powershell
$env:PYTHONPATH = "e:\drift\src"
python -m pytest -q
```

140 tests cover: chaos engine (schema-walked perturbations + intensities
including exhaustive), tiered divergence cascade + UNCHANGED audit,
LLM judge (6-family taxonomy + user guidelines + dedup), all 3 coordination
detectors with synthetic positive + negative + cross-specificity fixtures,
LangGraph adapter integration paths.

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
- **3 coordination detectors is a starting kit, not a complete library.**
  The detectors target universal patterns (auto-approving verifier, infinite
  handoff, excess fanout). Domain-specific failures need the user-guideline
  mechanism — which is currently a free-text textarea, not a structured DSL.
- **Empirical evidence is on one library** (langgraph-supervisor). Drift
  hasn't been shown to catch things engineers couldn't find by reading
  their own code; only that it surfaces them more systematically + with
  lower friction.

See `FUTURE_DIRECTIONS.md` and the post-compaction kickoff memory file for
the next builds.
