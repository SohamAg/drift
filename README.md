# drift

> Pre-deploy testing for the space between agents. Drift auto-generates
> coordination-breaking scenarios from your state schema, runs them through
> your compiled agent, and surfaces crashes plus silent divergences — the
> production failure modes you wouldn't write a test for because you didn't
> think of them.

---

## Use it with your LangGraph app

The fastest path. Drift takes any compiled LangGraph (or anything with
`.invoke(state) -> dict`), reads the runtime types of your state, generates
schema-driven perturbations, and reports which ones broke your graph.

```python
from drift.adapters.langgraph import drift_test

result = drift_test(
    graph=my_compiled_graph,
    initial_state={"messages": [...], "open_tickets": {...}, "is_premium": True},
    intensity="moderate",
)

for p in result.perturbations:
    if p.crashed:
        print(f"CRASH   {p.event_name}: {p.error_type}: {p.error}")
    elif p.diverged:
        print(f"DIVERGE {p.event_name}: {p.divergence_summary}")
```

Worked end-to-end demo at [examples/adapters/langgraph_demo.py](examples/adapters/langgraph_demo.py).
The same flow is also wired into the web UI under the **Adapter** tab.

---

## How to run the web UI / native simulator

Two ways: a **web UI** (recommended) or the **CLI**. Pick one — they hit the
same simulator.

### Prerequisites

- Python 3.10+
- The dependencies in [pyproject.toml](pyproject.toml) (`pydantic`, `pyyaml`,
  `fastapi`, `uvicorn`, and optionally `openai`, `anthropic` for real LLMs).
- An `.env` file at `e:\drift\.env` if you want to use OpenAI:
  ```
  OPENAI_API_KEY=sk-...
  ```
  See the *Configuration* section for where else `.env` can live.

### Web UI (recommended)

Open a PowerShell window in `e:\drift` and run:

```powershell
$env:PYTHONPATH = "e:\drift\src"
python -m drift serve
```

Then open **http://127.0.0.1:8765** in your browser. That's it.

The UI has four tabs:
- **New Run** — configure and launch a simulation; watch the live status panel.
- **Runs** — every run on disk, searchable.
- **Compare** — diff two runs side by side (failure counts, action mix, final state).
- **About** — quick reference for the topologies.

Useful flags:
```powershell
python -m drift serve --port 9000       # different port
python -m drift serve --reload          # auto-reload Python on edits
```

Stop with `Ctrl+C`. Frontend edits (`web/*`) only need a browser hard-refresh
(`Ctrl+Shift+R`); Python edits need a server restart (or `--reload`).

### CLI

Same engine, no browser. From `e:\drift`:

```powershell
$env:PYTHONPATH = "e:\drift\src"

# Run a simulation
python -m drift run `
  --topology code_review `
  --scenario scenarios/release_pressure.yaml `
  --steps 30 `
  --seed 7 `
  --llm mock `
  --prompt-variant naive `
  --run-id my_first_run

# Compare two completed runs
python -m drift compare runs/my_first_run runs/some_other_run
```

The CLI writes the run's logs to `runs/<run_id>/` as JSONL
(`events.jsonl`, `actions.jsonl`, `snapshots.jsonl`, `failures.jsonl`,
`run_meta.json`). The web UI reads from the same directory — runs are
interchangeable between the two interfaces.

### Configuration (.env)

The loader looks for `.env` in (first match wins):
1. `./drift.env` (CWD)
2. `./.env` (CWD)
3. `<project-root>/.env`

So putting it at `e:\drift\.env` works no matter where you launch from.
Already-set environment variables are not overridden.

---

## What it does

Drift simulates an organization where 4 AI agents share a mutable world.
On each timestep:

1. Scheduled and stochastic **events** fire, mutating the world (e.g. a
   Black Friday spike, a fresh CVE, a sev-1 incident).
2. All 4 **agents** observe the same world snapshot and decide concurrently
   what to do (via a mock LLM, real OpenAI, or a stub Anthropic adapter).
3. Their actions are **applied sequentially** in a deterministic order so
   the audit trail stays clean.
4. **Detectors** scan the new state and the action log for emergent
   failures — contradictions, loops, drift, coordination races.
5. A snapshot is written to disk.

After N steps you get a report (and JSONL logs). Comparing two runs side
by side is the workflow that makes this useful — change a prompt or a seed,
see whether the system as a whole got more or less coherent.

---

## The choices you make per run

Four knobs on every run. They're orthogonal — combine freely.

### Topology (`--topology`)

The *kind* of organization being simulated. Each topology brings its own
4 agents, events, detectors, and world fields.

| Topology      | Agents                                          | Domain failures it catches                                              |
|---------------|-------------------------------------------------|-------------------------------------------------------------------------|
| `support`     | Support, Refund, Escalation, Policy             | contradictory_refund, escalation_loop, policy_inconsistency             |
| `code_review` | Proposer, Reviewer, Security, Merge             | contradictory_review, security_bypass, merge_without_approval           |
| `ops`         | Triage, Diagnosis, Remediation, Comms           | contradictory_diagnosis, silent_remediation, comms_lag                  |

All three also run the four **general detectors**: `sentiment_collapse`,
`queue_explosion`, `hallucinated_reference`, `stale_snapshot_reference`.

### Scenario (`--scenario`)

A YAML file in [scenarios/](scenarios/) that schedules events at specific
timesteps and rolls a per-step probability for stochastic events.

| Scenario                      | Built for     | Stresses                                          |
|-------------------------------|---------------|---------------------------------------------------|
| `black_friday.yaml`           | support       | load spikes + angry customers + policy churn      |
| `policy_chaos.yaml`           | support       | frequent policy changes → policy_inconsistency    |
| `queue_overflow.yaml`         | support       | sustained spikes → queue_explosion                |
| `release_pressure.yaml`       | code_review   | deadline + CVE + conflicting rebases              |
| `ops_storm.yaml`              | ops           | sev-1 spikes + upstream outage + customer noise   |

You can also omit `--scenario` to run with only stochastic events.

### Prompt variant (`--prompt-variant`)

Two versions of every agent's system prompt:

- **`naive`** — what a developer writes on day one. Role + one line.
  No guardrails.
- **`hardened`** — same prompt plus explicit rules that map directly to
  drift's detectors (e.g. *"referenced_policy_version MUST equal current"*,
  *"target_case_id MUST be from open_case_ids"*, *"never approve a PR you
  previously rejected"*).

The demo loop: run naive → see failures → switch to hardened → re-run with
the same seed → compare.

**Caveat:** the mock LLM is a hardcoded dice-roller and ignores
`system_prompt`. So `naive` and `hardened` produce identical runs under
`--llm mock`. The variant only changes behavior with a real LLM
(`--llm openai`).

### LLM backend (`--llm`)

| Backend     | Cost          | Deterministic | Notes                                              |
|-------------|---------------|---------------|----------------------------------------------------|
| `mock`      | free          | yes (seeded)  | Default. Hardcoded role handlers; runs in seconds. |
| `openai`    | ~$0.05/run    | no            | `gpt-4o-mini` by default. Needs `OPENAI_API_KEY`.  |
| `anthropic` | (stub)        | —             | Adapter scaffolded; not wired.                     |

Change the model with `--model gpt-4o-mini` etc.

---

## Output and observability

Every run writes to `runs/<run_id>/`:
- `events.jsonl` — one event record per line
- `actions.jsonl` — one agent action per line
- `snapshots.jsonl` — full world state per timestep
- `failures.jsonl` — every detector hit
- `run_meta.json` — the config the run was launched with

The web UI reads these on demand. The CLI prints a summary; raw logs are
the source of truth.

---

## Project layout

```
e:\drift\
├── src/drift/
│   ├── agents/         — base Agent + support topology agents + prompt strings
│   ├── events/         — base Event + support events + YAML scheduler
│   ├── failures/       — base Detector + the 7 detectors (4 general, 3 support)
│   ├── llm/            — Protocol + mock + OpenAI adapter + Anthropic stub
│   ├── observability/  — JSONL logger + metrics tracker
│   ├── topologies/     — Topology registry + code_review + ops bundles
│   ├── world.py        — WorldState, Case, World API
│   ├── simulation.py   — the per-tick loop
│   ├── server.py       — FastAPI app for the web UI
│   ├── cli.py          — argparse entrypoint (`run`, `compare`, `serve`)
│   └── testing.py      — counter-reset helper for deterministic tests
├── web/                — vanilla HTML/CSS/JS frontend (no build step)
├── scenarios/          — YAML scenario library
├── tests/              — pytest suite (21 tests)
└── runs/               — every run's JSONL logs
```

## Tests

```powershell
$env:PYTHONPATH = "e:\drift\src"
python -m pytest tests -v
```

21 tests cover: world invariants, each detector firing on isolated fixtures
(and staying silent on clean state), per-topology smoke runs, and seed
determinism across the three topologies.

---

## Limitations to be aware of

- The mock LLM ignores prompt variants. Use `--llm openai` to see naive vs
  hardened actually differ.
- `stale_snapshot_reference` over-counts when multiple agents target the
  same case in one step. It's still a real coordination failure, just noisy.
- The Anthropic adapter is a stub — it raises `NotImplementedError`.
- The web server binds to `127.0.0.1` only. Don't expose it without an auth
  layer in front.
