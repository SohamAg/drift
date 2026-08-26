# Target systems — setup and status

Working doc for running drift against real third-party LangGraph multi-agent
systems. Written 2026-08-25.

**Why this exists:** drift's empirical evidence is two systems, one of which
it structurally cannot observe. To claim drift finds coordination bugs, it
has to find them in code we did not write. This file tracks which real
systems are verified, how to install them, and what's blocked.

**Hard rule going forward:** a target only counts if it is third-party code.
Patterns transcribed from tutorials into this repo are *not* targets — that
is drift testing our own code. Seven such fabricated targets were written and
deleted on 2026-08-25; don't recreate them.

---

## Verification standard

A candidate qualifies only if all five hold:

1. Built on **LangGraph** — verified by reading its source for `StateGraph`
   and `.compile()`, not by trusting its README.
2. **Multi-agent** — several coordinating nodes, not one ReAct loop.
3. **Importable** — you can get a compiled graph object in Python and call
   `.invoke()` / `.stream()` on it. A hosted product or CLI-only app does
   not qualify.
4. **Popular or officially maintained.**
5. Preferably **large** — many agents, deep graphs, real state.

---

## Environment

Targets and drift must share one environment (drift imports the graph object
to test it), so per-target venvs don't work. The global Python already carries
376+ packages and is showing strain (protobuf conflict with streamlit surfaced
while installing open-deep-research). Recommendation is a dedicated,
disposable venv in the repo.

```powershell
cd e:\drift
python -m venv .venv-targets
.venv-targets\Scripts\Activate.ps1

# drift itself, editable
pip install -e ".[openai,web,langgraph]"

# targets
pip install langgraph-supervisor langgraph-swarm deepagents
pip install open-deep-research
pip install tradingagents
```

Add to `.gitignore`:
```
.venv-targets/
```

Rollback snapshot of the global env (taken before today's installs):
`/tmp/env_before_targets.txt` — 376 packages. Restore with
`pip install -r`. Note `/tmp` does not survive a reboot; copy it somewhere
durable if you care about it.

Verify the venv:
```powershell
$env:PYTHONPATH = "e:\drift\src"
python examples\targets\verify_targets.py
```

---

## Status

### Installed and verified

| Package | Source | Topology | Notes |
|---|---|---|---|
| `langgraph-supervisor` | langchain-ai/langgraph-supervisor-py (official) | supervisor | Already validated. Drift found a real bug: the supervisor re-issues requests a specialist is answering. Zero false positives over 12 runs. |
| `langgraph-swarm` | langchain-ai/langgraph-swarm-py (official) | swarm | Peer-to-peer handoff, no central router. **Never run — new topology.** |
| `deepagents` | langchain-ai/deepagents (official, 25.7k ⭐) | tool_dispatch | **Negative control.** Sub-agents dispatched via tool calls, so drift sees only `model → tools → model`. Keep it: it re-confirms the observation boundary on every run. |
| `open-deep-research` | langchain-ai/open_deep_research (official) | hierarchical | Installed 2026-08-25, loads clean. |

`open-deep-research` entry point:
```python
from open_deep_research.deep_researcher import deep_researcher
# nodes: clarify_with_user, write_research_brief,
#        research_supervisor (subgraph), final_report_generation
```
There is also a legacy supervisor variant at `legacy/multi_agent.py`
exporting `graph` (supervisor + `research_team` subgraph).
Wants a **Tavily** key for real search.

### Verified, not yet installed

**`tradingagents`** — TauricResearch/TradingAgents. The largest target found.
Confirmed `langgraph>=1.1.10`; graph built in
`src/tradingagents/graph/setup.py`, ~20 nodes:

- 4 analyst types × 3 nodes each (analyst, msg-clear, tools)
- Bull Researcher, Bear Researcher, Research Manager
- Trader
- Aggressive / Neutral / Conservative analysts under a Risk Judge

Adversarial debate structure — the most interesting coordination shape
available, and the one most likely to surface contradictions. Expect a
financial-data key (FinnHub or similar).

### Blocked

**`gpt-researcher`** (assafelovic) — 8-node pipeline (browser, planner,
researcher, writer, fact_checker, visualizer, publisher, human) in
`multi_agents/agents/orchestrator.py`. Genuinely LangGraph and genuinely
multi-agent, but **v0.16.0 on PyPI is broken**:

```
NameError: name 'Any' is not defined
  gpt_researcher/actions/query_processing.py:6
```

Their released package has a missing `typing` import. Options: pin an older
release (0.15.1 and many earlier versions exist), or install from git.

### Rejected

- **STORM** (stanford-oval / `knowledge-storm`) — not LangGraph.
- **`langgraph-bigtool`** — is LangGraph, but single-agent tool retrieval.

---

## Open decisions

**1. API keys vs stubbed tools.**
Several targets need more than `OPENAI_API_KEY` — open_deep_research wants
Tavily, TradingAgents wants financial data. Two paths:

- *Real keys*: tests the system unmodified. Costs money, and non-deterministic
  tool results add noise to divergence detection (drift compares runs; if
  search returns different text each call, that noise competes with the
  signal).
- *Stubbed tools*: free, reproducible, clean divergence signal — but it is a
  modification, which weakens "we tested it unmodified."

Not yet decided. Leaning: real keys where cheap, stub only where a service is
paywalled, and state clearly in any writeup which targets were stubbed.

**2. Registry shape.**
`examples/targets/` currently holds `supervisor_basic.py`, `swarm_peer.py`,
`deepagents_tooldispatch.py`, plus `__init__.py`, `_tools.py`,
`verify_targets.py`. The registry in `__init__.py` still lists the seven
deleted modules, so `tests/test_target_registry.py` fails collection. That is
mid-refactor, not a regression — it resolves when the registry is rebuilt
around real targets.

Note: `supervisor_basic.py` and `swarm_peer.py` are thin wrappers around the
vendors' own README examples. Defensible, but they are our configuration of a
real library, not a real application. `open_deep_research` and `tradingagents`
are the genuine article — real applications people actually run.

---

## Next steps

1. Build `.venv-targets`, install everything, confirm 10/10 import.
2. Rebuild the registry around real targets only; get tests green.
3. Install `tradingagents`; resolve `gpt-researcher` via pin or git.
4. Decide keys-vs-stubs.
5. Build the harness that runs drift across all targets and emits one
   comparison report.
6. Run it. Estimate: 20–40 min, $3–8 at gpt-4o-mini rates with the judge on.

The interesting question is not "does drift run on system X." It is **what
does drift find across coordination topologies** — and specifically whether
the supervisor-repetition bug is peculiar to that one demo or a pattern that
recurs across supervisor-style systems. If it recurs, that is a finding worth
publishing on its own.

---

## Useful commands

```powershell
$env:PYTHONPATH = "e:\drift\src"

python examples\targets\verify_targets.py            # compile all, no LLM calls
python examples\targets\verify_targets.py --no-build # deps/registry only
python examples\targets\verify_targets.py --json

python -m pytest tests\ -q                           # full suite
python -m drift serve                                # dashboard, :8765
python -m drift report <results.json> --out out.html # evidence pack
```

Checking whether a candidate is really LangGraph, without installing it:
```bash
pip download --no-deps --no-binary :all: -d /tmp/chk <package>
cd /tmp/chk && tar xzf <package>*.tar.gz
grep -rn "StateGraph" <package>*/ | head
grep -n "langgraph" <package>*/pyproject.toml
```
