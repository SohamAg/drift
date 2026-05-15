---
name: drift-context
description: Load full strategic, product, and competitive context for the drift project — what it is, where it stands, how to talk about it, what's defensible, what's not. Invoke at the start of any session where the work is about drift's positioning, YC application, product direction, competitive analysis, or anything strategic. Skip for pure code-level tasks (bug fixes, refactors) — the project's CLAUDE.md / code already covers those.
---

# drift — company context

## One-line pitch

**Chaos engineering for multi-agent AI systems.** Anyone deploying multi-agent AI should be able to use drift to stress-test their agents pre-deployment, by injecting chaos events into running coordination flows and watching for named coordination failures.

Multi-agent systems fail in ways single-agent systems don't — and the discipline to catch those failures before production doesn't exist yet. Drift is building it.

## What drift IS

- A **deterministic detector library** for multi-agent coordination failures — 13 named modes (e.g., contradictory_refund, hallucinated_reference, silent_remediation), each a pure ~20-line Python function over (action history, world snapshots).
- A **chaos / scenario harness** that drives multi-agent systems through stress conditions (mid-flight policy changes, dependency outages, conflicting events) and watches what breaks at the coordination level.
- A **counterfactual replay engine** — fork any run at any timestep with deterministic overrides (different seed, different prompts per role, disable an agent), compare branches to isolate causes.
- A working web UI + CLI (FastAPI backend, vanilla HTML/CSS/JS frontend).

## What drift IS NOT (don't claim these)

- ❌ **NOT** "the first multi-agent failure taxonomy." MAST (Cemri et al., Berkeley, March 2025, arXiv:2503.13657) is. 14 named modes, 200 human-annotated traces, LLM-as-judge pipeline. They got there first.
- ❌ **NOT** "the first multi-agent eval framework." MAESTRO (Ma et al., KAUST, Jan 2026, arXiv:2601.00481) is a published eval suite with 117k OTEL spans across 12 MAS examples.
- ❌ **NOT** an orchestration layer. LangGraph, CrewAI, AutoGen, Microsoft's Conductor, zenflow already own that.
- ❌ **NOT** an LLM trace observability tool. Langfuse / Phoenix / LangSmith / AgentOps own that.
- ❌ **NOT** an LLM eval framework. Maxim, Braintrust, Ragas, Promptfoo own that.

## The core product (vision vs current state)

**Vision:** A user takes their existing multi-agent system (built in LangGraph / CrewAI / AutoGen / custom), wraps the agents with a drift SDK, defines or accepts drift's chaos scenarios, runs `drift test`, and gets a deterministic report of which coordination failures will emerge under stress — *before deploying to production*.

**Current state (as of 2026-05-15):**
- ✅ Simulator with drift's own hard-coded agents across 3 topologies (`support`, `code_review`, `ops`)
- ✅ Detector library running on drift's own simulator output
- ✅ Counterfactual replay (`drift fork`)
- ✅ Web UI (Runs, Compare, Run Detail, Fork modal, Analyze tab)
- ✅ Trace ingester — `drift analyze` accepts drift-format JSONL from external sources
- ⚠️ Trace ingester does NOT yet read OTEL/gen_ai spans natively — friction
- ❌ NO BYOA (Bring Your Own Agents) SDK — users can't wrap their existing agents with drift yet
- ❌ NO framework adapters (LangGraph / CrewAI / AutoGen hooks)
- ❌ NO empirical validation case study against real public data

**The gap:** drift is currently a closed-world simulator that demonstrates failure patterns. The *product* needed to deliver the pitch is the BYOA SDK + framework adapters. That's the ~6-8 hour build that moves drift from "research artifact" to "tool you can point at your stack."

## Defensible differentiation vs the competitive map

| Player          | What they have                                                  | What drift has that they don't                                              |
| --------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **MAST**        | Taxonomy + LLM-as-judge over completed traces                   | Deterministic detectors (~100x cheaper, CI-runnable). Chaos injection. Replay. |
| **MAESTRO**     | Telemetry collection + 12 example MAS systems benchmarked       | Test user-supplied agents. Chaos injection. Counterfactual replay.            |
| **Maxim AI**    | Persona-based simulation (single agent + simulated user)        | Genuine multi-agent coordination chaos. Coordination-failure detection.       |
| **Langfuse / Phoenix / LangSmith** | Trace observability postmortem                   | Pre-deployment stress testing. Detector taxonomy. Replay.                     |
| **Gremlin / LitmusChaos** | Infrastructure chaos (kill pods, throttle network)       | Speak the language of LLM agents.                                             |
| **Braintrust / Vellum / Ragas** | LLM output eval                                    | Multi-agent. Coordination. Chaos.                                             |

**Three pillars drift can credibly claim:**
1. **Deterministic detectors** — pure functions, reproducible, ~100x cheaper than LLM-as-judge, runnable in CI on every PR.
2. **Counterfactual replay** — fork at any timestep with overrides. Causal analysis, not just correlation. Neither MAST nor MAESTRO has this.
3. **Chaos injection** — generates failures pre-production rather than waiting to find them postmortem. Direct analog to Netflix Chaos Monkey for the multi-agent era.

## The detector taxonomy (current 13)

**General — work across any topology (4):**
- `sentiment_collapse` — aggregate health metric below threshold for 5 consecutive steps
- `hallucinated_reference` — agent acted on a case/PR/incident ID that never existed
- `stale_snapshot_reference` — agent acted on something that existed but was removed mid-step (coordination failure, not hallucination)
- `queue_explosion` — backlog growing faster than clearance over a rolling window

**Support topology (3):**
- `contradictory_refund` — same case got both approve and deny
- `escalation_loop` — same case escalated more than 3 times
- `policy_inconsistency` — action referenced a stale policy version

**Code-review topology (3):**
- `contradictory_review` — same PR got both approve and reject
- `security_bypass` — PR merged while security was blocked
- `merge_without_approval` — PR merged with zero approvals preceding

**Ops topology (3):**
- `silent_remediation` — fix shipped but no comms within 3 steps
- `comms_lag` — high-sev incident triaged but no comms within 5 steps
- `contradictory_diagnosis` — same incident received two different root-cause hypotheses

Each detector is in `src/drift/failures/detectors.py` or `src/drift/topologies/<name>/detectors.py`. Pure functions over `DetectorContext`. ~20 lines each. Auditable.

## The 3 shipped topologies

| topology      | domain                  | 4 agents                                      |
| ------------- | ----------------------- | --------------------------------------------- |
| `support`     | customer support / refunds | Support, Refund, Escalation, Policy          |
| `code_review` | PR review / merge        | Proposer, Reviewer, Security, Merge          |
| `ops`         | incident response        | Triage, Diagnosis, Remediation, Comms        |

These are scaffolding, not the product. Real users will bring their own agents via the (not-yet-built) BYOA SDK. The topologies exist to (a) demonstrate the detectors fire, (b) provide a sandbox for exploring chaos scenarios, (c) give a UI demo with familiar domains.

## Strategic context — where we are in the timeline

- **Multi-agent in production just became real.** Microsoft shipped MDASH this week (May 13, 2026) — 100+ agents in coordinated vulnerability scanning, beat Anthropic's Mythos 88.4% vs 83.1% on CyberGym. Anthropic's Mythos itself is restricted to a "Project Glasswing" consortium. Multi-agent is no longer a toy.
- **Academic foundations are being laid.** MAST (March 2025), MAESTRO (Jan 2026), plus papers on planning/coordination — the discipline is forming.
- **No commercial chaos-engineering tool for multi-agent exists.** This is the open lane.
- **Adoption curve:** today multi-agent is <1% of LLM deployments. By late 2026 / mid 2027 expect 10-20% as MDASH-class systems normalize. Drift is somewhere between "exactly right time" and "6-12 months early." The risk is being early; the upside is owning the category.

## Stage

- Solo founder (no co-founder as of this writing)
- Applying to YC soon (timing flexible but acute)
- No revenue, no paying users, no signed LOIs as of 2026-05-15
- One working MVP, three topologies, 29 tests passing, polished web UI
- 13 detector library

## Honest risks (do not paper over)

1. **Current product ≠ stated positioning.** Drift's simulator runs drift's own agents. The pitch promises "test your agents." The BYOA SDK closes this gap; it's not built yet.
2. **Timing risk.** Multi-agent adoption is at the inflection but most teams aren't there yet. A YC reviewer asking *"who buys this in the next 12 months?"* needs an answer like *"Cognition, Sierra, Decagon, Maven AGI, the wave of mid-market teams shipping MAS in late 2026."*
3. **Adoption friction.** Wrapping existing agents with a drift SDK is a 30-60 min integration. Some won't do it. Trace ingester partially mitigates by letting people try drift's value before SDK integration.
4. **Replacement risk.** LangSmith / Maxim / Langfuse could ship multi-agent chaos primitives in a quarter. The moat is execution speed + becoming canonical (the "drift score" everyone reports against). Not patent-protected.
5. **Validation gap.** Drift's detectors have not yet been validated against real-world (non-drift-simulator) traces. The MAESTRO + MAST datasets exist for this validation but no case study has been written yet.

## What good YC framing looks like

```
Multi-agent AI is now SOTA — Microsoft's MDASH shipped this week with 100+ agents.
MAST (Berkeley, 2025) named the failure modes; MAESTRO (KAUST, 2026) instrumented
the systems. Neither lets you stress-test your own agents before production.

We're building chaos engineering for multi-agent AI: a deterministic detector
library (100x cheaper than LLM-as-judge, runnable in CI), counterfactual replay
(fork at any step with overrides), and chaos event injection (generate failures
pre-deploy rather than wait for them in production). The wedge: developers
running multi-agent systems wrap their agents with our SDK, run our scenarios,
get a deterministic failure report before they ship.
```

What that sentence does:
- Cites prior art correctly (no overclaiming)
- Stakes three specific differentiators
- Names the wedge in concrete terms
- Has a plausible "who buys this" baked in

What NOT to say in the YC application:
- *"First multi-agent failure taxonomy"* (MAST is)
- *"First evaluation framework for MAS"* (MAST, MAESTRO are)
- *"We invented the emergence layer"* (vague; sounds academic; not defensible)
- *"We predict emergent behavior"* (overclaim; drift surfaces, doesn't predict)

## What's next, ranked by leverage (as of 2026-05-15)

1. **BYOA SDK + LangGraph adapter** (~6-8 hours). The single change that turns drift from research artifact into the product the pitch promises. Without this, the YC pitch is hypothetical.
2. **Empirical validation case study** (~6 hours). Run drift's detectors over MAESTRO's 1,056 multi-agent runs. Cross-walk results to MAST's 14 modes. Write `CASE_STUDY.md`. This becomes the credibility attachment to the YC application.
3. **OTEL ingester** (~2 hours). Lets drift consume any OTEL-instrumented MAS without users writing drift-format JSONL. Reduces friction substantially.
4. **DM 5-10 multi-agent teams** for user signal — Cognition, Sierra, Decagon, adamsreview author, others on the recent HN multi-agent feed. The strongest YC differentiator beyond product is *"we talked to N teams running MAS in prod; M confirmed they've hit drift's failure modes."*
5. **Fix mock LLM honoring prompt variant** (~45 min). Demo credibility on free tier — currently naive vs hardened produces identical output with `--llm mock`.

## How to talk about drift in a session

**Do:**
- Frame around "chaos engineering for multi-agent AI" or "reliability testing for multi-agent AI" — these are concrete, understandable categories with proven commercial precedent (Gremlin, Chaos Monkey).
- Cite prior art (MAST, MAESTRO) honestly. Demonstrating awareness of the academic landscape is itself credibility.
- Lead with the three pillars: deterministic detectors, counterfactual replay, chaos injection.
- Acknowledge what's not built when asked. Don't oversell the current state.

**Don't:**
- Claim drift is "first" at anything that has academic prior art (taxonomy, eval framework, detection pipeline).
- Use vague framing ("emergence layer", "predict emergent behavior") — they don't survive a 5-minute due diligence call.
- Conflate drift's simulator (the demo) with drift's product (BYOA + chaos + detectors).
- Pretend the trace ingester is the wedge. It's a forensic feature on the path; the wedge is pre-deployment stress testing.

## Files to read for code-level questions

- `src/drift/failures/detectors.py` — general detector library
- `src/drift/topologies/<name>/` — per-topology agents / events / detectors / prompts
- `src/drift/simulation.py` — the runner loop
- `src/drift/fork.py` — counterfactual replay
- `src/drift/analyze.py` — trace ingester
- `src/drift/server.py` — FastAPI backend
- `web/app.js`, `web/index.html` — frontend
- `TRACE_SCHEMA.md` — the trace format users need to emit
- `README.md` — how to run / scope
