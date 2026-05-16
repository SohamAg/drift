---
name: drift-context
description: Load drift's positioning, market context, and operating principles. Invoke at the start of any session that touches drift's positioning, pitch, product direction, or competitive analysis. Skip for pure code-level tasks — the code is the source of truth there.
---

# drift — context

## What drift is

**Chaos engineering for multi-agent AI systems.** Anyone deploying multi-agent AI should be able to use drift to stress-test their agents pre-deployment — inject chaos events into running coordination flows, detect named coordination failures, replay any run counterfactually with overrides to isolate causes.

Multi-agent systems fail in ways single-agent systems don't. The discipline to catch those failures before production hasn't been built yet. That's the lane.

## The three pillars

These are the load-bearing claims. Any positioning statement should pivot off some combination of these.

1. **Deterministic detectors** — pure functions over (action log, world state) that name specific coordination failures. Cheap, reproducible, runnable in CI on every commit. Distinct from LLM-as-judge approaches in the academic literature: drift's detectors are static-analysis-style, not interpretive.

2. **Counterfactual replay** — fork any run at any timestep with deterministic overrides (different seed, different prompts per role, disable an agent), compare branches to isolate causes. Causal analysis, not correlation.

3. **Chaos injection** — drive multi-agent systems through stress events (policy changes mid-run, dependency outages, conflicting inputs) and watch what breaks at the coordination level. Pre-deployment surfacing of failures, not postmortem analysis.

## Core failure families

Five categories of coordination failure drift is built to catch. The list is stable; specific detectors inside each family grow, get renamed, get retired as drift learns from real traces.

1. **Coordination contradictions** — multiple agents reach opposing decisions on the same target. Only possible with multiple agents; single-agent systems can't have this failure mode.
2. **Grounding failures** — actions reference targets that don't exist (fabrication) or no longer exist (stale, removed mid-step). Split because the fixes differ.
3. **State / memory drift** — agent acts on outdated world state. Rules/policies change mid-run; some agent kept operating off the old version.
4. **Emergent / system-level decay** — no single agent at fault, but the system is trending bad over time. Visible only across snapshot windows.
5. **Process / governance gates bypassed** — well-formed actions that aren't allowed. Required approvals skipped, blocked operations executed, required follow-up missed. The highest-stakes production failures live here.

New domains (legal review, content moderation, supply chain, etc.) add detectors that map to these same families. The structure holds across domains.

## Market context

Multi-agent systems are at the inflection point. Microsoft shipped MDASH (100+ agents in coordinated vulnerability scanning) in May 2026. Academic foundations are being laid — MAST for failure taxonomy, MAESTRO for evaluation/observability. Commercial chaos-engineering for multi-agent AI does not yet exist as a productized offering. That's the open lane drift is positioning into.

Adjacent spaces drift does not compete in:

- Orchestration frameworks (LangGraph, CrewAI, AutoGen, Conductor)
- LLM trace observability (Langfuse, Phoenix, LangSmith)
- LLM output eval (Braintrust, Ragas, Maxim)
- Infrastructure chaos (Gremlin, LitmusChaos — they don't speak agent semantics)

Drift's contribution sits *on top of* these layers, not against them. They're substrates and complements, not competitors.

## Positioning awareness

A few facts to carry without overclaiming:

- Drift's failure taxonomy is not the first published one. MAST (Cemri et al., Berkeley) named multi-agent failure modes with human annotations and an LLM-as-judge pipeline.
- Drift's evaluation harness is not the first published one. MAESTRO (Ma et al., KAUST) published a benchmark suite with telemetry across multiple MAS systems.
- Drift's edge is the *combination* of: deterministic detectors (cheap, reproducible, CI-runnable), counterfactual replay (no published prior art for this), chaos injection (pre-deployment surfacing, not postmortem), and the BYOA integration path that lets users plug drift into their own multi-agent stack.

When prior art comes up in a conversation, treat it as aligned with drift's mission, not as a competitor. The field's overall credibility is shared.

## How to operate on drift

When working on drift in a session, defaults:

- **Frame around "chaos engineering for multi-agent AI."** Concrete, understandable, has commercial precedent (Gremlin, Chaos Monkey). Better than vague framings like "emergence layer."
- **Lead with the three pillars.** They're load-bearing; everything else is supplementary.
- **Treat the five failure families as the unit of analysis.** Specific detector names and counts are implementation; the families are positioning.
- **Distinguish demo from product.** The shipped simulator with hand-coded topologies is scaffolding for exploring the idea. The product proper is the bring-your-own-agents path: a user wraps their existing multi-agent system, drift drives it through chaos scenarios, drift's detectors report what fired.
- **Acknowledge in-progress work plainly.** Overselling has more downside than acknowledging what's still being built.

Soft cautions:

- Avoid claiming "first" in the failure-taxonomy or eval-framework space — there is published prior art.
- Avoid academic framings that don't survive concrete questions ("predict emergent behavior", "the emergence layer"). Drift surfaces failures via deterministic rules; it doesn't predict.
- The trace ingester is a forensic capability on the path, not the wedge. The wedge is pre-deployment stress testing on user-supplied agents.

## Where the build is heading

Drift today ships the simulator, the detector library, counterfactual replay, the web UI, and a trace ingester that runs detectors on externally-produced JSONL. The hand-coded topologies (support, code review, ops) are scaffolding to demonstrate the idea, not the product surface.

The next major capability area is letting users plug their own multi-agent system into drift — both sides of it:

- **Bring your own agents** — register agents built in LangGraph, CrewAI, AutoGen, or custom code; drift drives them through scenarios and captures their actions.
- **Bring your own environment** — define your own world state schema (the fields drift's detectors read), your own chaos event library (what stress conditions matter in your domain), and your own scenarios (the sequences that exercise coordination). The detector families stay the same; what gets watched changes per domain.

Closing both sides is what moves drift from "tool that demonstrates failure modes on its own simulator" to "tool that tests your multi-agent stack."

Empirical validation against published real-world multi-agent trace datasets (MAESTRO, MAST) is also in scope. The point is to show drift's detector families fire on coordination failures in real systems, not just on synthetic scenarios.

## Working risks

- Drift's positioning ("test your agents") slightly outruns what the product can demonstrate today (which tests drift's own agents). Treat that gap as a real constraint when making claims.
- Multi-agent adoption is at the inflection but not yet mainstream. Drift may be early — the upside is owning the category as adoption matures.
- Large adjacent players (LangSmith, Maxim, Langfuse) could ship multi-agent chaos primitives. The moat is execution speed and becoming canonical.
- Drift's detectors have not yet been validated empirically against external real-world traces. The validation work is on the roadmap.

## File pointers

- `src/drift/failures/detectors.py` — general / cross-topology detectors
- `src/drift/topologies/<name>/` — per-topology agents / events / detectors / prompts
- `src/drift/simulation.py` — runner loop
- `src/drift/fork.py` — counterfactual replay
- `src/drift/analyze.py` — trace ingester
- `src/drift/server.py` — FastAPI backend
- `web/app.js`, `web/index.html` — frontend
- `TRACE_SCHEMA.md` — trace format
- `README.md` — how to run / scope
