---
name: drift-context
description: Load drift's positioning, market context, and operating principles. Invoke at the start of any session that touches drift's positioning, pitch, product direction, or competitive analysis. Skip for pure code-level tasks — the code is the source of truth there.
---

# drift — context

## What drift is

**Pre-deployment testing for the multi-agent coordination layer.** Drift is an evaluation methodology specific to multi-agent AI systems — generate world-level chaos events that fire between agent steps, drive the system through them, detect coordination failures, fork any run at any step to test fixes deterministically.

Multi-agent systems fail in ways single-agent systems don't. Single-agent evaluation platforms (Galileo, Maxim, Braintrust, Ragas) test input variation and output quality — that doesn't reach the failure modes that only emerge when multiple agents observe shared state changing under them. Drift specializes in that surface.

Chaos engineering is the methodology word for what drift does. "Pre-deployment testing for multi-agent coordination" is the category positioning.

## The three pillars

These are the load-bearing claims. Any positioning statement should pivot off some combination of these. The first two are genuinely distinctive; the third is table stakes done well.

1. **World-level chaos events** — state mutations that fire between agent steps during a multi-step run (policy changes mid-flight, dependency outages, conflicting inputs, removed cases). Distinct from input-level edge cases (Galileo's synthetic data, Maxim's persona-based multi-turn simulation), which vary what enters the system at run start. World-level chaos varies what changes during execution — the failure surface unique to shared-state multi-agent coordination. Drift generates these automatically: at run start it inspects the user's `WorldState` subclass and auto-injects type-appropriate mutations against every field (bool flips, dict clears / fake keys, list reverses, numeric boundaries, string corruption), so users don't have to enumerate the chaos themselves. User-defined events still compose on top. See `src/drift/chaos/`.

2. **Counterfactual replay** — fork any run at any timestep with deterministic overrides (different seed, different prompts per role, disable an agent), compare branches to isolate causes. Causal analysis, not correlation. No published comparable feature elsewhere.

3. **Hybrid detection** — both halves ship. Deterministic Python rules for crisp / named patterns (contradictory action on same target, merge while blocked, stale reference) live in `src/drift/failures/detectors.py`. An LLM-judged detector in `src/drift/failures/judge.py` reads a sliding trace window and reports failures across the five families (`llm:coordination_contradiction`, `llm:grounding_failure`, etc.), so when deterministic rules don't speak a user's domain the judge still catches things. Rules are cheap, reproducible, CI-runnable; the judge adapts to any domain. Configurable via `drift.run(judge_llm=build_judge('openai'), judge_every=5)`. The hybrid framing is load-bearing because pure deterministic doesn't generalize across domains (see CASE_STUDY.md MAESTRO zero-fires) and pure judge is expensive + non-reproducible.

## Core failure families

Five categories of coordination failure drift is built to catch. The list is stable; specific detectors inside each family grow, get renamed, get retired as drift learns from real traces.

1. **Coordination contradictions** — multiple agents reach opposing decisions on the same target. Only possible with multiple agents; single-agent systems can't have this failure mode.
2. **Grounding failures** — actions reference targets that don't exist (fabrication) or no longer exist (stale, removed mid-step). Split because the fixes differ.
3. **State / memory drift** — agent acts on outdated world state. Rules/policies change mid-run; some agent kept operating off the old version.
4. **Emergent / system-level decay** — no single agent at fault, but the system is trending bad over time. Visible only across snapshot windows.
5. **Process / governance gates bypassed** — well-formed actions that aren't allowed. Required approvals skipped, blocked operations executed, required follow-up missed. The highest-stakes production failures live here.

New domains (legal review, content moderation, supply chain, etc.) add detectors that map to these same families. The structure holds across domains.

## Market context

Multi-agent systems are at the inflection point. Microsoft shipped MDASH (100+ agents) in May 2026. Academic foundations are being laid — MAST for failure taxonomy, MAESTRO for evaluation/observability. The eval-and-observability layer for LLM/agent apps has multiple well-funded incumbents; none currently treats multi-agent coordination as a primary focus, and none offers world-level chaos events or counterfactual replay. That's drift's lane.

Most-relevant adjacent players to track:

- **Galileo** — LLM observability + eval with LLM-as-judge metrics (Tool Selection Quality, Action Completion, Agent Efficiency, etc.) and an Insights Engine that produces prose summaries of multi-agent failures. Their distinctive technical bet is "Luna" distilled judge models that make LLM-judged eval cheap enough for continuous production monitoring + runtime guardrails. No world simulation, no counterfactual replay.
- **Maxim** — agent simulation + eval, including persona-based multi-turn user simulation (an LLM playing a user, conversing with your single agent). Broad enterprise feature set, open-source AI gateway (Bifrost), visual prompt chain editor. Their simulation is user-side; the environment doesn't change during a run.

Drift sits next to both, not against them. Galileo + Maxim handle input variation, output quality, and observability; drift handles environmental chaos and coordination-failure detection specific to multi-agent. A serious multi-agent team plausibly uses both layers.

Other adjacent spaces drift does not compete in:

- Orchestration (LangGraph, CrewAI, AutoGen, Conductor)
- Pure trace observability (Langfuse, Phoenix, LangSmith)
- Single-agent output eval (Braintrust, Ragas, Patronus)
- Infrastructure chaos (Gremlin, LitmusChaos — don't speak agent semantics)

## Positioning awareness

A few facts to carry without overclaiming:

- Drift's failure taxonomy is not the first published one. MAST (Cemri et al., Berkeley) named multi-agent failure modes with human annotations and an LLM-as-judge pipeline.
- Drift's evaluation harness is not the first published one. MAESTRO (Ma et al., KAUST) published a benchmark suite with telemetry across multiple MAS systems.
- Galileo's Insights Engine already detects some multi-agent coordination failures (their docs show their engine catching supervisor-loses-track-of-specialist patterns) — via LLM judges, postmortem. Drift's edge is methodology (mid-run world-level chaos + hybrid detection) and counterfactual replay, not "we invented the category."
- The defensible combination is: world-level chaos events + counterfactual replay + hybrid detection + BYOA/BYOE integration. No single piece is unique on its own.
- "Deterministic" alone is not a moat — it's one half of a hybrid approach. Pitching pure-deterministic invites comparison drift can't win (LLM judges adapt to any domain without code).

When prior art comes up in a conversation, treat it as aligned with drift's mission, not as a competitor. The field's overall credibility is shared.

## How to operate on drift

When working on drift in a session, defaults:

- **Frame around "pre-deployment testing for multi-agent coordination."** Narrower than "chaos engineering for multi-agent AI" but more defensible against Galileo / Maxim positioning. Use "chaos engineering" as the methodology word, not the category name.
- **Lead with the three pillars.** World-level chaos and counterfactual replay are the genuinely-distinctive ones; lean on those when comparison comes up. Hybrid detection is table stakes done well.
- **Treat the five failure families as the unit of analysis.** Specific detector names and counts are implementation; the families are positioning.
- **Distinguish demo from product.** The shipped simulator with hand-coded topologies is scaffolding for exploring the idea. The product proper is the bring-your-own-agents/environment path: a user defines their agents and world schema, drift drives them through chaos events, drift's detectors report what fired.
- **Acknowledge in-progress work plainly.** Overselling has more downside than acknowledging what's still being built.

Soft cautions:

- Avoid claiming "first" anywhere in the eval / taxonomy space — there's published prior art (MAST, MAESTRO) and well-funded incumbents (Galileo, Maxim).
- Avoid academic framings that don't survive concrete questions ("predict emergent behavior", "the emergence layer"). Drift surfaces failures via rules + judges; it doesn't predict.
- Don't conflate input-level edge cases with world-level chaos. Edge cases vary inputs; chaos events vary the state agents share mid-run. The distinction is what makes drift not just another eval platform.
- The trace ingester is a forensic capability on the path, not the wedge. The wedge is pre-deployment stress testing on user-supplied agents and environments.

## Where the build is heading

Drift today ships the simulator, the detector library, counterfactual replay, the web UI, and a trace ingester that runs detectors on externally-produced JSONL. The hand-coded topologies (support, code review, ops) are scaffolding to demonstrate the idea, not the product surface.

The next major capability area is letting users plug their own multi-agent system into drift — both sides of it:

- **Bring your own agents** — register agents built in LangGraph, CrewAI, AutoGen, or custom code; drift drives them through scenarios and captures their actions.
- **Bring your own environment** — define your own world state schema (the fields drift's detectors read), your own chaos event library (what stress conditions matter in your domain), and your own scenarios (the sequences that exercise coordination). The detector families stay the same; what gets watched changes per domain.

Closing both sides is what moves drift from "tool that demonstrates failure modes on its own simulator" to "tool that tests your multi-agent stack."

Empirical validation against published real-world multi-agent trace datasets (MAESTRO, MAST) is also in scope. The point is to show drift's detector families fire on coordination failures in real systems, not just on synthetic scenarios.

## Working risks

- Drift's positioning ("test your agents") slightly outruns what the product can demonstrate today (which tests drift's own agents). Treat that gap as a real constraint when making claims.
- The competitive landscape is tighter than first impressions suggest. Galileo + Maxim are well-funded incumbents whose adjacent surface could absorb world-level chaos primitives within a quarter. The moat is execution speed + becoming the canonical name for multi-agent coordination testing specifically.
- Multi-agent adoption is at the inflection but not yet mainstream. Drift may be early — the upside is owning the category as adoption matures.
- Drift's shipped detectors are domain-specific and didn't fire on MAESTRO's task-completion traces (see CASE_STUDY.md). The framework generalizes; the shipped detector instances don't. Closing that gap (auto-chaos + per-domain detector packs + hybrid LLM-judged detection for fuzzy cases) is on the roadmap.

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
