---
name: drift-context
description: Load drift's positioning, market context, and operating principles. Invoke at the start of any session that touches drift's positioning, pitch, product direction, or competitive analysis. Skip for pure code-level tasks — the code is the source of truth there.
---

# drift — context

## What drift is

**An opinionated open-source test DSL for multi-agent coordination.** Pip-install drift, declare a typed `WorldState`, plug in your agents, and drift auto-derives chaos events from your state schema, runs them between agent steps, and reports failures across a five-family coordination-failure taxonomy. A user-extensible guideline language lets you codify domain-specific patterns drift wouldn't catch on its own.

Microsoft validated the category in April 2026 with the open-source Agent Governance Toolkit (Agent SRE package). They built the **heavy SRE-for-agents infrastructure** — SLOs, error budgets, circuit breakers, canary deploys, runtime guardrails, a fixed catalog of nine fault templates. Drift is the **lightweight, opinionated, developer-facing alternative** — the test layer engineers reach for pre-launch, not the production SRE stack.

Chaos engineering is the methodology. "Multi-agent coordination test harness" is the category. **OSS-first** is the distribution model; cloud tier comes later when pull from users justifies it.

## Strategic commitment (decided 2026-06-10)

- **OSS-first, pip-installable, MIT.** No managed SaaS in v1. Match the distribution model that worked for Langfuse, DeepEval, and Phoenix.
- **Lightweight + opinionated**, not heavy. The Datadog-vs-ELK / Prefect-vs-Airflow analog. Microsoft's toolkit is heavy SRE infrastructure for the platform team; drift is a few-hundred-LOC test runner for the engineer who hasn't shipped yet.
- **Programmable, not opinionated-fixed.** User-extensible guideline language is the durable differentiator. Microsoft has nine fixed fault templates; drift gives the user a DSL to declare their own chaos + detection rules.
- **Multi-agent coordination specifically.** Not single-agent eval. Not general agent observability. The coordination-failure taxonomy is the vocabulary that single-agent tools don't have.

## The three pillars

These are the load-bearing claims. Pillar 1 is the verified-novel one — it survived adversarial verification cleanly. Pillars 2 and 3 are real but have prior art elsewhere; lean on them as part of the bundle, not as solo moats.

1. **Schema-driven auto-chaos.** Drift walks the user's typed `WorldState` (Pydantic / dataclass / equivalent) at run start, dispatches per-field-type mutations (bool flips, dict clears + fake-key injection + key removal, list reverses + duplications, numeric boundary attacks, string corruption), and schedules them between agent steps at intensity-scaled frequency. **No other surveyed tool walks a typed state schema to generate chaos.** Microsoft Agent SRE has nine hand-coded fault templates (network delays, tool timeouts, LLM degradation, etc.); Maxim has input-side fault injection; the rest don't have schema-walking chaos at all. This is drift's clearest moat — keep it as the headline of the pitch. See `src/drift/chaos/`.

2. **Counterfactual fork-and-replay with per-role overrides.** Fork any run at any timestep with deterministic overrides (different seed, different prompts per role, disabled agents), compare branches to isolate causes. **Prior art exists** — LangGraph (manual state edits, non-deterministic per their docs), Microsoft AGDebugger (CHI 2025 paper), Microsoft Agent SRE Replay Engine (April 2026), Laminar (single-span re-run with mocked prior calls), Paracosm `forkFromArtifact` (byte-equal swarm replay), Inject-Fork-Compare (arXiv 2509.13712). Drift's version bundles per-role prompt overrides + seed + disable-agent + branch-diff UI in one production-ready surface; it is not unprecedented. See `src/drift/fork.py`.

3. **Hybrid detection over a named coordination-failure taxonomy.** Deterministic Python rules + LLM-judged sliding-window detection in one workflow, across five named families (`coordination_contradiction`, `grounding_failure`, `state_drift`, `emergent_decay`, `gate_bypass`). Microsoft Agent SRE uses SLI types (quality metrics — task success rate, hallucination rate, etc.); Patronus has 20+ single-agent failure modes; Atla is judge-only over recorded traces. **No competitor ships a named multi-agent coordination failure family taxonomy with both deterministic and judge detectors in the same pipeline.** Configurable via `drift.run(judge_llm=build_judge('openai'), judge_every=5)`. The hybrid framing matters because pure deterministic doesn't generalize across user domains (see CASE_STUDY.md MAESTRO zero-fires) and pure judge is expensive + non-reproducible.

## The fourth pillar (planned, the wedge for YC)

4. **User-extensible guideline language** (next major build). A small DSL — natural-language strings, YAML rules, and optional Python — for users to declare their own coordination patterns to stress-test and detect. The killer differentiator vs Microsoft Agent SRE's fixed primitives. Three surfaces:
   - **Natural-language guideline strings** injected into the LLM judge's prompt — cheapest, lowest friction, directly improves MAST F1 baseline.
   - **YAML chaos + detection rules** for declarative team-shareable test packs.
   - **Python `@drift.guideline` decorators** for power users.

   Microsoft Agent SRE explicitly does not expose a user-extensible test language. Galileo customers complain that "opinionated workflows don't match domain-specific needs." This pillar addresses both. Once shipped, becomes the YC pitch headline ("drift is the programmable test language for multi-agent systems").

## Core failure families

Five categories of coordination failure drift's taxonomy covers. The list is stable; specific detectors inside each family grow, get renamed, get retired.

1. **Coordination contradictions** — multiple agents reach opposing decisions on the same target. Only possible with multiple agents.
2. **Grounding failures** — actions reference targets that don't exist (fabrication) or no longer exist (stale, removed mid-step). Split because fixes differ.
3. **State / memory drift** — agent acts on outdated world state.
4. **Emergent / system-level decay** — no single agent at fault, but the system trends bad over time.
5. **Process / governance gates bypassed** — well-formed actions that aren't allowed.

New domains add detectors that map to these families. The structure holds across domains.

## Competitive landscape (current as of June 2026)

### The single biggest threat

- **Microsoft Agent Governance Toolkit** ([github.com/microsoft/agent-governance-toolkit](https://github.com/microsoft/agent-governance-toolkit)) — MIT, public preview, April 2026. The Agent SRE package ships SLO engine (7 SLI types), Replay Engine with deterministic capture + multi-agent distributed trace reconstruction, Progressive Delivery (shadow + canary + auto-rollback), Chaos Engineering (9 fault templates: network delays, tool timeouts, LLM degradation, cost explosions, cascading failures, trust score manipulation, memory corruption, concurrent access races, delegation rejection), Cost Guard, per-agent circuit breakers, postmortem generation, Slack/PagerDuty/Teams webhooks. **20,000+ LOC, 1,257 tests, Microsoft distribution.** They overlap ~70% with what drift's chaos+replay+multi-agent pitch describes.
  - **How drift differentiates:** (a) schema-walking chaos (they have a fixed fault catalog), (b) named coordination-failure taxonomy (they have SLI metrics), (c) user-extensible guideline language (they have fixed primitives), (d) pre-deploy framing + lightweight DX (they are heavy SRE infra for prod). The differentiation is real but narrow; speed-to-canonical-name on the test-DSL angle matters.

### Adjacent commercial players

- **Galileo** (acquired by Cisco, May 2026) — LLM observability + 20+ evaluators + Luna distilled judge models (97% lower cost than LLM judges, sub-200ms) + Insights Engine that clusters multi-agent failure patterns + Conversation Replay (rewind + fork dialogues with modified prompts) + Agent Control runtime steering (March 2026). Customer complaints: trace-level debugging and failure-pattern *discovery* are weak; opinionated workflows don't match domain-specific needs; eval datasets don't grow from production failures.
- **Maxim AI** ($3M seed) — explicitly markets simulation-first pre-deployment testing, persona-based multi-turn user simulation, input-side fault injection (tool timeouts, malformed responses), re-run-from-step. Closest commercial positioning to drift's pre-deploy framing. SOC 2, HIPAA, in-VPC. No multi-agent coordination taxonomy; no schema-walking chaos.
- **Braintrust** ($80M) — observability + evals + datasets + automated prompt/scorer optimization (Loop). No multi-agent-specific features. SOC 2, HIPAA.
- **Patronus AI** ($17M) — purpose-trained judges (Lynx, Glider) + Generative Simulators (RL training env, not pre-deploy testing) + FinanceBench / BLUR benchmarks.
- **LangSmith / LangGraph Studio** (LangChain, $125M Series B) — observability + Playground (single-LLM-call re-run with modified prompt) + Fleet. LangGraph Studio ships fork+replay with manual state edits, non-deterministic per their docs. Framework lock-in.
- **AgentOps, Arize Phoenix, Langfuse, Helicone, HoneyHive, Comet Opik, W&B Weave, Atla, Athina, Confident AI / DeepEval** — observation-layer (most), per-span LLM-judge (Atla), pytest-style offline eval (DeepEval, 15.7k GitHub stars). None of them ship schema-driven chaos or per-role-override fork-and-replay.

### Closely-watched OSS / academic

- **AGDebugger** (Microsoft Research, CHI 2025) — academic multi-agent fork-and-replay with message editing.
- **Inject-Fork-Compare** (arXiv 2509.13712) — academic paper, 14-agent demo, retroactive event injection + deterministic state clone + A/B dashboard. Closest published work to drift's whole pitch.
- **Paracosm** (agentos.sh) — swarm sim with `forkFromArtifact(trunk, atTurn)` + byte-equal replay claim.
- **LangWatch Scenario** (OSS) — simulator-driven + RedTeamAgent + LangGraph/CrewAI integration. The closest OSS competitor for the "pre-deploy multi-agent simulation" lane.
- **Flakestorm** (OSS) — fault injection + recorded-input regression replay.

### YC W26 batch-mates in this category

- **Cascade** — building self-improving agent reliability. No shipped product visible at time of survey.
- **Sentrial** — production monitoring; SDK is forward-tracking only despite "fork from step" marketing.
- **Salus** — runtime proxy + policy block/repair; shadow-mode policy replay on historical tool calls.

### Categories drift does not compete in

- Orchestration (LangGraph, CrewAI, AutoGen — build agents, don't test them).
- Pure trace observability (Langfuse, Phoenix, LangSmith).
- Single-agent output eval (Braintrust, Ragas, Patronus).
- Voice/audio agent vertical (Sierra, Coval, Hamming, Roark).
- Security / red-team (General Analysis).
- Infrastructure chaos (Gremlin, LitmusChaos — don't speak agent semantics).

## Positioning awareness — facts to carry without overclaiming

- **Drift's failure taxonomy is not the first published one.** MAST (Cemri et al., Berkeley, March 2025) named multi-agent failure modes with human annotations and an LLM-as-judge pipeline. MAESTRO (Ma et al., KAUST, Jan 2026) published a benchmark suite. Cite both with respect, not as competitors.
- **Counterfactual fork-and-replay has substantial prior art.** LangGraph fork, Microsoft AGDebugger, Microsoft Agent SRE Replay Engine, Laminar, Paracosm, AgentOps time-travel marketing, Inject-Fork-Compare. Drift's version is more polished and bundled; it is not unprecedented. **Never say "no comparable feature elsewhere"** about replay.
- **Drift does not currently drive execution any differently than Maxim, Sierra, Coval, LangGraph, OpenAI Agents SDK, or LangWatch Scenario.** The "drift drives execution; competitors observe" framing is wrong and gets refuted in seconds. Replace with the narrower true claim: drift drives multi-agent coordination execution with mid-run state mutation derived from a typed schema, vs persona/user-side simulation against a single agent.
- **Pre-deployment failure testing is not drift's exclusive lane.** Maxim, Patronus Generative Simulators, Latitude, LangWatch Scenario, Galileo pre-prod evals all occupy this category in some form. Drift's narrower true claim is *schema-driven chaos against shared multi-agent state pre-launch*.
- **Galileo's Insights Engine already detects multi-agent coordination failures.** Their Conversation Replay supports fork+modified-prompt re-runs. The "Galileo has no world simulation, no counterfactual replay" line is stale post-Cisco acquisition; do not use it.
- **Drift is not first to chaos engineering for agents.** Microsoft Agent SRE shipped a chaos catalog three months before drift's strategic commit. Speak of chaos as the methodology word, not as drift's invention.
- **The defensible bundle is schema-driven chaos + per-role-override fork-and-replay + named coordination-failure taxonomy + user-extensible guideline DSL.** No single piece is unique; the bundle is.

## What we have NOT verified as unique

Honesty about gaps protects credibility in YC interviews.

- **Fork-and-replay primitive itself** — exists in prior art, multiple shipped tools and papers.
- **Pre-deployment chaos workflow** — Maxim, Microsoft Agent SRE, LangWatch Scenario, Latitude all sell this.
- **"Drift drives execution, others observe"** — refuted, many competitors drive execution.
- **Multi-agent observability** — drift doesn't compete here at all; defer to Langfuse/Galileo/Phoenix.

## How to operate on drift

When working on drift in a session, defaults:

- **Frame around "OSS test DSL for multi-agent coordination."** Microsoft built the heavy SRE infrastructure; drift is the lightweight programmable test layer. Use Datadog-vs-ELK / Prefect-vs-Airflow analogs when explaining the positioning to outsiders.
- **Lead with pillar 1 (schema-driven auto-chaos) as the headline.** It is the only adversarially-verified novel primitive. Pillar 4 (user-guideline language) becomes co-headline once shipped.
- **Acknowledge counterfactual fork-and-replay as part of the bundle, not as a unique moat.** Multiple shipped tools and papers do some version of it.
- **Treat the five failure families as the unit of analysis.** Specific detector names and counts are implementation; the families are positioning.
- **Distinguish demo from product.** The shipped topologies (support, code_review, ops) are scaffolding. The product is the BYOA path + the guideline DSL (once shipped).
- **Acknowledge in-progress work plainly.** F1 = 0.16 on MAST today. Overselling has more downside than acknowledging what's still being built.
- **Cite Microsoft Agent SRE as the named threat** in any competitive section. Pretending it doesn't exist makes drift look uninformed.

Soft cautions:

- Avoid claiming "first" or "only" in the eval / chaos / replay space.
- Avoid academic framings that don't survive concrete questions ("predict emergent behavior", "the emergence layer"). Drift surfaces failures via chaos + rules + judges; it doesn't predict.
- Don't conflate input-level edge cases with world-level chaos. Edge cases vary inputs at run start; world-level chaos varies the state agents share mid-run.
- The trace ingester is a forensic path, not the wedge. The wedge is the pre-deploy DSL.

## Where the build is heading

Drift today ships the simulator, the detector library (deterministic + LLM judge), counterfactual fork-and-replay, the web UI with a Detect tab (MAST demo), Custom tab (BYOA), Compare tab (fork diff), and a trace ingester. The shipped topologies (support, code_review, ops) are scaffolding.

Next major build sequence (OSS-first, ship-fast):

1. **User-guideline language v1.** Natural-language strings injected into the judge's prompt + a YAML chaos/detection rule format. ~1 focused week. Directly improves MAST F1 baseline + ships the pillar-4 differentiator.
2. **Framework adapters.** `drift.from_langgraph(graph)`, `drift.from_crewai(crew)`, `drift.from_openai_agents(...)`. ~2-3 weeks for the three. Removes the BYOA rewrap friction users are stuck on today.
3. **Curated chaos+guideline packs.** Three packs (supervisor-worker, retrieval-augmented, tool-using). Cuts time-to-first-failure from a half-day to five minutes.
4. **pytest plugin.** `pytest --drift-chaos` as a CI gate. The OSS distribution channel DeepEval used to grow to 15.7k stars.
5. **Causal link UI.** Explicit "failure F was triggered by chaos C" attribution in the Custom + Compare tabs. Makes the bundle pitch demonstrable in 30 seconds.

Not on the v1 path: managed SaaS, runtime guardrails, framework-native production observability, voice/audio, security red-team, purpose-trained judge models, deterministic LLM replay.

## Working risks

- **Microsoft Agent SRE is the single biggest competitive threat.** April 2026 launch, MIT, 20k LOC, 1,257 tests, Microsoft distribution. They overlap ~70% with drift's chaos+replay+multi-agent pitch. drift's remaining moat is schema-driven chaos + the coordination-failure taxonomy + the user-guideline DSL. Speed to canonical-name on the test-DSL angle matters more than any individual feature.
- **OSS distribution requires real momentum.** Without 1k+ GitHub stars in the first six months, drift is invisible. The pytest-plugin path is the cheapest distribution play; ship it early.
- **Drift's product testing today is below useful precision/recall.** F1 = 0.16 on MAST with a generic prompt. User-guideline language is the primary path to moving this number; per-mode prompts + anti-examples is the secondary path.
- **40% of agentic AI projects will be canceled by 2027** per Gartner — escalating costs, unclear business value. Frame drift's value as "test before you ship so your project doesn't get canceled," not "world-class observability."
- **Drift has zero funding and zero distribution** against multi-million-funded competitors. The wedge has to be sharp enough that the bundle wins on capability + DX, not on enterprise SLAs or sales motion.
- **Multi-agent adoption is at the inflection but not mainstream.** If multi-agent stays niche through 2027, drift's TAM narrative is weak. Mitigant: drift's test DSL still works on single-agent + tool-using setups; just expand scope when proving the wedge.

## File pointers

- `src/drift/chaos/` — auto-chaos engine (the pillar-1 moat)
- `src/drift/failures/detectors.py` — deterministic detectors
- `src/drift/failures/judge.py` — LLM-judged detector
- `src/drift/failures/mast_eval.py` — MAST evaluation helpers
- `src/drift/topologies/<name>/` — shipped scaffolding topologies
- `src/drift/simulation.py` — runner loop
- `src/drift/fork.py` — counterfactual fork-and-replay
- `src/drift/analyze.py` — trace ingester
- `src/drift/server.py` — FastAPI backend (Detect tab endpoints live here)
- `web/app.js`, `web/index.html` — frontend
- `TRACE_SCHEMA.md` — trace format
- `CASE_STUDY.md` — MAESTRO empirical results (honest zero-fires writeup)
- `CASE_STUDY_MAST.md` — MAST empirical results (F1 = 0.16 baseline)
- `NEXT_STEPS.md` — parked + prioritized build list
- `README.md` — how to run / scope
