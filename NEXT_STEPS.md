# drift — next steps (parked)

Items committed to but not yet executed, in priority order. Captured here so
they don't get lost as other work moves forward.

## 1. Auto-chaos engine

**Problem:** the current Custom tab requires the user to define their own
chaos events (`SecurityFinding`, `DrugRecall`, etc.). That's backwards — if
the user already knows what chaos to inject, they could have written a unit
test for it. Drift's actual value-add only shows up when it *generates the
chaos itself* and surfaces coordination failures the user wouldn't have
thought to test for.

**What to build:**

a. **Generic chaos pattern catalog.** Ships built in; applies to any topology.
   Patterns to include:
   - tool latency spike
   - tool returns null / empty
   - tool returns contradictory data on consecutive calls
   - sudden state field flip (bool flips, counter resets, queue empties)
   - stale observation (agent sees a snapshot from N steps ago)
   - agent message duplication / reordering / drop
   - race condition (two agents act on the same target same step)
   - authority confusion (agent role renamed mid-run)

b. **State-schema-driven fuzzing.** Inspect the user's `WorldState` subclass
   at run time. For each field, auto-generate mutations:
   - `dict` → remove keys, add fake keys, duplicate keys
   - `bool` → flip mid-run
   - `int / float` → negative, zero, boundary values
   - `str` → empty, null, invalid
   - `list` → empty, duplicated entries, reversed

c. **API shape:** `drift.run(..., auto_chaos=True)` or
   `drift.run(..., auto_chaos="aggressive" | "moderate" | "off")`.
   Optionally `chaos_events=[...]` for user-supplied events on top.

d. **Reporting:** the result should include which auto-generated chaos
   events were injected, so users can see what drift tried and what fired.

**Why first:** this is the change that turns drift's pitch from "you bring
chaos, we detect failures" (a fancy test runner) into "we generate chaos
AND detect failures" (the actual chaos-engineering value prop). Sharpens
pillar 3 in the drift-context skill substantially.

**Once shipped:** update `.claude/skills/drift-context/SKILL.md` pillar 3
language to make auto-generation explicit.

## 2. Form-based test builder (level-2 UI)

**Problem:** the Custom tab asks users to paste Python in a textarea. That's
a developer-friendly stopgap, not a product UX. Real users (PMs, ML eng,
SREs) shouldn't have to write Python by hand to define a test.

**What to build:**

a. **Environment editor** — table editor: field name, type, default value.
   Generates the `WorldState` subclass under the hood.

b. **Agent editor** — add-an-agent form: name, role, decision-logic source
   (textarea OR upload OR "use template"). Generates the `@drift.agent`
   functions.

c. **Auto-chaos picker** — once item 1 ships, this becomes the *only* chaos
   surface the user sees. They pick intensity ("off / moderate / aggressive")
   and optionally exclude specific patterns. They don't author event classes.

d. **Test management** — saved test instances (state + agents + chaos config),
   with run history, ability to fork/edit/rerun.

e. **"Show generated code" toggle** for power users who want to see / copy
   the underlying Python.

**Scope:** ~2-3 focused days, achievable in a sprint. Forms generate the
same Python the Custom tab already eats; existing backend doesn't change.

**Why second:** the UI improvement matters less if the underlying capability
(auto-chaos) isn't there yet. Once it is, the form wizard becomes much more
compelling because "click Run" actually does work the user couldn't have
done themselves.

## 3. Framework adapters

Once the above two ship, real companies still need a frictionless way to
plug their *existing* LangGraph / CrewAI / AutoGen apps in without writing
per-agent wrapper functions. Build:

- `drift.from_langgraph(graph)` — auto-wraps every node as a drift agent
- `drift.from_crewai(crew)` — same for CrewAI
- (Later) AutoGen, MCP-Agent, AG2

Each adapter is ~half a day. LangGraph first since it's the most common.

## 4. UI consolidation

Today there are three parallel run paths: New Run tab (built-in topologies),
Custom tab (user code), Analyze tab (uploaded trace). Mental model is messy.
Long-term they should be one "New Test" flow with a source selector at the
top — built-in / custom / trace upload — all feeding the same runner →
detectors → results pipeline.

Not urgent. Defer until after items 1-3 ship, because the right consolidation
depends on what custom/auto-chaos ends up needing.

## 5. Mock LLM honoring prompt variant

Small polish (~45 min). Makes `naive` vs `hardened` actually diverge on
`--llm mock`, so the demo video doesn't require an API key. Currently the
mock handlers ignore the system prompt entirely.
