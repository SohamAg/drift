# `role_drift`

> An agent performs an action outside what its declared or inferred role
> should do — a verifier that executes, or an executor that renders a
> verdict.

## What fires

Two directions, both MAST 1.2 (Disobey Role Specification) in opposite
polarity:

- **Verifier executes.** An agent with role `verifier` (declared, or name
  matching `verif|review|critic|judge`) uses language shaped like an
  irreversible execution action: `merged`, `deployed`, `executed`,
  `closed`, `resolved`, `shipped`, `committed`, `paid`, `refunded`,
  `cancelled`, `deleted`.
- **Executor renders a verdict.** An agent with role `executor` (declared,
  or name matching `execut|worker|actor|implement`) uses language shaped
  like a verification verdict: `approved`, `rejected`, `verified`,
  `validated`, `lgtm`.

Role resolution: a **declared** role (`coordination_roles={"agent_name":
"verifier"}` — the same dict `drift_test_fork` already uses) is
authoritative and exclusive. Only when nothing is declared does the
detector fall back to matching the agent's own name against both
patterns. Fires **once per agent** (the first violation), not once per
occurrence.

## What this typically means

Verify-then-execute only works as a safety mechanism if the two stay
separate. If the verifier agent is the one pulling the trigger, there's no
independent check left in the loop — a bad call ships with nothing that
would have caught it. The inverse (an executor claiming to have "verified"
or "approved" something) means a "verified" label appears in your trace
or logs without anything resembling actual review behind it — dangerous
if anything downstream (a human, a report, an audit) trusts that label.

Common concrete manifestations:

- A code-review agent's rationale says "LGTM, merging now" — it reviewed
  AND merged in the same step, so a bad review has no second check.
- A worker agent's final message says "task complete, verified" with no
  separate verifier step anywhere in the trace.
- Role config drift over time: the graph was built with verify/execute
  separated, then a refactor collapsed them into one node without anyone
  updating the role declarations (or noticing).

## Sources

- **MAST 1.2 — Disobey Role Specification.**
  ([arXiv:2503.13657](https://arxiv.org/abs/2503.13657))

## Concrete example

```
step  node        update
────  ──────────  ──────────────────────────────────────────────
1     planner     {"task": "ship feature x"}
2     verifier    {"rationale": "looks good, merged it myself"}
```

With `coordination_roles={"verifier": "verifier"}`:

```json
{
  "failure_type": "role_drift",
  "agents_involved": ["verifier"],
  "timestep": 2,
  "summary": "agent 'verifier' (role=verifier) performed a 'merged'-shaped
              action at step 2, outside what a verifier should do"
}
```

## How to fix — architecture, not runtime

1. **Split the node.** If one LangGraph node both reviews and executes,
   that's the role collapse, not a prompt-wording issue. Make "verify"
   and "execute" two nodes with a conditional edge between them.
2. **Gate execution on an explicit upstream verdict field**, not on the
   verifier's free-text conclusion. If the executor node's entry
   condition is "a separate verdict field says approved," the LLM
   writing persuasive-sounding text in the verifier step can't skip the
   gate.
3. **Keep role declarations in sync with the graph**, not just the
   initial design. `coordination_roles` is cheap to pass on every
   `drift_test` call — treat a stale declaration as itself a signal the
   graph changed shape without anyone updating the model of it.

**What NOT to do:** don't add a second verifier to check the first
verifier's execution. That's the same collapse one level up.

## False positives to watch for

- **Legitimately dual-role agents.** Some systems intentionally give one
  agent both review and execute authority (e.g. a solo "operator" agent
  with no separate verifier at all). If that's a deliberate design, don't
  declare it as `"verifier"` or `"executor"` in `coordination_roles` —
  name it something the detector's patterns won't match (e.g.
  `"operator"`), or exclude this detector for that graph.
- **Name-pattern fallback on an unrelated agent.** An agent named
  `"reviewer_notes_formatter"` that only formats text (never performs an
  action) won't fire unless its output actually contains a forbidden verb
  — but a coincidentally verb-heavy summary could. Declaring the role
  explicitly removes the ambiguity.

## Related detectors

- **`verifier_always_approves`** — different failure: the verifier stays
  in its lane but never actually exercises judgment.
- **Judge `coordination_contradiction`** — broader semantic check; can
  catch role confusion the keyword list above doesn't recognize.

## Configuration

Optional: `coordination_roles: dict[str, str]` (agent/node name → role).
Same dict already used by fork-edit-replay's `coordination_roles` field.
Without it, the detector still runs via name-pattern fallback.

## Cost

Zero LLM calls. Runs in <1ms on typical traces. Safe to enable in CI.
