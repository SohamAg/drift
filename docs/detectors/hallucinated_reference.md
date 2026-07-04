# `hallucinated_reference`

> An agent references an entity id that has never appeared in state and is
> not being defined by the current step. The reference is against thin air.

## What fires

The detector fires on the FIRST agent to mention an unknown entity id in
free-text content (rationale, log message, generated string). Subsequent
agents referencing the same id inherit the hallucination and are NOT
re-flagged — we surface the origin, not the propagation.

**Id shapes recognized (conservative on purpose):**

| Shape | Example |
|---|---|
| Prefix-numeric | `TICKET-42`, `PR-123`, `CASE-7`, `ISSUE-9` |
| Hash-numeric (≥2 digits) | `#42`, `#100`, `#987` |
| Word-prefixed with digits | `case-99`, `task_7`, `order 15`, `pr 3` |

**Deliberately NOT recognized:**
- Bare numbers (`42`) — too ambiguous with counts, timestamps, phone numbers
- Single-digit hashes (`#1`, `#2`) — indistinguishable from bullet lists
- Word forms with no digits (`case sensitive`, `task list`) — no id-shape

## What this typically means

You're seeing a **grounding failure** — the agent's output is producing
identifiers that don't correspond to any real entity in the system's
state. In a single-agent app this would show up as a hallucination in a
final response. In a multi-agent system it's more dangerous: **downstream
agents inherit the hallucinated id as if it were real, and continue
operating on it.** The failure cascades.

Common concrete manifestations:

- Supervisor delegates work referencing a ticket id that was never opened.
- A "closer" agent claims to close a case that doesn't exist.
- A router picks between real IDs and one hallucinated ID; downstream
  agents can't tell which is which.
- Retrieval-augmented agent cites a doc ID or chunk ID from an earlier
  cached search that isn't in the current context.

## Sources

- **MAST 2.4 / 2.6 — grounding failures / disagreement with retrieved
  context.** ([arXiv:2503.13657](https://arxiv.org/abs/2503.13657))
- **Anthropic engineering blog — "How we built our multi-agent research
  system"** describes the failure class of subagents proceeding on stale
  or absent context.
  ([anthropic.com/engineering/multi-agent-research-system](https://www.anthropic.com/engineering/multi-agent-research-system))
- **Cognition — "Don't Build Multi-Agents"** frames the same pattern from
  the architecture angle: when context isn't shared, subagents produce
  outputs against inconsistent worlds.
  ([cognition.com/blog/dont-build-multi-agents](https://cognition.ai/blog/dont-build-multi-agents))

## Concrete example

**Trace (the detector fires on step 2):**

```
step  node       update
────  ─────────  ─────────────────────────────────────────────────
1     planner    {"rationale": "starting queue triage"}
2     worker_a   {"rationale": "closing TICKET-42 as duplicate"}
3     worker_b   {"rationale": "TICKET-42 confirmed closed"}
```

The finding:

```
{
  "failure_type": "hallucinated_reference",
  "agents_involved": ["worker_a"],
  "timestep": 2,
  "summary": "agent 'worker_a' referenced id 'ticket-42' at step 2 but no
              prior state or definition mentions it (evidence: 'closing
              TICKET-42 as duplicate')"
}
```

`worker_b`'s reference is a propagation of the same hallucination and is
not flagged — the origin is the actionable event.

## How to fix — architecture, not runtime

For coordination failures like hallucinated references, runtime exception
handlers are almost always a bandaid. Fixes should go in the initial
design:

1. **Explicit entity context injection.** If a downstream agent needs to
   act on a case_id or ticket_id, pass the FULL entity record (not just
   the id) via structured state. If the entity isn't in state, the agent
   physically can't reference it — the failure mode is designed out.

2. **Constrain what agents can output.** If an agent's structured output
   schema requires `target_case_id` to match one of the currently-open
   cases, the LLM can be told (via prompt + a validation node) that
   inventing a new id is forbidden. This is a design change to the
   graph, not a runtime handler.

3. **Sharing full traces between sub-agents (Cognition's Principle 1).**
   If a subagent is spawned without seeing the parent agent's discovered
   context, it has to guess — and guesses that look like ids look like
   hallucinations to this detector. Share the full trace.

4. **Single-threaded topology for the id-dependent portion.** Cognition's
   recommendation: use a linear single-threaded agent for anything where
   entity identity matters. Parallel sub-agents on shared entities is
   the highest-risk pattern.

**What NOT to do:**

- Don't add a "validator" agent that checks references and rejects
  hallucinations. That's another agent that can hallucinate. You've
  multiplied the surface, not shrunk it.
- Don't retry the failing agent with more context. If the design lets it
  hallucinate, a retry can hallucinate a different id.
- Don't just filter hallucinated ids at the boundary. You'll silently
  drop legitimate references you didn't include in your allowlist.

## False positives to watch for

The detector is deliberately conservative on id-shape recognition to
minimize false positives. Real edge cases that can still trigger it:

- **Ids introduced by an LLM as CANDIDATES for creation** ("I propose
  opening TICKET-NEW-01"). The proposal is legitimate but there's no
  entity yet. **Workaround:** define the entity in the same step's
  structured state (`ticket_id: "TICKET-NEW-01"`) — the detector treats
  same-step definition as legitimate.
- **Non-id content that matches id patterns** — e.g. a git commit SHA
  that starts with `AB` (matches `AB-...` prefix?). The prefix regex
  requires 2-10 uppercase letters + hyphen + digits, so most commit
  SHAs (mixed hex) won't match. But if you have naming conventions
  that collide, expect false positives.
- **Ids in tool call arguments before the tool has returned.** If the
  agent's rationale describes calling a tool with `case_id=CASE-99` and
  the tool later creates that case, the detector fires on the rationale
  step. **Workaround:** the state update from the tool return should
  include `case_id` as a structured field.

If you hit any of these, please [file an issue](https://github.com/YOUR/drift/issues)
with the trace excerpt so we can tighten the pattern.

## Related detectors

- **`stale_state_reference`** (planned) — an id that used to exist but
  has been closed/removed. Similar shape, different diagnostic.
- **Judge `grounding_failure`** — the LLM-judge equivalent. Catches
  hallucinated non-id content the detector can't parse.
- **Judge `state_drift`** — related but broader; catches propagated
  errors of many kinds, not just hallucinated references.

## Configuration

None currently. This detector runs with defaults on every `drift_test`.
Future extensibility would come via the planned user-guideline DSL —
e.g. declaring custom id-shape patterns for domain-specific entities
(SKUs, model version numbers).

## Cost

Zero LLM calls. Runs in <1ms on typical traces. Safe to enable in CI.
