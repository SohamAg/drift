# `contradictory_decisions`

> The same entity received two verdicts of opposing polarity in one trace.
> A canonical coordination failure — two decision-making paths landed at
> incompatible conclusions.

## What fires

The detector fires **once per contradicted entity** when:

1. An entity id (case_id, ticket_id, PR_id, target_case_id, …) is
   identifiable in a step's structured update, AND
2. That step (or another step targeting the same entity) carries a
   **positive** verdict (`approve`, `accept`, `pass`, `resolved`, `close`,
   `lgtm`, …), AND
3. Another step targeting the same entity carries a **negative** verdict
   (`reject`, `deny`, `fail`, `escalate`, `unresolved`, `hold`, …).

Both same-agent flip-flop (one agent approves then rejects the same case)
and multi-agent disagreement fire — a contradiction is a contradiction
regardless of who produced it.

**Verdict sources recognized (smart defaults, no config needed):**

- **Structured fields**: `verdict`, `decision`, `outcome`, `status`,
  `approval`, `resolution`, `review_result`, `approval_status`,
  `final_decision`, `action`.
- **Rationale text**: fallback scan of `rationale`, `reasoning`,
  `explanation`, `thought`, `message` for strong polarity tokens.

**Entity id sources recognized:**

- Structured fields: `case_id`, `ticket_id`, `pr_id`, `task_id`,
  `order_id`, `issue_id`, `entity_id`, `target_case_id`, `target_id`,
  `context_id`, `id`.

**Deliberately NOT flagged:**
- Two same-polarity verdicts (approve + approve) — that's agreement.
- Verdicts on **different** entities — no contradiction.
- A `pending`, `under_review`, or `n/a` value paired with a real verdict —
  the neutral doesn't count as opposition.
- Verdicts without an identifiable entity id — can't pair them safely.

## What this typically means

Two decision-making paths in your MAS reached incompatible conclusions on
the same object. Concrete manifestations:

- **Multi-reviewer disagreement**: one reviewer approves a PR, another
  rejects it, both write to state. Whichever fired last "wins" silently.
- **Verifier drift**: an approval-heavy verifier rubber-stamps a case that
  a downstream QA rejects. State now contains both a `verdict=approve` and
  a `verdict=reject`. Downstream logic is at coin-flip.
- **Same-agent flip-flop**: a supervisor prompted twice on the same case
  produces different verdicts because the prompt/context shifted between
  calls. Underlying: prompt is too underspecified, or the second call has
  polluted context.
- **Retry-then-invert**: agent fails once, retries, and the retry produces
  the opposite decision because the first failure is now in context and
  the LLM over-corrects.

## Why this is a coordination failure, not a normal disagreement

In a well-designed MAS the ownership of each decision is clear — one node
decides, others advise, and the "decide" write to state is monotonic
(never overwritten in-place). When two nodes both write a verdict to the
same entity, the topology allowed a race condition. That's a design bug
regardless of which verdict "wins" downstream.

## Sources

- **MAST 3.2 — specification ambiguity.**
  ([arXiv:2503.13657](https://arxiv.org/abs/2503.13657))
- **Anthropic engineering blog — "How we built our multi-agent research
  system"** — documents the "agent reaches different conclusions when
  re-prompted" pattern and the checkpoint-resume mitigation.
  ([anthropic.com/engineering/multi-agent-research-system](https://www.anthropic.com/engineering/multi-agent-research-system))
- **Cognition — "Don't Build Multi-Agents"** — Principle 2 warns:
  "Actions carry implicit decisions, and conflicting decisions carry bad
  results."
  ([cognition.com/blog/dont-build-multi-agents](https://cognition.ai/blog/dont-build-multi-agents))

## Concrete example

**Trace (fires on step 3):**

```
step  node        update
────  ──────────  ─────────────────────────────────────────────────
1     planner     {"context_id": "case-42", "note": "assigning"}
2     reviewer_a  {"case_id": "case-42", "verdict": "approve",
                   "rationale": "meets acceptance criteria"}
3     reviewer_b  {"case_id": "case-42", "verdict": "reject",
                   "rationale": "missing test coverage"}
```

Finding:

```
{
  "failure_type": "contradictory_decisions",
  "agents_involved": ["reviewer_a", "reviewer_b"],
  "timestep": 3,
  "summary": "entity 'case-42' received contradictory verdicts:
              POSITIVE at step 2 by 'reviewer_a' (verdict='approve');
              NEGATIVE at step 3 by 'reviewer_b' (verdict='reject')"
}
```

Both agents wrote to the same case's `verdict` field — the second write
silently overrode the first (in a standard LangGraph reducer). Whichever
one runs later "wins" downstream, but the trace now shows the earlier
agent's work was implicitly discarded.

## How to fix — architecture, not runtime

The Camp-A response applies especially strongly here. **Adding a
"tie-breaker agent" that catches contradictions is the wrong fix.** That
agent has less context than either original decider; it will guess. The
fix goes into the topology and state schema.

### 1. Single decider per verdict field

The most direct fix: only ONE node in your graph is allowed to write to
`verdict`. Other nodes emit `advice: {agent, position, rationale}` into
a `pending_advice` list; the decider node reads that list and produces
the authoritative verdict.

In LangGraph:

```python
class State(TypedDict):
    case_id: str
    pending_advice: Annotated[list, add]
    verdict: str  # only decider node writes this

def reviewer_a(state):
    return {"pending_advice": [{"agent": "a", "position": "approve", ...}]}

def reviewer_b(state):
    return {"pending_advice": [{"agent": "b", "position": "reject", ...}]}

def decider(state):
    advice = state["pending_advice"]
    # Deterministic rule OR one LLM call over all advice.
    verdict = _decide(advice)
    return {"verdict": verdict}
```

The topology makes the race impossible.

### 2. Monotonic verdict schema

If multiple nodes must write verdicts, make the field append-only:

```python
class State(TypedDict):
    verdicts: Annotated[list[Verdict], add]  # can't overwrite
```

Then a downstream `resolver` node reads all verdicts and produces the
final decision. Contradictions become visible instead of silent — but
they're still contradictions; the architectural fix is still (1).

### 3. Structured disagreement is fine — silent disagreement is not

If your product genuinely has parallel reviewers whose disagreement
matters (e.g. dual-approval for high-risk changes), the design should
explicitly represent it: a `disagreements: list[EntityId]` field, a
node that only fires when disagreements exist, a human-in-the-loop
escalation. Design the disagreement in. Don't leave it as an accidental
race.

### What NOT to do

- **Don't add a "reconciliation agent"** that reads both verdicts and
  picks one. Same reasoning as `hallucinated_reference`: you're stacking
  another LLM's guess on top of an already-broken coordination.
- **Don't retry the losing reviewer.** If the design allows two agents
  to disagree, retries won't fix the topology.
- **Don't silently pick "latest wins".** That's what a standard reducer
  already does; it's how the bug happens.

## False positives and known limitations

- **Positive/negative token lists are heuristic.** Domain-specific
  vocabulary won't be matched by default. Pass `verdict_fields=` to
  `detect()` for custom field names; the polarity lists themselves can
  be extended by monkey-patching `POSITIVE_TOKENS` /
  `NEGATIVE_TOKENS` for now (a proper config surface arrives with the
  user-guideline DSL).
- **The "action" field is treated as a verdict.** This matches native-sim
  topologies where `action=issue_refund` is a decision. If your graph
  uses `action` for something else (e.g. a tool name), the field
  configuration should be tightened via `verdict_fields=`.
- **Multi-step decisions.** If a decision is legitimately updated over
  time (`pending → in_review → approved`), the trace looks contradictory
  when we compare `pending` (neutral, ignored) to `approved` (positive)
  — that's fine, we don't fire. But `approved → rescinded` in the same
  trace WILL fire, and that may be a legitimate business flow. Add
  `rescinded` to a domain neutral list if so.
- **Text-only variant is recall-conservative.** Without structured data
  we require the same id to appear in two separate utterances with
  opposing polarity. Real contradictions expressed indirectly (paraphrase,
  reference-by-context) will not be caught.

## Related detectors

- **`verifier_always_approves`** — the opposite pathology: a rubber-stamp
  reviewer. Contradictory-decisions requires disagreement; verifier-always
  requires unanimous approval.
- **Judge `coordination_contradiction`** — the LLM-based version. Catches
  semantic contradictions the token matcher misses. Fires when this
  deterministic detector can't.
- **`hallucinated_reference`** — sometimes the contradiction is downstream
  of a hallucinated id (agent A approved a real case, agent B rejected a
  hallucinated one with the same shape). Look for a `hallucinated_reference`
  finding in the same trace first.

## Configuration

Optional kwargs to `detect()`:

| Arg | Default | Purpose |
|---|---|---|
| `verdict_fields` | `DEFAULT_VERDICT_FIELDS` | Which structured fields carry a verdict |
| `entity_id_fields` | `DEFAULT_ENTITY_ID_FIELDS` | Which structured fields carry an entity id |

Future extensibility (per the user-guideline DSL) will let you declare
domain-specific verdict vocabularies (`resolved` / `unresolved` for
support tickets, `merged` / `abandoned` for PRs) without editing the
detector code.

## Cost

Zero LLM calls. Runs in <1ms on typical traces. Safe to enable in CI.
