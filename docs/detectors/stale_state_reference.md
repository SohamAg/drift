# `stale_state_reference`

> An agent references an entity that had been closed/removed/resolved in
> an earlier step and not reopened. Distinct from `hallucinated_reference`
> — the entity existed, but its lifecycle has moved on.

## What the detector fires on

The detector fires once per stale reference. A stale reference is:

1. Step K writes a structured `case_id` (or ticket_id, task_id, …) plus a
   status field whose value is a **closure token** (`closed`, `resolved`,
   `archived`, `deleted`, `cancelled`, `completed`, `terminated`, …).
2. Some later step N > K writes the same entity id in a structured
   entity-id field.
3. No step between K and N (or step N itself) writes a **reopen token**
   (`reopened`, `reactivated`, `restored`, `unclosed`, …) for that entity.

Both same-agent and cross-agent stale references fire. If two later
agents both reference the closed entity, both fire (each is a distinct
downstream propagation event).

**Recognized entity id fields** (smart defaults, no config needed):
`case_id`, `ticket_id`, `pr_id`, `task_id`, `order_id`, `issue_id`,
`entity_id`, `target_case_id`, `target_id`, `target_ticket_id`,
`target_task_id`, `context_id`, `id`, `acting_on`, `operating_on`,
`processing_id`.

**Recognized status fields**: `status`, `state`, `lifecycle`,
`case_status`, `ticket_status`, `resolution`, `outcome`.

**Deliberately NOT flagged**:
- Same-step close + mention (the closer is describing what it's doing).
- Reference after reopen (lifecycle is legitimately reset).
- Reference to an entity that was never closed.
- Different entity id (no cross-contamination).

## What this typically means

The system's coordination broke down around the lifecycle boundary. Two
main patterns:

- **Parallel branch race**: LangGraph `Send()` fans out to workers in
  parallel; one worker closes the case while another continues processing.
  The parallel worker never sees the close event because they were
  spawned from the same snapshot.
- **Delayed downstream**: a slow downstream node was queued while the
  case was still open; by the time it runs, an upstream fast path has
  already resolved the case. The downstream still acts on its stale view.

These are exactly the failure modes MAST 1.5 describes and Cognition's
open problem #2 (cross-sibling discovery sharing) motivates. In
well-designed MAS the lifecycle is monotonic and single-threaded — one
node owns the state transition. When two paths can touch the same entity
and one closes it, the other can't safely continue.

## Sources

- **MAST 1.5 — parallel-agent state race.**
  ([arXiv:2503.13657](https://arxiv.org/abs/2503.13657))
- **MAST 4.1 — termination-order errors.** (same paper)
- **Cognition — "Don't Build Multi-Agents"** — open problem #2, on
  cross-sibling discovery sharing / context split between parallel
  sub-agents.
  ([cognition.com/blog/dont-build-multi-agents](https://cognition.ai/blog/dont-build-multi-agents))

## Concrete example

**Trace (fires on step 3):**

```
step  node      update
────  ────────  ──────────────────────────────────────────────
1     intake    {"case_id": "case-42", "content": "PR for feature X"}
2     closer    {"case_id": "case-42", "status": "closed"}
3     auditor   {"target_case_id": "case-42", "rationale": "processing"}
```

Finding:

```
{
  "failure_type": "stale_state_reference",
  "agents_involved": ["auditor", "closer"],
  "timestep": 3,
  "summary": "agent 'auditor' referenced entity 'case-42' at step 3, but
              it was closed at step 2 by 'closer' (status='closed')"
}
```

The `auditor` node had no way to know at spawn time that its target had
been closed. In a real graph this typically means the auditor was
spawned from a snapshot taken before step 2, or was reading state that
wasn't yet synced with the closer's write.

## How to fix — architecture, not runtime

For coordination failures like stale references, runtime exception
handlers are the wrong tool. The fix goes into the graph topology and
state schema.

### 1. Single owner per lifecycle transition

Only one node in the graph is allowed to close entities. Other nodes
that would touch the entity read its current `status` field and route
themselves out when they see it's closed:

```python
def auditor(state):
    case = state["cases"][state["target_case_id"]]
    if case["status"] in {"closed", "resolved", "archived"}:
        return {"skipped": True, "reason": "already-closed"}
    return {"audit_result": _audit(case)}
```

Same effect as the runtime check, but expressed *inside the node* as
part of its normal control flow — not as an exception handler wrapping
the whole graph.

### 2. Fan-out from a single lifecycle checkpoint

If you must fan out (parallel workers on shared entities), fan out
AFTER the lifecycle decision, not before. Serialize the close decision,
then dispatch the workers only for still-open entities:

```python
def lifecycle_gate(state):
    open_cases = [c for c in state["cases"] if c["status"] == "open"]
    return {"dispatch_queue": open_cases}

# fan-out happens only over dispatch_queue
```

### 3. Monotonic status schema

If your product genuinely has agents that may need to act on closed
entities (auditor reading history, reporter summarizing), model that
explicitly:

```python
class State(TypedDict):
    active_cases: Annotated[list[Case], add]     # can only append
    archived_cases: Annotated[list[Case], add]   # can only append
```

Auditors read `archived_cases`; workers read `active_cases`. The type
system prevents crossing the boundary.

### What NOT to do

- **Don't add a "state-sync agent"** that catches stale references and
  reissues actions. It has less context than the original agent; you're
  stacking guesses.
- **Don't wrap the whole graph in try/except**. The stale operation
  still commits to state before an exception is raised.
- **Don't silently skip stale references** at read time. If your graph
  is producing stale references, the topology has a race — fix the
  topology, don't paper over the symptom.

## False positives and known limitations

Honest about failure modes:

- **Auditors and reporters are treated as stale.** A node whose only job
  is to describe a closed entity (e.g. "log the fact that case-42 was
  closed") will fire this detector. Workaround: use a distinct field
  name (e.g. `logged_case_id` not `case_id`), or add the field to a
  domain-specific exclusion list via future DSL config.
- **Closure signals must be structured.** If your graph writes closure
  via free text ("we've closed this case"), the structured path misses
  it — only the text variant catches it, and less precisely.
- **The token vocabularies are heuristic.** Domain-specific closure
  terms ("shipped", "billed", "reimbursed") that aren't in the default
  set won't register. Pass `status_fields=[…]` to `detect()` to redirect
  which fields we scan; extend `CLOSED_STATUS_TOKENS` by monkey-patch
  for now.
- **Text-only variant** looks for close verbs + id pattern in later
  utterances. Real staleness expressed via implication ("we'll handle
  it next week") is missed. Documented recall-conservative.

## Related detectors

- **`hallucinated_reference`** — entity NEVER existed. Different
  diagnostic. Look for both together — a hallucinated close followed by
  a stale reference is a compound signal.
- **`contradictory_decisions`** — closer says "closed", reopener says
  "open" for the same entity — that's contradiction, not stale reference.
- **Judge `state_drift`** — the LLM-based version. Catches propagated
  staleness expressed in more complex ways than the token matcher can.
  Fires when this deterministic detector can't.

## Configuration

Optional kwargs to `detect()`:

| Arg | Default | Purpose |
|---|---|---|
| `entity_id_fields` | `DEFAULT_ENTITY_ID_FIELDS` | Which structured fields carry an entity id |
| `status_fields` | `DEFAULT_STATUS_FIELDS` | Which structured fields carry a lifecycle status |

Future extensibility (user-guideline DSL) will let you declare
domain-specific closure vocabularies (`shipped` for orders, `merged`
for PRs, `billed` for invoices) without editing detector code.

## Cost

Zero LLM calls. Runs in <1ms on typical traces. Safe to enable in CI.
