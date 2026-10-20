# `context_loss_on_handoff`

> The receiving agent's output references none of the specific content the
> previous, different agent just wrote.

## What fires

Agent N writes something concrete and trackable — an id-shaped token
(`TICKET-42`, `#101`, `case-7`) or a short new discrete state value (a
classification, decision, or verdict write). Agent N+1 — a DIFFERENT
agent, the very next step — produces output that mentions none of it.
Repeated across **at least 2** such handoffs in the trace, the detector
fires once, citing the first occurrence as evidence.

Deliberately conservative:
- **Requires a different agent.** One agent thinking out loud across
  several steps is not a handoff.
- **Requires >= 2 misses, not 1.** A single step where the receiver
  happened not to restate something is normal; a repeated pattern of zero
  cross-reference is the signal.
- **Requires concrete content.** Non-string values (bools, numbers) and
  common/short reused strings never become anchors — most steps write
  nothing distinctive enough to expect a citation, and flagging those
  would be noise, not signal.

## What this typically means

You're seeing **conversation history loss at the handoff boundary** — the
receiving agent is acting as if the sender's specific finding never
happened. In a system that's supposed to pass context forward (the whole
point of a multi-step pipeline), this means downstream decisions are made
on less information than the trace makes it look like they had.

Distinct from two detectors it can look similar to:
- **`hallucinated_reference`** — agent invents an id that was never real.
  This detector is the opposite: the id/value WAS real, upstream, and
  simply never made it into the next agent's output.
- **`stale_state_reference`** — agent cites something that used to exist
  but was removed. This detector doesn't require removal — the content is
  still live, just dropped.

Common concrete manifestations:
- A router hands off a classified ticket id; the handler agent's response
  never mentions which ticket it's handling.
- A research step surfaces a specific finding; the writer step produces
  generic text that could apply to any input.
- Parallel branches in a fan-out/fan-in topology drop each other's
  outputs at the merge point.

## Sources

- **MAST 1.4 — Loss of Conversation History.**
- **MAST 2.5 — Ignoring Other Agents' Input.**
  ([arXiv:2503.13657](https://arxiv.org/abs/2503.13657))

## Concrete example

```
step  node       update
────  ─────────  ─────────────────────────────────────
1     worker_a   {"rationale": "found issue in TICKET-42"}
2     worker_b   {"rationale": "proceeding with next task"}
3     worker_a   {"rationale": "found issue in TICKET-43"}
4     worker_b   {"rationale": "proceeding with next task"}
```

```json
{
  "failure_type": "context_loss_on_handoff",
  "agents_involved": ["worker_a", "worker_b"],
  "timestep": 2,
  "summary": "2 handoff(s) where the receiving agent referenced none of the
              sender's content; first at step 2: 'worker_a' wrote
              [ticket-42] but 'worker_b''s output mentions none of it"
}
```

## How to fix — architecture, not runtime

1. **Pass the specific finding forward in structured state**, not just in
   free text the next prompt may or may not attend to. If `worker_b`
   genuinely needs the ticket id, make it a required field in the state
   `worker_b` reads, not an implicit expectation that it'll notice it in
   `worker_a`'s rationale.
2. **Share the full trace with downstream agents (Cognition's Principle
   1)**, not just a summary. Summarization is exactly where specific ids
   and short discrete values get dropped.
3. **Add an explicit "cite your inputs" requirement** to a handoff node's
   prompt when the architecture genuinely requires free-text continuity —
   a stopgap, not a fix, but cheaper than a state redesign.

**What NOT to do:** don't add a "context checker" agent that verifies the
next agent referenced the prior one's output. That agent is itself
subject to the same failure mode, and you've added a step instead of
removing the gap.

## False positives to watch for

- **Deliberately independent agents.** Some topologies have agents that
  genuinely shouldn't reference each other (e.g. parallel specialists
  whose outputs get merged by a separate aggregator, not by each other).
  If that's the design, this detector will still fire on the merge
  boundary unless the aggregator explicitly cites each branch.
- **Synthetic/templated steps with no natural-language continuity.** A
  graph whose nodes mostly write short structured values with no shared
  vocabulary at all (flags, counters) rarely has anchors to miss — but
  when it does, those values won't look like "concepts" a human reviewer
  would expect carried forward, just a coincidental string match.

## Related detectors

- **`hallucinated_reference`** — origin of a NEW invented id.
- **`stale_state_reference`** — citation of a REMOVED id.
- **`infinite_handoff`** — no progress across repeated alternation; this
  detector can fire alongside it (no progress often co-occurs with no
  cross-reference) but measures something different.

## Configuration

Optional: `min_misses` (default 2) — raise it on long traces where chatty
back-and-forth between two agents is expected, if the default under- or
over-fires for your topology. Not currently exposed through
`drift_test`'s public kwargs — pass it directly if calling
`context_loss_on_handoff.detect()` yourself.

## Cost

Zero LLM calls. Runs in <1ms on typical traces. Safe to enable in CI.
