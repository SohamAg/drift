# `repeated_identical_action`

> A single agent repeated the exact same action several times in a row
> with no change in output. The single-agent sibling of
> `infinite_handoff` — not two agents bouncing a task, just one agent
> stuck re-emitting the same tool call or turn.

## What fires

The detector walks the trace looking for a run of `min_repeats`
(default 3) **consecutive** steps where:

1. Every step in the run is the same agent — no other agent interleaves.
2. Every step's `update` dict is **exactly equal** to the first step's
   `update` dict (deep equality, not similarity).
3. The first step's `update` is non-empty.

It fires once per qualifying run (a 5-step identical run is one finding,
not three overlapping ones), with the finding's `timestep` at the last
step of the run.

Equality is deliberately strict. A retry that changes even one field —
an incrementing attempt counter, a timestamp, a slightly reworded
message — means the agent's output *did* change, which is a materially
different (and less clearly broken) situation than a byte-identical
repeat. Strict equality trades some recall (near-identical retries are
missed) for near-zero false positives on legitimate iterative refinement.

**Deliberately NOT flagged:**
- Consecutive steps from the same agent whose output changes each time
  — even a one-character difference breaks the match.
- The same agent acting identically more than once, but with another
  agent's step in between — not consecutive, so not "stuck."
- An agent that writes an empty update (`{}`) repeatedly — a
  pass-through or no-op node isn't "repeating an action."
- Fewer than `min_repeats` consecutive identical steps.

## What this typically means

The agent (or the tool/LLM call underneath it) is stuck in a loop that
produces no new information each iteration. Concrete manifestations:

- **Tool-call retry with no backoff logic**: a tool call fails silently
  (or the agent misreads its own tool's output) and the agent re-issues
  the exact same call, expecting a different result from identical
  input.
- **ReAct-style self-loop with a broken exit condition**: a node that
  routes back to itself keeps producing the same "thought → action"
  pair because whatever should change the state between iterations
  (new information, an incremented counter the node actually reads,
  user input) never reaches it.
- **Silent cost burn**: unlike a crash, this looks like "the system is
  working" from the outside — each repeat is a real LLM turn or tool
  call that bills and takes time, without the trace showing *any*
  anomaly other than the repetition itself.

## Sources

- **MAST 1.3 — Step Repetition.** The same mode `infinite_handoff`
  cites — that detector covers the two-agent alternating manifestation,
  this one covers the single-agent manifestation of the same underlying
  failure family.
  ([arXiv:2503.13657](https://arxiv.org/abs/2503.13657))
- **Anthropic engineering blog — "How we built our multi-agent research
  system"** — "minor failures compound catastrophically in stateful
  tool calls": a single stuck tool-call retry is exactly this class of
  failure, just not yet compounded into a larger system-level one.
  ([anthropic.com/engineering/multi-agent-research-system](https://www.anthropic.com/engineering/multi-agent-research-system))

## Concrete example

**Trace (fires on step 3):**

```
step  node     update
────  ───────  ──────────────────────────────────────────────
1     worker   {"tool_call": "search(query='pricing page')"}
2     worker   {"tool_call": "search(query='pricing page')"}
3     worker   {"tool_call": "search(query='pricing page')"}
```

Finding:

```
{
  "failure_type": "repeated_identical_action",
  "agents_involved": ["worker"],
  "timestep": 3,
  "summary": "agent 'worker' repeated the exact same action 3 times in a
              row (steps 1-3) with no change to its output"
}
```

The `worker` node issued the identical tool call three times running.
Whatever should have changed between calls — a different query, a
fallback strategy, an escalation — never happened.

## How to fix — architecture, not runtime

### 1. Make retries structurally visible to the node

If a node is allowed to retry, give it a counter it actually reads and
branches on, not just an external loop that re-invokes it blindly:

```python
def worker(state):
    attempt = state.get("search_attempt", 0) + 1
    if attempt > MAX_ATTEMPTS:
        return {"search_attempt": attempt, "escalate": True}
    return {"search_attempt": attempt, "tool_call": _search(state, attempt)}
```

A node that writes a changing `search_attempt` each time will no longer
produce byte-identical updates — and more importantly, it now has a
real exit condition instead of looping forever.

### 2. Route to a fallback after N identical attempts

```python
def route(state):
    if state.get("search_attempt", 0) >= MAX_ATTEMPTS:
        return "escalate_to_human"
    return "worker"
```

### 3. Vary the approach, not just the attempt count

If the retry exists because the first approach genuinely isn't working,
make the retry logic actually try something different (a reformulated
query, a different tool) rather than resubmitting the same input and
hoping for a different result.

### What NOT to do

- **Don't add a "watchdog agent"** that detects the loop and intervenes
  — it has no more context than the stuck agent did about why the
  approach isn't working.
- **Don't just raise `min_repeats`** to stop the detector from firing on
  your graph. If a real stuck loop exists, hiding it from drift doesn't
  stop it from burning cost in production.

## False positives and known limitations

- **Deliberate, intentional repetition looks identical to being stuck.**
  A polling node that legitimately re-checks the same condition with
  the same call shape (e.g. "has the external job finished yet?") will
  fire this detector even though the repetition is by design. If your
  graph has a legitimate poll loop, either make its update payload
  include something that changes each iteration (a poll counter,
  elapsed time) or treat a `repeated_identical_action` finding on that
  node as expected noise.
- **Exact-equality means paraphrased retries are missed.** An agent that
  rewords its tool call slightly each time (same intent, different
  phrasing) won't match — this detector catches the byte-identical
  case, not the semantic-identical case. The LLM judge's
  `emergent_decay` family is the closer fit for catching that softer
  pattern.
- **Text-only variant (`detect_from_text`) matches on normalized line
  text, not semantic content** — natural-language rewording of the same
  underlying action breaks the match, so it under-fires relative to the
  structured path on paraphrased retries. Documented recall-conservative,
  same as the rest of the library's text variants.

## Related detectors

- **`infinite_handoff`** — the two-agent sibling: strict alternation
  between exactly two agents with no state progress, instead of one
  agent repeating in place.
- **Judge `emergent_decay`** — catches the softer, paraphrased version
  of "the system isn't making progress" that strict equality here
  deliberately doesn't try to catch.

## Configuration

Optional kwarg to `detect()`:

| Arg | Default | Purpose |
|---|---|---|
| `min_repeats` | `3` | Minimum consecutive identical-update steps required to fire |

## Cost

Zero LLM calls. Runs in <1ms on typical traces. Safe to enable in CI.
