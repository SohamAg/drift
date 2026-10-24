# `infinite_handoff`

> Two agents hand a task back and forth without the state ever measurably
> advancing — the "you handle it" / "no you handle it" loop. Looks like
> activity; is wasted work.

## What fires

The detector looks at the trailing window of a trace (last
`max_look_back` super-steps, default 12) and fires once when **all** of
the following hold:

1. The same unordered pair of agents `{A, B}` strictly alternates —
   `A → B → A → B → …` — for at least `min_alternations` consecutive
   steps (default 4).
2. No new top-level key appeared in state across that run.
3. No string field that was empty at the start of the run became
   non-empty by the end.
4. No list/dict/set/tuple field grew in length.

All three "no progress" conditions are required together — any single
one alone has real false positives (chitchat between two agents is
normal; state growth unrelated to the handoff is normal). The
conjunction is the signature of an actual stuck loop rather than
legitimate back-and-forth collaboration.

The window is the *trailing* part of the trace deliberately — a graph
that made real progress earlier and only got stuck at the end shouldn't
be penalized for its earlier, productive steps.

**Deliberately NOT flagged:**
- A `A → B → A → B` run where state keeps gaining new fields (the agents
  are legitimately iterating, not stuck).
- More than two agents cycling (e.g. `A → B → C → A → B → C`) — this
  detector only looks for a strict two-party alternation; three-or-more
  agent cycles aren't covered.
- A run shorter than `min_alternations` — two bounces isn't a loop yet.

## What this typically means

The handoff topology let two nodes punt a task to each other without
either one actually resolving it — usually because neither has the
authority or information to finish, and nothing in the graph forces
either side to make progress or escalate. Concrete manifestations:

- **Underspecified routing condition**: a conditional edge sends work
  back to the same node pair because the routing function's exit
  condition never becomes true for this input.
- **Missing escalation path**: both agents correctly recognize "this
  isn't my job" and hand off, but there's no third node (human-in-the-
  loop, specialist, default handler) to break the cycle.
- **Context not actually transferred**: agent A hands to B expecting B
  to have what it needs, but the handoff doesn't carry the missing
  piece, so B hands straight back. This is Cognition's cross-sibling
  discovery-sharing problem — the loop keeps re-confirming the same gap
  instead of closing it.

## Sources

- **MAST 1.3 — Step Repetition.**
  ([arXiv:2503.13657](https://arxiv.org/abs/2503.13657))
- **Cognition — "Don't Build Multi-Agents"** — open problem #2,
  cross-sibling discovery sharing between agents that lack a shared
  context channel.
  ([cognition.com/blog/dont-build-multi-agents](https://cognition.ai/blog/dont-build-multi-agents))

## Concrete example

**Trace (fires on step 5, the end of a 4-step alternating run):**

```
step  node        update
────  ──────────  ─────────────────────────────────────────
1     triage      {"ticket_id": "t-9", "category": "billing"}
2     billing     {"note": "not a billing issue, see tech"}
3     tech        {"note": "not a tech issue, see billing"}
4     billing     {"note": "not a billing issue, see tech"}
5     tech        {"note": "not a tech issue, see billing"}
```

Finding:

```
{
  "failure_type": "infinite_handoff",
  "agents_involved": ["billing", "tech"],
  "timestep": 5,
  "summary": "agents 'billing' and 'tech' alternated for 4 consecutive
              steps (steps 2–5) with no new keys, no new non-empty
              fields, and no container growth"
}
```

Both agents overwrite the same `note` field each turn — no new field
appears, the string field isn't newly non-empty (it was already set at
step 2), and no container grows. Nothing routes this ticket anywhere
else.

## How to fix — architecture, not runtime

### 1. Add a decision budget to the routing edge

The simplest fix: track a bounce counter in state and force an
escalation path once it's exceeded.

```python
class State(TypedDict):
    bounce_count: int
    note: str

def route(state):
    if state.get("bounce_count", 0) >= 2:
        return "escalate_to_human"
    return "billing" if ... else "tech"
```

### 2. Require the handoff to carry new information

If agent B routinely hands back to A, make the graph's type system
force B to add something A didn't have — a diagnosis, a missing field,
a specific question — so "no progress" becomes structurally impossible
rather than a runtime hope:

```python
def tech(state):
    # must populate `diagnosis` before handing back
    return {"diagnosis": _diagnose(state), "note": "see diagnosis"}
```

### 3. Give the pair a shared escalation target

If neither agent can resolve the case, the topology should route to a
third node (specialist, generalist, human) after N bounces rather than
looping the same two nodes forever.

### What NOT to do

- **Don't add a "mediator agent"** between the two that tries to guess
  the right answer. It has the same information gap the other two had.
- **Don't just raise `min_alternations`** to make the detector stop
  firing on your graph — if a real loop exists, hiding it from drift
  doesn't fix the underlying topology.

## False positives and known limitations

- **Only detects strict two-party alternation.** A three-or-more-agent
  cycle (`A → B → C → A → …`) is not covered by this detector.
- **"Progress" is structural, not semantic.** A field that changes value
  without growing or going from empty to non-empty (e.g. `status:
  "pending" → "pending"` vs. a free-text note being reworded) won't
  register as progress — which is usually correct, but a graph that
  legitimately refines a field's *value* each turn without adding new
  keys could false-positive.
- **Text-only variant (`detect_from_text`) cannot check state progress
  at all** — raw transcripts carry no structured state. It fires purely
  on a long strict two-speaker alternation, so it is looser than the
  structured variant and documented as precision-loose, recall-
  conservative for the MAST validation pass.
- **Window size matters.** `max_look_back` bounds how far back the
  detector looks; a loop longer than the window but with a productive
  stretch inside it may be partially missed, by design (to avoid
  penalizing graphs that already escaped the loop).

## Related detectors

- **`subagent_fanout_excess`** — the opposite cost pathology: too much
  parallel work instead of too much serial bouncing. Look for both when
  auditing a graph's cost profile.
- **Judge-based coordination findings** — the LLM judge can catch
  semantically stuck loops that don't match this detector's strict
  two-party alternation (e.g. a three-agent round-robin, or a loop where
  state keeps "changing" in form but not in substance).

## Configuration

Optional kwargs to `detect()`:

| Arg | Default | Purpose |
|---|---|---|
| `min_alternations` | `4` | Minimum consecutive alternating steps required to fire |
| `max_look_back` | `12` | How many trailing steps of the trace to scan |

## Cost

Zero LLM calls. Runs in <1ms on typical traces. Safe to enable in CI.
