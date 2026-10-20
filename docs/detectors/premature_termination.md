# `premature_termination`

> An agent acted but the required follow-up agent never acted afterward.

## What fires

You declare `required_followups={"trigger_agent": "required_agent"}` —
"whenever `trigger_agent` acts, `required_agent` must act within N steps
afterward" (`N` defaults to 5, overridable via `within_steps`). When
`trigger_agent` acts and `required_agent` never shows up in the following
window, the detector fires once for that trigger agent (not once per
occurrence — a trigger that fires repeatedly without follow-up is one
finding, not N).

**This detector requires config.** Without `required_followups` declared,
it always returns `[]` — there's no way to guess what "required" means
for an arbitrary graph, and guessing would produce false positives on
every graph that doesn't have this shape at all.

## What this typically means

Some step in your pipeline is supposed to trigger a next step —
escalation after triage, verification after a change, cleanup after a
remediation — and it didn't. The work that happened is real, but
whatever was supposed to happen next silently never did. This is
distinct from a crash (nothing errored) and distinct from the graph
simply ending (the graph may have terminated "successfully" while
skipping a required step).

Generic form of two patterns from this project's native-simulator era:
`silent_remediation` (an agent fixes something without telling anyone)
and `comms_lag` (an agent triages without ever communicating the
result) — both collapse into "action X happened, required action Y
didn't," which is what this detector checks.

## Sources

- **MAST 4.1 — termination errors**, per this project's own detector-
  mining notes (`detector_mining_findings.md`).
  ([arXiv:2503.13657](https://arxiv.org/abs/2503.13657))

## Concrete example

```python
result = drift_test(
    graph=my_graph,
    initial_state={...},
    required_followups={"triage": "escalate"},
)
```

```
step  node       update
────  ─────────  ─────────────────────────
1     triage     {"status": "flagged"}
2     responder  {"reply": "..."}
3     responder  {"reply": "..."}
```

```json
{
  "failure_type": "premature_termination",
  "agents_involved": ["triage"],
  "timestep": 1,
  "summary": "agent 'triage' acted at step 1 but the required follow-up
              'escalate' did not occur within 5 step(s) afterward"
}
```

## How to fix — architecture, not runtime

1. **Make the follow-up a structural requirement in the graph**, not an
   implicit expectation. If `triage` flagging something should always
   route to `escalate`, that should be a conditional edge, not a hope
   that the LLM-driven routing node gets it right every time.
2. **Add a terminal-state assertion.** If your graph has a notion of
   "done," check at that point whether any flagged-but-unescalated items
   remain, and refuse to terminate (or surface it loudly) if so.
3. **Treat this detector's config as a living spec.** Declaring
   `required_followups` is itself documentation of an invariant your
   graph is supposed to uphold — keep it next to the graph definition,
   not buried in a one-off test script.

**What NOT to do:** don't add a retry loop that keeps re-triggering
`trigger_agent` until `required_agent` shows up. If the routing is broken,
retrying the same broken routing just burns cost.

## False positives to watch for

- **Follow-up happens, but outside the window.** `within_steps` defaults
  to 5. A legitimately slow pipeline where the real follow-up happens at
  step 8 will fire. Raise `within_steps` for graphs with longer natural
  cadences between trigger and follow-up.
- **Conditionally-required follow-ups.** If `required_agent` is only
  supposed to run when some OTHER condition holds (not unconditionally
  after every `trigger_agent` action), this detector can't express that —
  it checks "every time," not "when X." That conditional-rule case is
  exactly what the (not yet built) user-guideline DSL is for.

## Related detectors

- **`infinite_handoff`** — opposite flavor of "no progress": looping
  instead of silently stopping.
- **Judge `gate_bypass`** — semantic, LLM-judged version of a required
  check being skipped; broader but costs tokens.

## Configuration

**Required** to do anything: `required_followups: dict[str, str]`
(trigger agent/node name → required follow-up agent/node name). Optional:
`within_steps` (default 5) — not currently exposed through `drift_test`'s
public kwargs; pass it directly if calling `premature_termination.detect()`
yourself.

Python-API-only for now (`drift_test(..., required_followups=...)`) — not
yet wired into `drift_test_fork`, the server's request models, or the web
UI. This is provisional, single-hint config standing in for the real
user-guideline DSL; expect this surface to change when that ships.

## Cost

Zero LLM calls. Runs in <1ms on typical traces. Safe to enable in CI —
but only actually checks anything once you've declared `required_followups`.
