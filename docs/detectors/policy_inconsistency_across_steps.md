# `policy_inconsistency_across_steps`

> An agent referenced a stale value of a tracked policy/config field the
> state has since moved past.

## What fires

You declare `policy_field="policy_version"` (or whatever your state's
version/config key is named). The detector tracks that field's
authoritative value as the trace writes it — the most recent write is
"current." If a LATER step's text content (any string field, or its
rationale) mentions a PRIOR value of that field, that's a stale citation:
the agent is talking about — or acting on — a version of the policy that
the system has already moved past.

Checks against **any** prior version, not just the immediately-previous
one — citing `v1` three versions after it was current still fires.

**This detector requires config.** Without `policy_field` declared, it
always returns `[]`. Version values under 2 characters are excluded from
matching (a bare `"1"` is too likely to collide with unrelated text) —
if a trace never establishes a current value at all (`policy_field`
never appears in `initial_state` or any step's update), it also stays
silent, since there's nothing to compare against.

## What this typically means

Something in your system updated a policy, config version, ruleset, or
similar versioned field mid-run, and an agent downstream is still
operating on the old understanding. In a single-agent app this would
just be a stale-cache bug. In a multi-agent system it's sharper: **one
agent's local view of "current config" has drifted from the shared
state**, and nothing caught it until this detector did.

Common concrete manifestations:
- A supervisor updates a routing policy mid-run (new rules apply from now
  on); a worker spawned before the update still cites the old rules in
  its reasoning.
- A config-reload step bumps `ruleset_version`; a downstream verifier's
  rationale references the prior ruleset's criteria.
- Parallel branches: one branch sees the update, the other — started
  before the update propagated — doesn't.

## Sources

- **MAST 1.5 — configuration drift**, per this project's detector-mining
  notes (`detector_mining_findings.md`).
  ([arXiv:2503.13657](https://arxiv.org/abs/2503.13657))

## Concrete example

```python
result = drift_test(
    graph=my_graph,
    initial_state={"policy_version": "v1", ...},
    policy_field="policy_version",
)
```

```
step  node       update
────  ─────────  ───────────────────────────────────────────
1     updater    {"policy_version": "v2"}
2     worker     {"rationale": "applying rule from v1"}
```

```json
{
  "failure_type": "policy_inconsistency_across_steps",
  "agents_involved": ["worker"],
  "timestep": 2,
  "summary": "agent 'worker' referenced stale policy_version='v1' at step
              2; current value is 'v2'"
}
```

## How to fix — architecture, not runtime

1. **Push the current version into every agent's context explicitly** at
   the point of use, rather than relying on an agent "remembering" it
   from earlier in the trace. If the version is re-read from state every
   time it's needed, staleness becomes structurally impossible.
2. **Invalidate in-flight work on a version bump.** If a policy update
   mid-run genuinely should restart affected agents rather than let them
   finish under the old rules, make that explicit in the graph (a
   conditional edge back to re-entry), not implicit.
3. **Version-stamp agent outputs.** If an agent's output records which
   policy version it believed was current, a stale citation becomes
   detectable even outside drift — useful defense in depth.

**What NOT to do:** don't silently "correct" a stale reference at the
boundary by substituting the current version in. That hides the fact
that an agent reasoned under outdated assumptions; the output text and
the decision behind it may both need to be redone, not just relabeled.

## False positives to watch for

- **Legitimately historical references.** An agent summarizing "what
  changed since v1" is citing `v1` on purpose, not out of staleness. If
  your graph has a legitimate audit/changelog agent, this detector can't
  distinguish that from genuine drift — exclude that agent's role from
  consideration if it's chatty about history.
- **Short, generic version values.** Versions like `"1"` or `"a"` are
  excluded by the minimum-length guard specifically to avoid this, but
  even two-character values (`"v1"`) can coincidentally appear in
  unrelated text. Prefer more distinctive version strings
  (`"policy-2026-06"` over `"v1"`) if false positives show up in practice.

## Related detectors

- **`stale_state_reference`** — same shape (cites something that used to
  be true), different subject: an entity's existence, not a config
  version.
- **`role_drift`** — different axis of "the system's model of itself
  drifted," but about role boundaries rather than configuration state.

## Configuration

**Required** to do anything: `policy_field: str` (the state field name
holding the policy/config version). Python-API-only for now
(`drift_test(..., policy_field=...)`) — not yet wired into
`drift_test_fork`, the server's request models, or the web UI. This is
provisional, single-hint config standing in for the real user-guideline
DSL; expect this surface to change when that ships.

## Cost

Zero LLM calls. Runs in <1ms on typical traces. Safe to enable in CI —
but only actually checks anything once you've declared `policy_field`.
