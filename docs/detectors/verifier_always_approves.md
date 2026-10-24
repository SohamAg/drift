# `verifier_always_approves`

> An agent assigned to review/verify/audit approves nearly everything it
> sees, with zero rejections. The verifier isn't verifying — it's
> rubber-stamping, and the safety layer the system was designed around
> is silently disabled.

## What fires

For each agent whose role matches a verifier pattern, the detector
collects every step that looks like a decision and classifies it as
`approve`, `reject`, or `unknown` (ambiguous or unrecognized — excluded
from the count). It fires once per qualifying agent when **all** hold:

1. The agent's role was declared as verifier-shaped (`roles_by_agent`),
   or — if no role was declared — its node/agent name matches
   `DEFAULT_VERIFIER_PATTERN` (`verif`, `review`, `audit`, `critic`,
   `check`, `approv`).
2. At least `min_decisions` classifiable decisions came from that agent
   (default 3).
3. The approval rate across those decisions is `>= approval_rate_threshold`
   (default 0.95).
4. Zero rejections were observed among them.

**How a step is classified**: structured decision fields win first —
`verdict`, `decision`, `approved`, `approval`, `status`, `result`,
`outcome`, `vote`, `review`, `assessment` — checked for a boolean value
or a recognized approval/rejection token (`approve`/`accept`/`pass`/
`lgtm`/`verified`/`valid`/`yes`/… vs. `reject`/`deny`/`fail`/`invalid`/
`blocked`/`no`/…). If no structured field is present, the detector falls
back to scanning the step's rationale/update text for an *unambiguous*
verdict token — if both an approval and a rejection token appear in the
same blob, the step is left unclassified rather than guessed.

**Deliberately NOT flagged:**
- A verifier with any rejections at all, regardless of how high the
  approval rate otherwise is — the detector specifically targets the
  *zero-rejection* pattern, not merely a high approval rate.
- Fewer than `min_decisions` classifiable decisions — too little signal
  to conclude rubber-stamping rather than a genuinely clean run.
- Steps whose decision value doesn't match any recognized token (e.g.
  `verdict: "escalated"`) — these count as `unknown`, not `approve`,
  and don't inflate the approval rate.

## What this typically means

The verification step exists in the graph's topology but isn't
providing real signal. Concrete manifestations:

- **Underspecified verifier prompt**: the verifier's instructions don't
  give it a concrete, checkable bar to apply, so it defaults to
  agreeing with whatever it's shown.
- **Verifier sees only the producer's framing**: if the verifier's input
  is the producer agent's own summary of its work rather than
  independently-derived evidence, there's nothing for the verifier to
  disagree with.
- **No cost/incentive to reject**: rejecting usually means triggering a
  retry or escalation path. If that path is missing or expensive, the
  verifier (and the system around it) has an implicit bias toward
  approval just to keep things moving.
- **The role exists for show**: a verification node was added to satisfy
  a design requirement ("multi-agent systems should verify") without
  giving it independent evidence or a real rejection criterion.

## Sources

- **MAST 3.x family** — "Incorrect Verification," together with
  **4.2/4.3** — "Lack of result verification" / "Lack of critical
  verification."
  ([arXiv:2503.13657](https://arxiv.org/abs/2503.13657))
- **Anthropic engineering blog — "How we built our multi-agent research
  system"** — cites this class among the harder bugs: minor verification
  failures compound catastrophically once a verifier is trusted
  downstream.
  ([anthropic.com/engineering/multi-agent-research-system](https://www.anthropic.com/engineering/multi-agent-research-system))

## Concrete example

**Trace:**

```
step  node      update
────  ────────  ──────────────────────────────────────────
1     drafter   {"draft": "..."}
2     verifier  {"verdict": "approve", "rationale": "looks correct"}
3     drafter   {"draft": "..."}
4     verifier  {"verdict": "approve", "rationale": "lgtm"}
5     drafter   {"draft": "..."}
6     verifier  {"verdict": "approve", "rationale": "ok to ship"}
```

Finding:

```
{
  "failure_type": "verifier_always_approves",
  "agents_involved": ["verifier"],
  "timestep": 6,
  "summary": "verifier 'verifier' approved 3/3 decisions (100%) with 0
              rejections — e.g. step 2 (verdict='approve'),
              step 4 (verdict='approve'), step 6 (verdict='approve')"
}
```

Three consecutive approvals with no rejection is enough to meet the
default threshold (3 decisions, 100% ≥ 95%, zero rejections). Whatever
the drafter produces, this verifier is currently guaranteed to wave it
through.

## How to fix — architecture, not runtime

### 1. Give the verifier independent evidence, not the producer's summary

```python
def verifier(state):
    # Bad: verifier only sees the drafter's own account of its work.
    # return _judge(state["draft_summary"])

    # Better: verifier re-derives evidence independently.
    evidence = _rerun_checks(state["draft"])
    return {"verdict": _judge(evidence)}
```

If the verifier's only input is "trust me, I did it right," there's
nothing for it to actually check.

### 2. Make the rejection path cheap and real

If rejecting triggers an expensive full restart, the system (and the
verifier, if it's LLM-driven and aware of that cost in context) will
implicitly avoid it. Make retry-on-reject narrow and cheap — re-run the
specific failing step, not the whole pipeline.

### 3. Give the verifier a concrete, checkable bar

Replace open-ended "does this look right?" prompts with specific pass/
fail criteria tied to the task (schema validity, required fields
present, specific acceptance tests). A verifier with nothing concrete to
check has nothing to reject.

### What NOT to do

- **Don't add a second verifier "to catch what the first one misses."**
  If the first verifier has no real signal, stacking another LLM call
  with the same underspecified prompt just adds cost without adding
  verification.
- **Don't force a minimum rejection rate.** Artificially rejecting some
  fraction of inputs to avoid tripping this detector defeats its
  purpose — the fix is giving the verifier real signal, not gaming the
  statistic.

## False positives and known limitations

- **Small, genuinely clean runs can false-positive at the margin.** A
  verifier that correctly approved 3 good-quality inputs in a row with
  zero rejections (because nothing was actually wrong) will trip this
  detector identically to a rubber-stamping one. The detector cannot
  distinguish "verifier is broken" from "verifier is working and
  everything it saw was fine" — raise `min_decisions` for graphs you
  expect to run with few, high-quality inputs per trace.
- **Role identification depends on naming or declared roles.** A
  verifier node named something idiosyncratic (and with no
  `roles_by_agent` entry) that doesn't match the default pattern won't
  be evaluated at all.
- **Token vocabulary is heuristic and English-centric.** Domain-specific
  or non-English verdict vocabulary won't be recognized by the default
  token lists; unrecognized values are treated as `unknown` and simply
  excluded from the count rather than misclassified.
- **Text-only variant (`detect_from_text`) is explicitly lower-precision.**
  It classifies any line mentioning a verifier-shaped word alongside an
  unambiguous verdict token, with no per-agent attribution — used for
  the MAST text-benchmark validation pass specifically as a weaker
  apples-to-apples baseline, not claimed to match the structured
  detector's precision.

## Related detectors

- **`contradictory_decisions`** — the opposite-shaped failure: instead
  of one verifier always agreeing, two decision paths land on opposing
  verdicts for the same entity.
- **Judge `verifier_rubber_stamp`-style findings** — the LLM judge can
  catch verification theater that doesn't fit this detector's
  structured-token model, e.g. approvals phrased without any of the
  recognized tokens.

## Configuration

Optional kwargs to `detect()`:

| Arg | Default | Purpose |
|---|---|---|
| `min_decisions` | `3` | Minimum classifiable decisions required before evaluating |
| `approval_rate_threshold` | `0.95` | Minimum approval rate to fire |
| `verifier_name_pattern` | `DEFAULT_VERIFIER_PATTERN` | Override the name-matching regex used when no role is declared |

## Cost

Zero LLM calls. Runs in <1ms on typical traces. Safe to enable in CI.
