# drift trace schema (v1)

The `drift analyze` command runs drift's coordination-failure detectors over
an action log produced by *any* multi-agent system. This document specifies
the trace format.

The goal: if your system can emit one JSON record per agent decision and one
per timestep of world state, drift can find named coordination failures in it
without you porting your agents into drift's simulator.

## Two accepted layouts

### Layout A — directory (drift's native log format)

A directory containing:

```
<run_dir>/
  snapshots.jsonl   required — one WorldState per timestep
  actions.jsonl     required — one Action per agent decision
  events.jsonl      optional — one EventRecord per exogenous event
```

This is exactly what drift writes to `runs/<run_id>/` when its simulator
runs. If your tool can emit logs in this layout, drift treats them as
first-class input.

### Layout B — single mixed JSONL

One `.jsonl` file where each line is a JSON object with a `"type"` field
of `"snapshot"`, `"action"`, or `"event"`. All other fields match the
per-record schemas below.

```jsonl
{"type": "snapshot", "timestep": 1, "customer_sentiment": 0.7, ...}
{"type": "action",   "timestep": 1, "agent_name": "reviewer-1", "kind": "approve_review", ...}
{"type": "event",    "timestep": 3, "name": "PolicyChange", ...}
```

## Per-record fields

### `snapshot` — world state at end of timestep

| field                     | type            | required | notes                                                                 |
| ------------------------- | --------------- | -------- | --------------------------------------------------------------------- |
| `timestep`                | int             | yes      | monotonic; one snapshot per step                                      |
| `customer_sentiment`      | float [0..1]    | no       | default 0.7. Drives `sentiment_collapse`                              |
| `refund_policy_version`   | int             | no       | default 1. Drives `policy_inconsistency`                              |
| `system_load`             | float [0..1]    | no       | default 0.3. Surface for stress-correlation analysis                  |
| `open_cases`              | dict[str, Case] | no       | id -> case object. Drives hallucination + stale-reference detectors   |
| `escalation_queue`        | list[CaseRef]   | no       | drives `queue_explosion` + `escalation_loop`                          |
| extra fields              | any             | no       | topology-specific state (e.g. PR status, incident severity). Allowed. |

`Case` shape: `{"case_id": str, "issue": str, "opened_at_step": int, "escalation_count": int, ...extra}`.

`CaseRef` shape: `{"case_id": str, "enqueued_at_step": int}`.

### `action` — one agent decision

| field                       | type        | required | notes                                                                                                            |
| --------------------------- | ----------- | -------- | ---------------------------------------------------------------------------------------------------------------- |
| `action_id`                 | str         | yes      | unique within the trace. Detectors cite this as evidence                                                         |
| `timestep`                  | int         | yes      | the step at which the agent decided                                                                              |
| `agent_name`                | str         | yes      | the actor                                                                                                        |
| `kind`                      | str         | yes      | free-form action kind. Detectors filter by these; vocabulary is topology-defined (see below)                     |
| `target_case_id`            | str \| null | no       | which case/PR/incident this action is about                                                                      |
| `rationale`                 | str         | no       | optional free text                                                                                               |
| `referenced_policy_version` | int \| null | no       | the policy version the agent *thought* was current. Mismatch with world's current version = `policy_inconsistency` |

#### Reserved action `kind` vocabularies per topology

These are the action kinds the shipped detectors specifically watch for. You
can emit other kinds too; they're just ignored by these specific detectors.

| topology      | kinds                                                                              | detector(s) that read it                                |
| ------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `support`     | `refund_approve`, `refund_deny`                                                    | `contradictory_refund`                                  |
| `support`     | `no_op`, `policy_update` (excluded)                                                | hallucinated/stale detectors skip these kinds           |
| `code_review` | `approve_review`, `reject_review`                                                  | `contradictory_review`                                  |
| `code_review` | `merge`, `security_block`, `security_clear`                                        | `security_bypass`, `merge_without_approval`             |
| `ops`         | `triage`, `diagnose`, `remediate`, `communicate`                                   | `silent_remediation`, `comms_lag`, `contradictory_diagnosis` |

### `event` — exogenous change

| field      | type | required | notes                                  |
| ---------- | ---- | -------- | -------------------------------------- |
| `event_id` | str  | yes      | unique within the trace                |
| `timestep` | int  | yes      |                                        |
| `name`     | str  | yes      | event type (e.g. `PolicyChange`)       |
| `summary`  | str  | no       | one-line description                   |

## What detectors fire

`drift analyze` runs the topology's detector list against the cumulative
action log + snapshot history at each timestep. The detector taxonomy is:

- **Coordination contradictions:** `contradictory_refund`, `contradictory_review`, `contradictory_diagnosis`
- **Grounding failures:** `hallucinated_reference` (case never existed), `stale_snapshot_reference` (case existed but was removed)
- **State / memory drift:** `policy_inconsistency`
- **Emergent system-level failure:** `sentiment_collapse`, `queue_explosion`, `escalation_loop`
- **Process gates bypassed:** `security_bypass`, `merge_without_approval`, `silent_remediation`, `comms_lag`

See [src/drift/failures/detectors.py](src/drift/failures/detectors.py) and
[src/drift/topologies/](src/drift/topologies/) for the exact rules each
detector enforces.

## Mapping from OpenTelemetry / MAESTRO-style spans

If your system emits OTEL spans (e.g. MAESTRO's Listing 1 format), the
mapping to drift's `action` record is:

| OTEL attribute                          | drift field                  |
| --------------------------------------- | ---------------------------- |
| `span_id` (or `gen_ai.tool.call.id`)    | `action_id`                  |
| `start_time` rounded to step bucket     | `timestep`                   |
| `gen_ai.agent.name`                     | `agent_name`                 |
| custom span attribute (your choice)     | `kind`                       |
| custom span attribute                   | `target_case_id`             |
| custom span attribute                   | `referenced_policy_version`  |

The `kind` field has no OTEL equivalent because it's drift's taxonomy on top
of the trace — you'll need to assign it during conversion based on which
agent/role emitted the span.

## Minimal end-to-end example

See [examples/traces/support_sample.jsonl](examples/traces/support_sample.jsonl)
for a 10-step trace that exercises four detectors. Run it with:

```
drift analyze examples/traces/support_sample.jsonl --topology support
```

Expected output: `contradictory_refund`, `hallucinated_reference`,
`policy_inconsistency`, and `sentiment_collapse` all fire.
