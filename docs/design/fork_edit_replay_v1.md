# Fork-edit-replay — v1 design

> Green-lit 2026-07-04. Scope locked. See
> [feature_ideas.md](../../memory/feature_ideas.md#fork-edit-replay-augmentations)
> in the auto-memory for the deferred features (bounded replay, consistency
> check, prompt editing) — v1 ships state edit + top-vs-bottom compare only,
> but the deferred features stay on the roadmap.

## What it is

A diagnostic tool for developers. Take a completed drift adapter run, pick a
step in the trace, edit its `state_after`, re-run the graph forward from
that state, and compare the resulting trace side-by-side with the parent.

Not a runtime feature. Not something end-users of the developer's MAS ever
see. Developer-facing debugging + hypothesis-testing surface.

## The primitive

```python
result = drift_test_fork(
    graph=my_compiled_graph,
    parent_result=original_adapter_result,
    fork_step=5,
    edits={"target_case_id": "case-99"},
    also_apply_at_initial=True,   # optional; top-vs-bottom compare
)
```

Returns a `ForkResult` with the forked branch's trace, final state, coord
findings, and (when `also_apply_at_initial`) the second branch created by
applying the same edits at the parent's initial state.

Parent's trace steps 1..(fork_step-1) are referenced, not copied.

## v1 scope — what ships

| Decision | v1 answer |
|---|---|
| Execution model | Rerun-from-state — invoke the graph with edited `state_at_step[fork_step]` as the initial input |
| Edit format | Sparse deep-merge — user passes `{field: new_value}` and we merge into fork-point state |
| Edit surface | State only (no prompt editing in v1 — deferred) |
| Bounded replay | `run_until="completion"` only (bounded modes deferred to v1.1) |
| Top-vs-bottom compare (feature 2) | `also_apply_at_initial=True` opt-in — v1 ships this |
| Consistency check / `path_dependence` finding (feature 3) | Deferred to v1.5 |
| Storage | Parent + fork results in same JSON file per run; fork branches on a `forks: []` field |
| API path | `drift_test_fork()` + `drift_test_fork_async()` in `src/drift/adapters/langgraph.py` |
| HTTP endpoint | `POST /api/adapter-fork` — server-side graph cache keyed by run_id |
| UI shape | "🔱 fork from here" button per trace step → modal with editable JSON `state_after` → three-column render (original / fork-edited / initial-edited when opt-in) |

## What's deferred and why

All deferred items live under
[feature_ideas.md → Fork-edit-replay augmentations](../../memory/feature_ideas.md#fork-edit-replay-augmentations).

- **Prompt editing** — needs graph reconstruction (LangGraph-specific). Ship
  state-only first, add if users ask.
- **Bounded replay** (`run_until={"steps": N}`, `run_until={"until_state": fn}`) —
  cost-control feature, cheap add later once base pipeline works.
- **Consistency check / `path_dependence` finding** — requires per-field
  whitelist + tiered comparator + judge integration. Full detector-shaped
  build; v1.5.
- **LangGraph checkpointer path** — rerun-from-state is semantically weaker
  (loses tool-call resume position) but works on ALL graphs. Add checkpointer
  path in v2 for graphs where it matters.

## Non-determinism handling

LLM-driven graphs produce different outputs each rerun. When comparing
forked branch vs parent branch:

- Use the tiered divergence cascade we already ship (t0 structural, t1
  exact, t2 noise band, t3 judge).
- A field that differs solely because of LLM wobble is dropped at t2.
- Fork-edit inherits the same noise handling as chaos perturbation.

## Known limitations

Honest list up front:

- **Rerun-from-state loses in-flight tool calls.** If the parent trace was
  in the middle of a tool call at fork_step, the fork restarts fresh — the
  tool call doesn't resume. Users needing that need the v2 checkpointer path.
- **Fork on a crashed parent isn't well-defined.** If parent's baseline
  crashed before step N, we can't fork at N. Error clearly.
- **Server-side graph cache is process-memory.** Restart the server, the
  graph reference is gone. Users have to re-run the parent to fork again.
  Fine for v1 (this is a dev tool, not prod).
- **UI is Adapter-tab only for v1.** Fork on chaos-perturbed traces (not
  just baseline) is easy to add later; v1 forks off baseline only.

## Test plan

- Synthetic: a deterministic graph, fork at step N, verify the new trace
  differs from parent in the way we edited.
- Verify coord detectors surface on forked runs.
- Cross-test: fork twice at the same step with different edits, verify both
  branches are captured independently.
- Integration: run drift_test on the adversarial contradictory MAS, then
  fork at step 2 with `verdict="approve"` — verify the resulting trace
  doesn't fire `contradictory_decisions`.
