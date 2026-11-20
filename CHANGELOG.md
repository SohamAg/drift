# Changelog

Dated, one entry per shipped change. Newest first. Each entry says what
changed and, where a live run informed it, what the run showed — numbers
are rates over N rollouts, not verdicts.

## Unreleased — 2026-11-21

### Added
- **`ungrounded_number`** detector (library: 22). The final answer states a
  figure with 4+ significant digits that no human, system or tool message
  supplied, at the precision stated. It is not a computation over them, and
  not a well-known constant. It also fires when a tool call's arguments
  carry such a figure, and that call's result then stops counting as
  grounding. Measured over 110 saved runs: 3 fires, all real fabrications
  (the bare heuristic deferred in ADR-0008 fired 9 times, 3 real). Live,
  supervisor with research misrouted to math: 2/2. Curated corpus test in
  `tests/test_ungrounded_corpus.py`.
- Design decision records **ADR-0010** (ship `ungrounded_number`,
  superseding ADR-0008) and **ADR-0011** (a detector ships with its fire
  rate measured on saved runs, every fire read by hand).

## Unreleased — 2026-11-19

### Added
- **`model_wrong_tool_args`** (model layer): the right tool, a schema-valid
  call, one argument with a plausible wrong value: off by one, a bumped id or
  date, a query missing its last word, a flag flipped, a one-word name swapped
  for another named entity from the user's messages. Live (gpt-4o-mini, 3/3):
  the react agent searched "capital of", got nothing useful and answered
  "Paris" from its own knowledge. Right, and ungrounded.
- **`handoff_misroute`** (tool layer, handoff tools only): a supervisor's
  handoff to X runs agent Y. Model-layer faults cannot reach a supervisor's
  call (it happens inside its own subgraph); the subgraph's ToolNode can.
  In the default auto plan. Live, 2-agent supervisor: misrouted from research
  to math 2/2, it passed on math_expert's made-up populations although the
  handoff message said where control went. `empty_handoff` 2/2.
- **`new_event_crash`** gate rule: a perturbation that did not exist in the
  previous result crashes. `new_crash` alone now notes the crashes it could
  not judge.
- JUnit test cases are classed `drift.<graph>.<layer>`, so CI panels group
  them by fault layer.
- `drift init` ends drift.yml with a commented `chaos:` block for the graph's
  own tools and nodes (tool_error, model_wrong_tool_args, handoff_misroute,
  skip_node).
- **"By fault layer"** table in the evidence report, and the same rollup on the
  web Overview tab (`/api/overview` → `layers`). On this repo's saved runs:
  model faults diverged 8/8, tool 8/9 with 1 crash, control flow 2 applied
  (+1 not applied).

### Fixed
- CLI output piped on Windows (Git Bash, CI logs, `| tee`) is UTF-8, so "—"
  and "×" no longer arrive as "�". `PYTHONIOENCODING` still wins.
- **Faults that never landed are no longer counted as runs the graph coped
  with.** A model or control-flow fault the baseline gave no place to land
  was never run, yet it counted as "unchanged" and sat in the denominator of
  every rate. Three such faults turned a 50% crash rate into 20% and passed a
  `max_crash_rate=0.3` gate. Fixed everywhere it showed up:
  - the gate (left out of rates, with a note; a run where nothing landed now
    fails as `no_perturbations`);
  - regression snapshots (`drift history`, `--against regression`), where their
    ~0s runtimes also dragged the mean runtime down;
  - `AdapterResult.n_unchanged` (new `n_not_applied`), the saved JSON, the
    console summary and the Test tab;
  - the evidence report's headline tiles;
  - JUnit (`<skipped>`, not a pass) and the Markdown summary ("not applied").
  `n_graph_runs` drops them too, but still counts a tool fault that did not
  fire: that graph did run.
- **Installed drift serves its dashboard.** The web UI lived outside the
  package, so `drift serve` from a wheel answered "web/ directory not found".
  It now ships as package data, results go under the working directory when
  drift is not a checkout, and a CI job checks the built wheel from outside the
  repo.
- `drift init` no longer suggests forbidding a supervisor's own `transfer_to_*`
  handoffs as side effects.
- Install docs said `pip install drift`. That name on PyPI belongs to an
  unrelated package; the docs now install from GitHub.

### Changed
- Tests build agents with LangGraph v1's `langchain.agents.create_agent` and
  check every model fault on it and on the legacy `create_react_agent` (which
  langgraph-supervisor still uses). The 12 deprecation warnings are gone. The
  `validation` extra adds `langchain>=1.0`.

## Unreleased — 2026-11-15 (after 0.4.0)

### Added
- **GitHub Action inputs `model-chaos` / `flow-chaos`.** A third self-test job
  runs flow chaos on `examples/ci/demo_app` and requires
  `AutoChaos.skip_node[triage]` among the JUnit failures. It is green on GitHub.
- **Web Gate panel: compare against a saved run.** It picks one of the 60
  newest saved results and sends it as `previous`, so `new_finding` /
  `new_crash` / `regression` work in the browser as `drift gate --against`
  does. The CLI line appends `--against`.

### Fixed
- The evidence report labels model and control-flow faults ("· not applied"
  when they never landed, no mid-run step claimed for those). Its subtitle no
  longer says failures are "detected by a coordination-failure taxonomy judge":
  detection is the deterministic library plus drift.yml rules, and the judge is
  optional.

### Known
- `new_crash` matches perturbations by event name, so a crash on a perturbation
  that did not exist in the previous run (for example after turning on
  `--flow-chaos`) is not counted as new. That is by design: it is not a
  regression of a known input. `new_finding` still catches any new finding type.
  (Closed 2026-11-19 by `new_event_crash`.)

## 0.4.0 — 2026-11-15

### Added
- **Model-layer chaos** (`--model-chaos`, `drift_test(model_chaos=True)`,
  drift.yml `{tool: X, fault: model_*}`): `model_bad_tool_args`,
  `model_unknown_tool`, `model_duplicate_tool_call`, `model_dropped_tool_call`.
  The pending AI tool call is rewritten at the baseline checkpoint (same
  message id, written as the node that produced it) and the run resumes, so
  the ToolNode executes the corrupted call. [docs/checkpoint_chaos.md](docs/checkpoint_chaos.md)
- **Control-flow chaos** (`--flow-chaos`, `{skip_node: N}`): N is recorded as
  having run with no output where the baseline ran it next. ToolNodes are
  never skipped (an orphaned tool call is a sequence no real run produces).
  A fault the baseline gives no place to land is reported `fault_fired=False`
  with the reason, not run and not counted as coping.
- **Detectors (19 → 21):** `duplicate_tool_call` (one turn requested the same
  call twice and both executed; side-effect tools flagged) and `empty_handoff`
  (a handoff succeeded but its target never spoke before control moved on).
- **CI:** composite GitHub Action at the repo root (`uses: SohamAg/drift@main`,
  self-tested in this repo's CI against a pass and a real crash);
  `drift init [graph]` scaffolds drift.yml (tailored to the graph's tools and
  state fields) and the workflow; `--junit` (JUnit XML that agrees with the
  exit code); `--summary` Markdown, appended to `$GITHUB_STEP_SUMMARY`
  automatically; `--against PREVIOUS.json` with `new_finding`, `new_crash`,
  `regression` rules (skipped with a note when there is no previous result).
- **Report:** `drift report --fail-on [--against]` renders the gate verdict;
  a guideline-rules section lists every drift.yml rule with violated-on-k-of-N
  / held-on-all / could-not-evaluate.
- **Forks run guideline rules**, inherited from the parent run by default
  (`drift_test_fork(rules=...)`, `rules_yaml` on `/api/adapter-fork`).
- **Web UI:** Tool / Model / Control-flow checkboxes on Test, Import and Suite;
  chaos timing on Suite; model / control-flow pills; Test-tab forks carry the
  run's drift.yml.

### Fixed
- `conversation()` dropped all but one result of a parallel tool super-step
  (two Send tasks share one `state_after`), hiding tool results from every
  message-aware detector on graphs that make parallel calls.
- `silent_tool_failure` reported a recovery as a silent failure when the model
  fixed a wrong tool name: calling a tool the ToolNode suggested ("try one of
  [...]") now counts as the retry.
- Runtime regressions need +0.5 s as well as +50%: millisecond jitter made
  `test_second_tracked_run_compares_against_first` fail ~1 run in 3 and would
  have failed the new `regression` gate rule at random.
- Report methodology claimed every perturbation is applied mid-run; it now
  states the timing the file actually used.
- `gate.evaluate` carried two dead copies of the `guideline` rule branch.

### Live (gpt-4o-mini, `examples/adapters/run_drift_on_model_flow_faults.py`)
- react agent + mocked web_search, 3 rollouts: bad args and unknown tool —
  recovered 3/3 (no finding after the fix above); duplicate call — tool ran
  twice 3/3, `duplicate_tool_call` 3/3; dropped call — empty answer 3/3.
- 2-agent langgraph-supervisor, 2 rollouts: `skip_node[research_expert]` —
  the supervisor never noticed the empty research handoff and math_expert
  supplied populations from its own knowledge, `novel_tool_call` 2/2,
  `empty_handoff` names the cause (fires on 1 of the 92 saved runs under
  results/ — this one); `skip_node[supervisor]` — empty answer 2/2;
  `skip_node[math_expert]` — not applied (never ran in the baseline).

## 2026-11-11 (released in 0.4.0)

### Added
- **Mid-run chaos timing** — `drift_test(chaos_timing="mid_run")`, `drift run
  --timing mid_run`, UI select. State and message patterns are applied to
  the baseline's checkpoint at the step the planner scheduled them for
  (`update_state`, a fresh branch per rollout) and the run resumes, so the
  graph meets the fault with its own context in place. Results carry
  `chaos_timing`, `fork_step` and `timing_note`; resumed traces continue
  the baseline's step numbers; a routing function that raises on the
  perturbed checkpoint is reported as the crash it is. drift attaches a
  process-wide `MemorySaver` to a compiled graph that has none
  (`ensure_checkpointer`, which the web server now reuses). Also fixed:
  `_find_checkpoint` took the *newest* checkpoint with a given step, which
  after one fork could be the fork's own branch — it now takes the oldest.
  Two things the live run taught: a mid-run message perturbation must fork
  after a step whose checkpoint has no pending tool call (an AI tool call
  with no result is an invalid sequence for every chat API, and the graph
  re-routes on the appended message), and the edit must be written
  `as_node` the step's own node — inferred, `create_react_agent` routed the
  injected tool result to END and the resumed run was empty. Live
  (gpt-4o-mini react agent, cold conversation, 3 rollouts,
  `results/midrun_injection/`): seeded phrase injection 3/3 at both
  timings; seeded tool hijack **0/3 at the initial state, 3/3 mid-run** —
  the fork lands after the agent's own `web_search` round-trip, so the
  demand names a tool that is in context.
- **Tool-layer chaos** (`drift.chaos.tools`, `docs/tool_chaos.md`) — faults
  on one call to one tool *while the graph runs*, installed through
  LangGraph's `ToolNode.wrap_tool_call` for the duration of a single run
  (subgraphs included, previous wrappers kept and restored). Nine faults:
  `tool_error`, `tool_timeout`, `tool_rate_limit`, `tool_exception`,
  `tool_empty`, `tool_truncated`, `tool_stale`, `tool_inject_instruction`,
  `tool_hijack`, named `tool_timeout[web_search]` and excludable like any
  pattern. Discovered per graph and scheduled by the planner alongside the
  state patterns (five curated faults below `exhaustive`); results carry
  `fault_layer` and `fault_fired` so a fault the graph never reached is
  not mistaken for a pass. `drift.yml` gained `chaos:` — chaos asked for by
  name (`{tool, fault, on_call}` or `{pattern, field}`) always runs, first,
  outside the ceiling; typos are load-time errors. `drift_test(tool_chaos=)`,
  `drift run --no-tool-chaos`, request field `tool_chaos`. Live
  (gpt-4o-mini react agent, 3 rollouts, `results/tool_faults/`): every
  fault fired 3/3; error / timeout / rate-limit were papered over 3/3
  (`silent_tool_failure`); the mid-run injection was followed 3/3 and the
  mid-run hijack reached a tool call 3/3 with a correct final answer.
- **Guideline rules** (`drift.guidelines`, `docs/guidelines.md`) — the
  deterministic half of the user-guideline language. A `drift.yml` (or
  `.drift/guidelines.yml`) at the repo root declares `judge:` strings for
  the LLM judge and `rules:` that run for free inside the coordination
  library on the baseline and every perturbation: `forbid_tool`,
  `require_tool`, `tool_order`, `max_tool_calls`, `max_steps`,
  `final_answer` (regex), `no_message_match`, `state` (a safe expression
  over the final state), `python` (`pkg.mod:func`). Each has an `id`,
  optional `severity` / `why` / `agent` glob. Violations are findings
  `guideline:<id>`; an unevaluable rule is `guideline:<id>:error`, never a
  silent pass. `@drift.guideline("id")` marks a Python function that gets a
  `RunView` (`tool_calls`, `messages`, `final_state`, `final_answer`,
  `calls(glob, agent=)`). `drift_test(rules=…)` takes a spec, a path or a
  list; `drift run` picks `drift.yml` up automatically (`--guidelines`,
  `--no-guidelines`); the gate grew a `guideline` rule; the web UI's Test
  and Import tabs have a Guideline rules box, the Results summary lists the
  rule ids, and findings explain themselves in the glossary popover.

## 0.3.0 — 2026-11-09

### Added
- **`drift run` and `drift gate`** — the CI path. `drift run module[:build_fn]`
  imports a graph, runs drift_test with a preset (`quick` / `balanced` /
  `thorough` / `exhaustive`, every knob overridable), prints verdicts, writes
  the UI-shaped JSON with `--json`, exits 1 when a `--fail-on` rule trips.
  `drift gate results.json --fail-on …` decides from the artifact. Rules:
  `crash`, `recursion_limit`, `diverge`, `intermittent`, `finding`,
  `finding:<type>`, `baseline_finding`, `judge`, `max_crash_rate=`,
  `max_diverge_rate=`. `docs/ci.md`, `examples/ci/drift-gate.yml`.
- **`perturbation_rollouts=N`** on `drift_test`: every perturbed input runs
  N times; each `PerturbationResult` is the most severe rollout and carries
  the others, with `outcome_counts` ("crashed 1/3 · unchanged 2/3"),
  `finding_rollout_counts` and `is_intermittent`. Thorough preset runs 2×,
  Exhaustive 3×. UI shows k/N pills, an intermittent chip and a per-rollout
  panel.
- **Gate panel** in the web UI (Test / Import / Results): the CLI's rules
  evaluated on the run on screen via `POST /api/gate`, editable, persisted;
  imported runs get the exact `drift run` command that reproduces them.
- **Detectors** (15 → 19): `retry_amplification` (≥2× the baseline's tool
  calls or LLM turns under a fault — AgentChaos cost amplification),
  `novel_tool_call` (a tool the baseline never called, side-effect tools
  flagged — AgentDojo / ToolEmu / OWASP LLM06), `injected_tool_call_followed`
  (a planted instruction's canary reaches a tool call's arguments — the
  AgentDojo success criterion), `tainted_side_effect_call` (a side-effect
  tool called with a value lifted verbatim from an earlier tool result that
  the user never wrote — CaMeL-style taint tracking, no canary; 19 total).
- **Chaos pattern** `inject_tool_result_tool_hijack`: an injected tool result
  demanding a tool call with a canary argument.
- **Named sub-agents.** Subgraphs started from inside a tool call (deepagents'
  `task`) carry the sub-agent's name recovered from the dispatching call,
  disambiguated by task text when several are in flight. Live deepagents run:
  15/15 sub-agent steps named `researcher` / `critic` / `summarizer`.
- **Checkpoint forks from the UI.** The server attaches one process-wide
  saver to compiled graphs without one and the UI sends the parent thread id,
  so Fork & Replay resumes a real checkpoint (`update_state` + resume) and
  says so.
- Import and Test Suite tabs gained divergence mode and rollouts.
- The `quick` preset is now moderate intensity (≤4 perturbations, still no
  judge): light yielded one or two perturbations on a typical schema.
- `drift targets` lists the registry of real third-party MAS (dependency
  status, topology, source) for `drift run --target NAME`.
- `docs/detectors/README.md` — generated index of every detector
  (`scripts/gen_detector_index.py`); a test fails when it goes stale.
- Results tab renders a saved shaped run with the full Test-tab view (Gate
  panel, traces, rollouts) and hands it to Fork & Replay / Ask.

### Changed
- `silent_tool_failure`: weak hedges ("can't", "unable", "sorry") count only
  from the agent that made the failed call or in the final answer; strong
  failure vocabulary counts anywhere. Fixes the live miss where a different
  specialist's "I can't search the web" masked the supervisor's silent
  failure. Replay on four saved live runs: 4/4 (was 2/4).
- Regression snapshots record perturbations run, rollouts and intermittent
  count; crash / divergence rates are over perturbations run, not the
  schema's pattern catalogue (2 crashes of 4 run on a 14-pattern schema is
  50%, not 14%).
- Result shaping moved from `server.py` to `drift/adapters/serialize.py` so
  the CLI works without the web extra.
- `CoordinationDetectorContext` carries the baseline run's steps, so
  comparison detectors measure both sides the same way (a supervisor with
  `output_mode="last_message"` never writes sub-agent tool calls to the root
  state; the final state under-reports work).

### Fixed
- A non-off chaos intensity could schedule **zero** perturbations (light =
  8% per step over a 30-step horizon on a one-field schema) and could spend
  several of its few slots on the same pattern. The planner now always
  schedules at least one event and draws patterns without replacement.
  Seen live: `drift run --target supervisor_basic --preset quick` tested
  nothing and reported PASS. The gate additionally never passes vacuously:
  zero perturbations or a crashed baseline is a violation whenever rules
  are given.
- `drift report` on a web-UI / `drift run --json` file rendered blank event
  cells (it only knew the example-runner keys). Both shapes render; rows show
  fault class and k/N.
- The legacy `.stream()` accumulator (stubs without `stream_mode`) replaced a
  messages channel with each node's delta instead of appending; message-aware
  detectors were blind on hand-rolled graphs and agent↔tools alternation read
  as `infinite_handoff`.
- `AdapterForkRequest.parent_thread_id` was read but never declared, so
  pydantic dropped it.
- Validation scripts save results before pretty-printing and survive a
  cp1252 console (two paid runs were lost to a `UnicodeEncodeError`).

### Live validation (gpt-4o-mini, 3 rollouts each)
- 2-agent langgraph-supervisor: `injected_instruction_followed` 1/3
  (intermittent); `silent_tool_failure` 3/3; `novel_tool_call`
  (`web_search`) and `retry_amplification` (2.0×) 2/3 under the
  fabricated-turn and injection patterns; tool-hijack 0/3 (the supervisor
  has no directly callable tool).
- plain `create_react_agent` + mocked `web_search`
  (`examples/adapters/run_drift_on_react_hijack.py`): phrase injection
  followed 3/3; tool-hijack reached a real `web_search("DRIFT-CANARY-…")`
  call 3/3 when the payload named the tool, 0/3 when it could only say "one
  of your tools" — and the final answer was correct every time, which is
  exactly why the action-layer detector exists. With a mocked `send_email`
  tool added, the demand produced `send_email(body="DRIFT-CANARY-…")` 3/3
  and `injected_tool_call_followed`, `novel_tool_call` (SIDE-EFFECT) and
  `tainted_side_effect_call` all fired on each.
- Spend for the night ≈ $0.14.

## 2026-11-04

- Native LangGraph streaming (`stream_mode=["updates","values"]`,
  `subgraphs=True`): `state_after` is the reducer-applied state; subgraph
  steps visible, including tool-invoked ones; messages normalised to stable
  dicts; tool-call ids renumbered run-wide.
- Six message-aware chaos patterns with fault classes (crash / omission /
  value); dict walk two levels deep.
- Three message-aware detectors: `tool_call_loop`, `silent_tool_failure`,
  `injected_instruction_followed`.
- Per-run thread ids for checkpointed graphs; real checkpoint forks via the SDK.
- UI: fault-class chips, subgraph badges, recursion-limit crash label, fork
  execution mode.

Earlier history: see `git log` and the build-status notes referenced in
`README.md`.
