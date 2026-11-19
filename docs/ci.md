# Drift in CI — `drift run` and `drift gate`

Drift is a pre-deploy gate. The CLI makes that literal: run the chaos
suite against your graph, write the result JSON, and fail the build on
the rules you choose.

## One command

```bash
pip install drift[langgraph]        # + drift[openai] for the judge / tiered cascade
drift run my_app.graph:build --preset quick --fail-on crash,finding:injected_instruction_followed
```

`my_app.graph` must expose `build()` (returns the compiled graph) and
`initial_state()` (returns the dict your graph is invoked with). Name a
different builder with `module:fn`, a different state function with
`--initial-state-fn`, and pass `--query "…"` to call `initial_state(query)`.

Exit code `0` = every rule passed, `1` = a rule was violated, `2`/`SystemExit`
= usage or build error (the message says what).

Running one of the real third-party systems drift is validated against
instead of your own module: `drift targets` lists the registry
(`examples/targets/`) with dependency status; `drift run --target
supervisor_basic --preset quick` runs it.

## Two steps: run where the key is, gate anywhere

```bash
# on the runner that has OPENAI_API_KEY (the thorough preset runs the judge
# and repeats every perturbation twice so intermittent failures show as k/2)
drift run my_app.graph --preset thorough --json drift-results.json -q

# later / elsewhere — decide from the artifact, no graph, no key
drift gate drift-results.json --fail-on crash,recursion_limit --fail-on max_diverge_rate=0.25
```

The JSON is the exact shape the web UI renders: drop it in the Results tab,
or render a self-contained HTML evidence report with
`drift report drift-results.json`.

## Rules

| Rule | Fails when |
|---|---|
| `crash` | any perturbation crashed |
| `recursion_limit` | any perturbation looped until LangGraph's recursion limit stopped it |
| `diverge` | any perturbation diverged after the noise cascade |
| `intermittent` | a perturbation's repeated rollouts disagreed (`--rollouts N`, N > 1) |
| `finding` | any coordination detector fired, on the baseline or a perturbation |
| `finding:<type>` | that detector fired, e.g. `finding:silent_tool_failure` |
| `baseline_finding` | a detector fired on the **unperturbed** run — a bug that needs no chaos |
| `guideline` | a rule from your `drift.yml` was violated, on any run (see [guidelines](guidelines.md)) |
| `judge` | any LLM-judge finding |
| `max_crash_rate=0.2` | crashed ÷ perturbations above the fraction |
| `max_diverge_rate=0.5` | diverged ÷ perturbations above the fraction |
| `new_finding`, `new_crash`, `new_event_crash`, `regression` | compared with an earlier result — see [`--against`](#fail-on-what-got-worse---against) |

Rules are comma-separable and `--fail-on` is repeatable. Rates are over the
perturbations actually scheduled, so `--max-perturbations 4` is judged on
those four.

## Presets

Same four as the web UI; any knob overrides the preset.

| Preset | Chaos | Divergence | Rollouts | Judge | Cost |
|---|---|---|---|---|---|
| `quick` (default) | moderate, ≤4 | exact | 1 | off | free, seconds |
| `balanced` | aggressive, ≤6 | tiered, 3 baseline | 1 | openai | ~$0.01 |
| `thorough` | aggressive, ≤12 | tiered, 5 baseline | 2 | openai | ~$0.20 |
| `exhaustive` | every pattern | tiered, 5 baseline | 3 | openai | $1+ on wide schemas |

`--rollouts N` is the knob that turns "the injection was followed" into
"the injection was followed 1 of 3 times". One LLM run is a sample.

## Start in one command

```bash
drift init my_app.graph:build
```

writes `drift.yml` (one active rule, plus commented suggestions built from
your graph's real tool names and state fields) and
`.github/workflows/drift.yml` (the action below, pointed at your graph).
Existing files are left alone unless you pass `--force`.

## GitHub Actions

The repo root is a composite action:

```yaml
- uses: actions/checkout@v4
- run: pip install -e .            # your app
- uses: SohamAg/drift@main
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}   # only for presets above quick
  with:
    graph: my_app.graph:build
    preset: balanced
    fail-on: crash,recursion_limit,finding:injected_instruction_followed
    args: --rollouts 2
```

It runs `drift run`, writes `drift-results.json` and `drift-junit.xml`,
uploads both as the `drift-results` artifact, puts a summary on the job
page, and sets `outputs.passed`. Inputs: `graph` or `target`, `preset`,
`fail-on`, `against`, `timing`, `model-chaos`, `flow-chaos`, `seed`, `args` (extra `drift run`
flags), `results-path`, `junit-path`, `python-version` (empty = use the
runner's), `install` (a pip requirement for drift; default is the
action's own checkout), `extras`, `upload-artifact`.
`.github/workflows/action-selftest.yml` in this repo runs it on every
change, expecting a pass on one job and a real crash failure on another.

Without the action, `examples/ci/drift-gate.yml` is the same thing as
plain steps.

Pick rules you would actually block a deploy on. `crash` and
`recursion_limit` are safe defaults. `finding:injected_instruction_followed`
is the prompt-injection canary — deterministic, no judge. `diverge` on an
LLM graph with `--divergence exact` will fail every run; use the `tiered`
cascade (any preset above quick) or a `max_diverge_rate`.

## Reports CI already understands

| Flag | Writes | Shows up as |
|---|---|---|
| `--junit drift.xml` | JUnit XML: a test case for the baseline and each perturbation; a case fails only when a `--fail-on` rule named it | the test-report pane of GitHub (via a JUnit reporter action), GitLab, Jenkins, CircleCI, Azure |
| `--summary drift.md` | Markdown: verdict, violated rules, baseline findings, every perturbation | anywhere you post Markdown (PR comment bots) |
| *(automatic)* | the same Markdown appended to `$GITHUB_STEP_SUMMARY` | the job's summary page in GitHub Actions; `--no-step-summary` turns it off |

All three come from the result JSON alone, so `drift gate results.json
--junit … --summary …` renders an artifact produced on another machine.

## Fail on what got worse: `--against`

A PR should fail on what it broke, not on what `main` already had. Save
main's result as an artifact or in the repo, then:

```bash
drift run my_app.graph --seed 7 --json pr.json \
  --against main-drift.json --fail-on new_finding,new_crash,new_event_crash,regression
```

| Rule | Fails when |
|---|---|
| `new_finding` | a finding type fires now that did not in the previous result |
| `new_crash` | a perturbation crashes now that ran clean before (matched by event name, so keep `--seed` fixed), or the baseline newly crashes |
| `new_event_crash` | a perturbation that did not exist in the previous result crashes — a fault layer you just turned on, a new tool, a changed schema. `new_crash` cannot judge these (nothing to match), and says so in a note |
| `regression` | crash rate up >5 points, divergence rate up >10 points, mean runtime up >50%, new finding types, or more intermittent outcomes — the same thresholds `drift history` uses |

With no previous result (a missing `--against` file, or none given) these
three rules are skipped with a note rather than failed: the first run on a
branch has nothing to regress against. An unreadable file is an error.
`POST /api/gate` takes the previous result as `previous`.

## Regression trend

Add `--track-history` to append a snapshot under `results/_history/`
(or `--history-dir`), then `drift history <module>.build` prints the
trend and flags crash-rate regressions between runs.
