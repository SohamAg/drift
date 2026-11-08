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
| `judge` | any LLM-judge finding |
| `max_crash_rate=0.2` | crashed ÷ perturbations above the fraction |
| `max_diverge_rate=0.5` | diverged ÷ perturbations above the fraction |

Rules are comma-separable and `--fail-on` is repeatable. Rates are over the
perturbations actually scheduled, so `--max-perturbations 4` is judged on
those four.

## Presets

Same four as the web UI; any knob overrides the preset.

| Preset | Chaos | Divergence | Rollouts | Judge | Cost |
|---|---|---|---|---|---|
| `quick` (default) | light, ≤4 | exact | 1 | off | free, seconds |
| `balanced` | aggressive, ≤6 | tiered, 3 baseline | 1 | openai | ~$0.01 |
| `thorough` | aggressive, ≤12 | tiered, 5 baseline | 2 | openai | ~$0.20 |
| `exhaustive` | every pattern | tiered, 5 baseline | 3 | openai | $1+ on wide schemas |

`--rollouts N` is the knob that turns "the injection was followed" into
"the injection was followed 1 of 3 times". One LLM run is a sample.

## GitHub Actions

`examples/ci/drift-gate.yml` is a complete workflow. The shape:

```yaml
- run: pip install -e ".[langgraph,openai]"
- run: drift run my_app.graph --preset balanced --json drift-results.json --fail-on crash,recursion_limit,finding:injected_instruction_followed
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
- uses: actions/upload-artifact@v4
  if: always()
  with: { name: drift-results, path: drift-results.json }
```

Pick rules you would actually block a deploy on. `crash` and
`recursion_limit` are safe defaults. `finding:injected_instruction_followed`
is the prompt-injection canary — deterministic, no judge. `diverge` on an
LLM graph with `--divergence exact` will fail every run; use the `tiered`
cascade (any preset above quick) or a `max_diverge_rate`.

## Regression trend

Add `--track-history` to append a snapshot under `results/_history/`
(or `--history-dir`), then `drift history <module>.build` prints the
trend and flags crash-rate regressions between runs.
