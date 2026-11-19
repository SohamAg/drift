# ADR-0001: Deterministic detectors are the primary failure signal; the LLM judge is optional

| | |
|---|---|
| Status | Accepted |
| Decided | 2026-06-25 — `7504f49` (detector library introduced). Restated in the report's own wording on 2026-11-15 — `361c2f0` |
| Recorded | 2026-11-19 (after the fact) |
| Owner | Soham Agarwal |

## Context
drift's first failure signal was an LLM judge reading each trace against a
taxonomy of multi-agent failures. Measured against MAST, the Berkeley
human-labelled dataset of real multi-agent traces, the judge reached
P = 0.226, R = 0.125, F1 = 0.161 (`CASE_STUDY_MAST.md`). It missed seven in
eight real failures. Its output also changes from run to run. A CI gate
built on it would fail or pass the same code at random, and every check
would cost an API call.

## Decision
Failures are detected first by a library of deterministic detectors, each
a plain function over the trace (21 as of 2026-11-15). Every detector has
a written definition, the evidence it cites and the false-positive guards
it applies (`docs/detectors/`). The LLM judge stays, but it is optional:
off in the `quick` preset, never required for a gate rule except `judge`.

## Alternatives considered
- **Judge only, with better prompts.** Rejected. Prompting may raise F1,
  but the result stays non-deterministic and paid for per run.
- **Fine-tune a classifier.** Rejected for now. There is not enough
  labelled multi-agent data: MAST has 19 traces.

## Consequences
- The same trace gives the same findings every time, offline, for free.
  That is what makes `drift gate` usable in CI.
- Each detector only finds the one failure it was written for. Coverage
  grows by adding detectors, each validated on real runs before it ships
  (see ADR-0008 for one that was held back).
- Before 2026-11-15, the report's subtitle still said failures were
  "detected by a taxonomy judge". `361c2f0` corrected it to match this
  decision.
