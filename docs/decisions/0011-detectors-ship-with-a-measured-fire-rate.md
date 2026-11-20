# ADR-0011: A new detector ships with its fire rate measured on the saved runs, every fire read by hand

| | |
|---|---|
| Status | Accepted |
| Decided | 2026-11-21, made a rule with ADR-0010. Practised since 2026-11-15: `empty_handoff` was measured on 92 saved runs before it shipped |
| Recorded | 2026-11-21, the same day |
| Owner | Soham Agarwal |

## Context
drift's detectors are meant to be gated on in CI (ADR-0001, ADR-0005).
A detector that fires on healthy runs teaches teams to ignore it, or to
turn it off. Unit tests show that a detector fires on the case it was
written for. They say nothing about what else it fires on.

`results/` holds saved baseline and perturbed runs from real third-party
systems (langgraph-supervisor, swarm, deepagents, react agents, an
adversarial system) and the live runs behind the docs. It is the closest
thing drift has to a corpus of healthy and broken behaviour.

Two detectors were built this way before the rule existed. `empty_handoff`
fired once in 92 runs, on the run it was built for. ADR-0008 was the
opposite case: the measurement showed a heuristic was not ready, and it
was not shipped.

## Decision
Before a library detector ships on by default:
1. **Run it over every saved run under `results/`** that has the inputs it
   reads, baselines and perturbations alike.
2. **Read every fire** and label it by hand: a real failure, or not.
3. **Put the numbers in the detector's docs page**: runs checked, fires,
   how many were real, and what the guards cleared.
4. **If most fires are not real failures, it does not ship by default.**
   It is deferred (as in ADR-0008) or made opt-in, with the measurement
   recorded.

## Alternatives considered
- **Unit tests only.** Rejected. They cannot show false positives on
  behaviour nobody thought to write a test for.
- **A fixed precision threshold, such as 90%.** Not adopted yet. The
  corpus is too small for a percentage to mean much, so reading every
  fire is more honest at this size.
- **The MAST human-labelled dataset.** Used for the judge (ADR-0001).
  It does not fit most message-aware detectors, because MAST traces are
  unstructured text with no tool-call structure.

## Consequences
- Each detector's docs page states its measured fire rate.
- The corpus is small and drawn from a handful of systems. The measurement
  is a floor, not proof of precision in the wild.
- `results/` is git-ignored, so the corpus lives on the owner's machine.
  The counts are reproducible from the saved JSON, but not by anyone else
  until a curated subset is committed. That is a follow-up.
  *(Note, 2026-11-21: done for `ungrounded_number` in `076fb2e`, a 10-run
  corpus in `tests/fixtures/`. Other detectors still to follow.)*
