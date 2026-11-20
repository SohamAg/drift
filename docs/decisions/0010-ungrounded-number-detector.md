# ADR-0010: Ship `ungrounded_number`, on by default, with three guards and tool-argument tracking

| | |
|---|---|
| Status | Accepted. Supersedes ADR-0008 |
| Decided | 2026-11-21 01:52 IST — `dfe91fa` (detector); live check 01:54 IST — `5c27d91` |
| Recorded | 2026-11-21, the same day |
| Owner | Soham Agarwal |

## Context
ADR-0008 deferred a detector for precise numbers that no tool supplied.
The bare heuristic caught real fabrications, but it also fired on correct
sums and on the speed of light. ADR-0008 named three conditions for
revisiting:
1. a guard for computed values;
2. a guard for well-known constants;
3. re-measurement on the saved runs, with every fire checked by hand.

Building the computed-value guard exposed a fourth problem. A fabricated
input can be recovered backwards from its own tool result. On a saved
swarm baseline, the search returned nothing. The math agent called
`multiply(13929286, 0.5)` and got 6,964,643. The answer's "13,929,286"
then counted as computed, because 6,964,643 / 0.5 = 13,929,286. The guard
excused exactly the fabrication the detector exists to catch.

## Decision
- **Scope.** A number counts only if it has at least 4 significant
  digits and is not a bare year or part of an identifier, date or
  version. A wrong but round answer ("approximately 25 million") is out of
  scope. The detector catches invented precision, not every wrong number.
- **Grounding.** A number is grounded if it matches a number in a human,
  system or tool message, or in the initial state, at the precision the
  answer states it.
- **Guard 1, computations.** The sum, difference, product, ratio or
  percentage of any two grounded numbers is grounded. So are the sum of
  any three and the sum of all of them.
- **Guard 2, constants.** A fixed list of well-known constants, compared
  scale-free.
- **Tool-argument tracking (new).** Numbers the model writes into a tool
  call's arguments are checked against what the run had supplied *before*
  that call. A call carrying an ungrounded number fires the detector, and
  its result no longer counts as grounding.
- **Applicability.** The detector only runs when the run had at least one
  successful tool result.
- **On by default**, like every other library detector. It is
  deterministic and costs no LLM calls.

## Evidence (condition 3)
Measured over the 110 saved runs under `results/` that have tool results.
Every fire was checked by reading the conversation:

| | Fires | Real fabrications |
|---|---|---|
| ADR-0008 heuristic (2026-11-19) | 9 | 3 |
| This detector | 3 | 3 |

- **The constants guard** cleared 19 speed-of-light mentions.
- **Tool-argument tracking** recovered the swarm case that the
  computation guard had wrongly excused.
- **Live re-run, 2026-11-21**, gpt-4o-mini, supervisor with research
  misrouted to math: it fired 2 of 2 times and stayed silent on the
  baseline.

## Alternatives considered
- **An LLM judge asked "is every figure sourced?"** Rejected as the
  default. ADR-0001 keeps deterministic detectors as the primary signal,
  and a judge cannot be gated on reliably.
- **Ship it opt-in.** Rejected. 3 fires out of 110 runs, all real, is the
  precision bar the other default detectors meet.
- **Treat every number in a tool result as grounded, regardless of how
  the call was made.** Rejected. The swarm case shows it hides the
  fabrication.
- **Check every AI message, not just the final answer and tool
  arguments.** Not done. Intermediate reasoning often restates numbers
  loosely. The final answer is what the user holds, and tool arguments are
  where fabricated figures enter.

## Consequences
- The library has 22 detectors.
- Precision has been measured on 110 runs from a handful of systems. That
  is not a benchmark. The constants list will need additions as new
  domains appear. A guard miss means a false positive, not a missed
  failure.
- Out of scope: wrong but low-precision answers, fabricated non-numeric
  facts (names, dates, quotes), and numbers in agents with no tools.
