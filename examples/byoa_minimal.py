"""Minimal BYOA example for drift.

Shows the smallest possible "bring your own agents" path. The user:
  1. Optionally subclasses drift.WorldState to add domain-specific fields.
  2. Writes async agent functions and decorates each with @drift.agent.
  3. Pre-populates the initial state with whatever work items they care about.
  4. Calls drift.run(...) — drift drives the agents through N timesteps and
     runs its coordination-failure detectors over the resulting action log.

The agents below are deterministic for demo-clarity. Real users would put
their own LLM calls, tool invocations, RAG lookups, etc. inside the decorated
function body — drift doesn't care how the decision gets made; it just calls
the function and reads back the Action.

Run from project root with:

    PYTHONPATH=src python examples/byoa_minimal.py
"""
from __future__ import annotations

import drift
from drift.failures.detectors import GENERAL_DETECTORS
from drift.topologies.code_review.detectors import CODE_REVIEW_DETECTORS
from drift.world import Case


# --- 1. Domain state ----------------------------------------------------
#
# drift.WorldState has `open_cases` and a few other generic fields. Subclass
# it to add any additional state your domain needs (open_prs, security_status,
# severity_levels, etc.). The detectors read from `open_cases` for the general
# grounding-failure detectors, so it's worth keeping your work-items in there
# unless you also bring custom detectors.
class CodeReviewState(drift.WorldState):
    repository: str = "demo/repo"


# --- 2. Agents ----------------------------------------------------------
#
# Each @drift.agent function takes (state, memory) and returns a drift.Action.
# Whatever you do inside the body is yours — call an LLM, dispatch a tool,
# look up RAG context, etc. drift only requires the structured Action back so
# detectors can read it.

@drift.agent(role="reviewer", name="reviewer_a")
async def reviewer_a(state, memory):
    """Approves the first open PR each step. No-op when nothing's open."""
    if state.open_cases:
        target = sorted(state.open_cases)[0]
        return drift.Action(
            kind="approve_review",
            target_case_id=target,
            rationale=f"reviewer_a thinks {target} looks fine",
        )
    return drift.Action(kind="no_op")


@drift.agent(role="reviewer", name="reviewer_b")
async def reviewer_b(state, memory):
    """Disagrees with reviewer_a (rejects the same PR each step).

    At t=4 also references a PR that doesn't exist in any snapshot — meant to
    trigger the `hallucinated_reference` detector.
    """
    if state.timestep == 4:
        return drift.Action(
            kind="reject_review",
            target_case_id="PR-PHANTOM",
            rationale="reviewer_b confidently rejects a PR that does not exist",
        )
    if state.open_cases:
        target = sorted(state.open_cases)[0]
        return drift.Action(
            kind="reject_review",
            target_case_id=target,
            rationale=f"reviewer_b objects to {target}",
        )
    return drift.Action(kind="no_op")


@drift.agent(role="merger")
async def merger(state, memory):
    """A merger that does nothing in this demo. Real merger logic would gate
    on having an approve from at least one reviewer and a clear security signal."""
    return drift.Action(kind="no_op")


# --- 3. Run -------------------------------------------------------------

def main() -> None:
    initial = CodeReviewState(
        open_cases={
            "PR-1": Case(
                case_id="PR-1",
                customer_id="alice",
                issue="add dark mode toggle",
                opened_at_step=0,
            ),
        },
    )

    result = drift.run(
        agents=[reviewer_a, reviewer_b, merger],
        state=initial,
        steps=10,
        # Default detectors are the general/cross-topology ones; layer the
        # code-review-specific detectors on top so `contradictory_review`
        # also fires.
        detectors=list(GENERAL_DETECTORS) + list(CODE_REVIEW_DETECTORS),
    )

    # --- Report -------------------------------------------------------------
    print("=" * 64)
    print(f"  drift BYOA demo — {result.final_state.timestep} steps")
    print("=" * 64)
    print(f"  Actions:  {len(result.actions)}")
    print(f"  Events:   {len(result.events)}")
    print(f"  Failures: {len(result.failures)}")
    print()
    if result.failures:
        print("Detected coordination failures:")
        by_type: dict[str, list] = {}
        for f in result.failures:
            by_type.setdefault(f.failure_type, []).append(f)
        for ftype, items in sorted(by_type.items()):
            print(f"  [{ftype}]  x{len(items)}")
            for f in items[:3]:
                agents = ",".join(f.agents_involved) or "-"
                print(f"    t={f.timestep:>3}  agents={agents}  {f.summary}")
            if len(items) > 3:
                print(f"    ... +{len(items) - 3} more")
    else:
        print("(no failures detected)")


if __name__ == "__main__":
    main()
