"""One synthetic fixture per detector. Each test ensures its own detector
fires AND the others stay silent on that fixture, so we know detectors are
discriminating, not blanket-firing."""
from drift.agents.base import Action
from drift.failures.base import DetectorContext
from drift.failures.detectors import (
    detect_contradictory_refunds,
    detect_escalation_loop,
    detect_hallucinated_reference,
    detect_policy_inconsistency,
    detect_queue_explosion,
    detect_sentiment_collapse,
    detect_stale_snapshot_reference,
)
from drift.world import Case, CaseRef, World, WorldState


def _ctx_for(world: World, actions: list[Action], t: int) -> DetectorContext:
    return DetectorContext(
        timestep=t,
        history=world.history,
        actions=actions,
        events=[],
        already_reported=set(),
    )


def _step(world: World, t: int) -> None:
    world.begin_step(t)
    world.commit_step()


def test_contradictory_refunds():
    w = World(initial=WorldState())
    w.begin_step(1)
    w.add_case(Case(case_id="c1", customer_id="u1", issue="x"), source="event", source_id="e1")
    w.commit_step()
    actions = [
        Action(timestep=1, agent_name="refund_a", kind="refund_approve", target_case_id="c1"),
        Action(timestep=1, agent_name="refund_b", kind="refund_deny", target_case_id="c1"),
    ]
    out = detect_contradictory_refunds(_ctx_for(w, actions, 1))
    assert len(out) == 1
    assert out[0].failure_type == "contradictory_refund"


def test_escalation_loop():
    w = World(initial=WorldState())
    w.begin_step(1)
    w.add_case(Case(case_id="c1", customer_id="u1", issue="x"), source="event", source_id="e1")
    for _ in range(4):
        w.enqueue_escalation("c1", source="action", source_id="a")
    w.commit_step()
    out = detect_escalation_loop(_ctx_for(w, [], 1))
    assert len(out) == 1
    assert out[0].failure_type == "escalation_loop"


def test_policy_inconsistency():
    w = World(initial=WorldState(refund_policy_version=3))
    w.begin_step(1)
    w.commit_step()
    actions = [
        Action(timestep=1, agent_name="refund", kind="refund_approve",
               target_case_id="c1", referenced_policy_version=2),
    ]
    out = detect_policy_inconsistency(_ctx_for(w, actions, 1))
    assert len(out) == 1
    assert out[0].failure_type == "policy_inconsistency"


def test_sentiment_collapse():
    w = World(initial=WorldState(customer_sentiment=0.10))
    for t in range(1, 7):
        w.begin_step(t)
        w.commit_step()
    out = detect_sentiment_collapse(_ctx_for(w, [], 6))
    assert len(out) == 1
    assert out[0].failure_type == "sentiment_collapse"


def test_hallucinated_reference_fabricated_id():
    """True fabrication: the case_id never existed in any snapshot."""
    w = World(initial=WorldState())
    _step(w, 1)
    actions = [
        Action(timestep=1, agent_name="support", kind="escalate", target_case_id="ghost_99"),
    ]
    out = detect_hallucinated_reference(_ctx_for(w, actions, 1))
    assert len(out) == 1
    assert out[0].failure_type == "hallucinated_reference"


def test_stale_snapshot_reference_distinct_from_hallucination():
    """A case that WAS open and got resolved should fire stale_snapshot, not hallucination."""
    w = World(initial=WorldState())
    # t=1: case c1 exists.
    w.begin_step(1)
    w.add_case(Case(case_id="c1", customer_id="u1", issue="x"), source="event", source_id="e1")
    w.commit_step()
    # t=2: case c1 is removed mid-step (e.g. resolved by an earlier-ordered agent).
    w.begin_step(2)
    w.remove_case("c1", source="action", source_id="a1")
    w.commit_step()
    actions = [
        Action(timestep=2, agent_name="refund", kind="refund_approve", target_case_id="c1"),
    ]
    ctx = _ctx_for(w, actions, 2)
    halluc = detect_hallucinated_reference(ctx)
    stale = detect_stale_snapshot_reference(ctx)
    assert halluc == [], "c1 was real once; should not be flagged as fabricated"
    assert len(stale) == 1
    assert stale[0].failure_type == "stale_snapshot_reference"


def test_queue_explosion():
    w = World(initial=WorldState())
    w.begin_step(1)
    w.add_case(Case(case_id="c1", customer_id="u", issue="x"), source="event", source_id="e1")
    w.commit_step()
    for t in range(2, 7):
        w.begin_step(t)
        w.state.escalation_queue.append(CaseRef(case_id="c1", enqueued_at_step=t))
        w.commit_step()
    out = detect_queue_explosion(_ctx_for(w, [], 6))
    assert len(out) == 1
    assert out[0].failure_type == "queue_explosion"


def test_detectors_dont_fire_on_clean_state():
    w = World(initial=WorldState())
    for t in range(1, 7):
        w.begin_step(t)
        w.commit_step()
    ctx = _ctx_for(w, [], 6)
    assert detect_contradictory_refunds(ctx) == []
    assert detect_escalation_loop(ctx) == []
    assert detect_policy_inconsistency(ctx) == []
    assert detect_sentiment_collapse(ctx) == []
    assert detect_hallucinated_reference(ctx) == []
    assert detect_stale_snapshot_reference(ctx) == []
    assert detect_queue_explosion(ctx) == []
