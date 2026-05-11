from drift.world import Case, World, WorldState


def test_world_apply_records_history():
    w = World(initial=WorldState())
    w.begin_step(1)
    w.add_case(Case(case_id="c1", customer_id="u1", issue="x", opened_at_step=1), source="event", source_id="e1")
    w.adjust_sentiment(-0.1, source="event", source_id="e1")
    w.commit_step()
    assert len(w.history) == 1
    snap = w.history.latest()
    assert snap is not None
    assert "c1" in snap.open_cases
    assert snap.customer_sentiment < 0.7


def test_sentiment_clamped():
    w = World(initial=WorldState(customer_sentiment=0.95))
    w.begin_step(1)
    w.adjust_sentiment(+0.5, source="event", source_id="e1")
    assert w.state.customer_sentiment == 1.0
    w.adjust_sentiment(-2.0, source="event", source_id="e1")
    assert w.state.customer_sentiment == 0.0


def test_remove_case_clears_queue_entries():
    w = World(initial=WorldState())
    w.begin_step(1)
    w.add_case(Case(case_id="c1", customer_id="u1", issue="x"), source="event", source_id="e1")
    w.enqueue_escalation("c1", source="action", source_id="a1")
    w.enqueue_escalation("c1", source="action", source_id="a2")
    w.remove_case("c1", source="action", source_id="a3")
    assert "c1" not in w.state.open_cases
    assert all(r.case_id != "c1" for r in w.state.escalation_queue)
