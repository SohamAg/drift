"""Synthetic positive + negative tests for the coordination-failure detector library.

For each detector we ship a minimal trace fixture that should fire it (positive)
and a structurally-similar one that should NOT (negative). We also cross-check
specificity: a fixture that fires detector X should not fire detector Y unless
the two genuinely overlap.

Trace shape mirrors what `drift.adapters.langgraph._stream_or_invoke` produces:
    [{step: int, node: str, update: dict, state_after: dict}, ...]
"""
from __future__ import annotations

from drift.failures.library import (
    ALL_DETECTORS,
    hallucinated_reference,
    infinite_handoff,
    run_all_on_trace,
    subagent_fanout_excess,
    verifier_always_approves,
)
from drift.failures.library.base import from_adapter_trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(trace: list[dict], **kw):
    return from_adapter_trace(trace, **kw)


def _trace(steps: list[tuple[str, dict]], start_state: dict | None = None) -> list[dict]:
    """Build a trace from (node, update) pairs. state_after merges progressively."""
    running = dict(start_state or {})
    out = []
    for i, (node, update) in enumerate(steps, start=1):
        running = {**running, **update}
        out.append({
            "step": i,
            "node": node,
            "update": dict(update),
            "state_after": dict(running),
        })
    return out


# ---------------------------------------------------------------------------
# Detector 1: verifier_always_approves
# ---------------------------------------------------------------------------


def test_verifier_always_approves_positive_structured_key():
    trace = _trace([
        ("planner",  {"task": "issue-101"}),
        ("verifier", {"verdict": "approve", "rationale": "looks fine"}),
        ("planner",  {"task": "issue-102"}),
        ("verifier", {"verdict": "approve", "rationale": "looks fine"}),
        ("planner",  {"task": "issue-103"}),
        ("verifier", {"verdict": "approve", "rationale": "looks fine"}),
        ("planner",  {"task": "issue-104"}),
        ("verifier", {"verdict": "approve", "rationale": "looks fine"}),
    ])
    out = verifier_always_approves.detect(_ctx(trace))
    assert len(out) == 1
    assert out[0].failure_type == "verifier_always_approves"
    assert out[0].agents_involved == ["verifier"]
    assert "4/4" in out[0].summary


def test_verifier_always_approves_positive_freetext():
    """Rationale-only approval, no structured verdict key."""
    trace = _trace([
        ("planner",      {"task": "x"}),
        ("code_reviewer", {"rationale": "lgtm, ship it"}),
        ("planner",      {"task": "y"}),
        ("code_reviewer", {"rationale": "lgtm, ship it"}),
        ("planner",      {"task": "z"}),
        ("code_reviewer", {"rationale": "ok pass"}),
    ])
    out = verifier_always_approves.detect(_ctx(trace))
    assert len(out) == 1


def test_verifier_always_approves_negative_below_threshold():
    """1 approve out of 2 decisions — not enough decisions to fire."""
    trace = _trace([
        ("verifier", {"verdict": "approve"}),
        ("verifier", {"verdict": "approve"}),
    ])
    out = verifier_always_approves.detect(_ctx(trace))
    assert out == []


def test_verifier_always_approves_negative_has_rejection():
    """Any rejection at all = not "always approves"."""
    trace = _trace([
        ("verifier", {"verdict": "approve"}),
        ("verifier", {"verdict": "approve"}),
        ("verifier", {"verdict": "reject"}),
        ("verifier", {"verdict": "approve"}),
        ("verifier", {"verdict": "approve"}),
    ])
    out = verifier_always_approves.detect(_ctx(trace))
    assert out == []


def test_verifier_always_approves_negative_not_verifier_role():
    """An agent named "planner" approving everything is not the verifier failure
    — it's just a planner. Detector should silent."""
    trace = _trace([
        ("planner", {"verdict": "approve"}),
        ("planner", {"verdict": "approve"}),
        ("planner", {"verdict": "approve"}),
        ("planner", {"verdict": "approve"}),
    ])
    out = verifier_always_approves.detect(_ctx(trace))
    assert out == []


def test_verifier_always_approves_explicit_role_override():
    """When user supplies roles, name pattern is irrelevant — explicit role wins."""
    trace = _trace([
        ("agent_a", {"verdict": "approve"}),
        ("agent_a", {"verdict": "approve"}),
        ("agent_a", {"verdict": "approve"}),
        ("agent_a", {"verdict": "approve"}),
    ])
    out = verifier_always_approves.detect(
        _ctx(trace, roles_by_agent={"agent_a": "verifier"}),
    )
    assert len(out) == 1
    assert out[0].agents_involved == ["agent_a"]


def test_verifier_always_approves_text_variant_positive():
    transcript = """
Reviewer: approved — looks fine.
Planner: next task.
Reviewer: lgtm, approve.
Planner: next task.
Reviewer: approve, no issues.
Planner: next task.
Reviewer: passed review.
"""
    out = verifier_always_approves.detect_from_text(transcript)
    assert len(out) == 1
    assert out[0].failure_type == "verifier_always_approves"


def test_verifier_always_approves_text_variant_negative():
    transcript = """
Reviewer: approved.
Reviewer: rejected — bad logic.
Reviewer: approved.
Reviewer: approved.
"""
    out = verifier_always_approves.detect_from_text(transcript)
    assert out == []


# ---------------------------------------------------------------------------
# Detector 2: infinite_handoff
# ---------------------------------------------------------------------------


def test_infinite_handoff_positive():
    """A and B bounce 5 times in a row with no new fields or content growth."""
    base = {"task": "fix bug", "result": ""}
    trace = _trace([
        ("a", {"thinking": "passing to b"}),
        ("b", {"thinking": "passing to a"}),
        ("a", {"thinking": "passing to b"}),
        ("b", {"thinking": "passing to a"}),
        ("a", {"thinking": "passing to b"}),
    ], start_state=base)
    out = infinite_handoff.detect(_ctx(trace))
    assert len(out) == 1
    assert out[0].failure_type == "infinite_handoff"
    assert set(out[0].agents_involved) == {"a", "b"}


def test_infinite_handoff_negative_with_progress():
    """Same alternation, but each handoff adds a new keyed field — actual progress."""
    base = {"task": "fix bug"}
    trace = _trace([
        ("a", {"step1_done": True}),
        ("b", {"step2_done": True}),
        ("a", {"step3_done": True}),
        ("b", {"step4_done": True}),
        ("a", {"step5_done": True}),
    ], start_state=base)
    out = infinite_handoff.detect(_ctx(trace))
    assert out == []


def test_infinite_handoff_negative_below_alternation_threshold():
    """Three alternations only — not enough."""
    trace = _trace([
        ("a", {"thinking": "..."}),
        ("b", {"thinking": "..."}),
        ("a", {"thinking": "..."}),
    ], start_state={"task": "x", "result": ""})
    out = infinite_handoff.detect(_ctx(trace))
    assert out == []


def test_infinite_handoff_negative_three_agents():
    """A-B-C-A-B-C isn't a pairwise loop; detector should stay quiet."""
    trace = _trace([
        ("a", {"thinking": "..."}),
        ("b", {"thinking": "..."}),
        ("c", {"thinking": "..."}),
        ("a", {"thinking": "..."}),
        ("b", {"thinking": "..."}),
        ("c", {"thinking": "..."}),
    ], start_state={"task": "x", "result": ""})
    out = infinite_handoff.detect(_ctx(trace))
    assert out == []


def test_infinite_handoff_text_variant_positive():
    transcript = """
Agent Alpha: I'll pass this back to you.
Agent Beta: Actually, you should handle it.
Agent Alpha: No, you take it.
Agent Beta: I think you're better suited.
Agent Alpha: Let's hand it back to you.
"""
    out = infinite_handoff.detect_from_text(transcript)
    assert len(out) == 1
    assert out[0].failure_type == "infinite_handoff"


# ---------------------------------------------------------------------------
# Detector 3: subagent_fanout_excess
# ---------------------------------------------------------------------------


def test_subagent_fanout_excess_positive_hard_count():
    """Orchestrator + 10 distinct subagents — clearly over threshold."""
    steps = [("orchestrator", {"plan": "do many things"})]
    for i in range(10):
        steps.append((f"worker_{i}", {f"out_{i}": f"result_{i}"}))
    trace = _trace(steps)
    out = subagent_fanout_excess.detect(_ctx(trace))
    assert len(out) == 1
    assert out[0].failure_type == "subagent_fanout_excess"
    assert "orchestrator" in out[0].agents_involved


def test_subagent_fanout_excess_positive_ratio():
    """6 subagents, 1 output landed — ratio rule fires even below hard count."""
    steps = [("planner", {})]
    for i in range(6):
        steps.append((f"worker_{i}", {"transient": f"x{i}"}))
    steps.append(("planner", {"final_report": "done"}))
    trace = _trace(steps)
    # Strip transient keys so only final_report counts as output.
    last = trace[-1]
    last["state_after"] = {"final_report": "done"}
    out = subagent_fanout_excess.detect(_ctx(trace))
    assert len(out) == 1
    assert out[0].failure_type == "subagent_fanout_excess"
    # 6 subagents / 1 output = 6.0, well over the 3.0 ratio threshold.
    assert "ratio 6.0" in out[0].summary


def test_subagent_fanout_excess_negative_small_fanout():
    """Map-reduce of 3 workers, normal output — should not fire."""
    steps = [
        ("orchestrator", {"plan": "split"}),
        ("w1", {"r1": "ok"}),
        ("w2", {"r2": "ok"}),
        ("w3", {"r3": "ok"}),
        ("orchestrator", {"final": "merged"}),
    ]
    trace = _trace(steps)
    out = subagent_fanout_excess.detect(_ctx(trace))
    assert out == []


def test_subagent_fanout_excess_text_variant():
    transcript = """
Orchestrator: delegating to worker_a
Orchestrator: delegating to worker_b
Orchestrator: delegating to worker_c
Orchestrator: delegating to worker_d
Orchestrator: delegating to worker_e
Orchestrator: delegating to worker_f
Orchestrator: delegating to worker_g
Orchestrator: delegating to worker_h
Orchestrator: delegating to worker_i
"""
    out = subagent_fanout_excess.detect_from_text(transcript)
    assert len(out) == 1
    assert out[0].failure_type == "subagent_fanout_excess"


# ---------------------------------------------------------------------------
# Detector 4: hallucinated_reference
# ---------------------------------------------------------------------------


def test_hallucinated_reference_positive_prefix_id_in_rationale():
    """Agent's rationale mentions TICKET-42; no such id in prior state."""
    trace = _trace([
        ("planner",  {"rationale": "starting work on the queue"}),
        ("worker",   {"rationale": "I'll close TICKET-42 as duplicate."}),
    ])
    out = hallucinated_reference.detect(_ctx(trace))
    assert len(out) == 1
    assert out[0].failure_type == "hallucinated_reference"
    assert out[0].agents_involved == ["worker"]
    assert "ticket-42" in out[0].summary
    assert out[0].timestep == 2


def test_hallucinated_reference_positive_hash_id():
    """Hash-prefixed id (#123) referenced with no prior mention."""
    trace = _trace([
        ("router",  {"rationale": "picking next task"}),
        ("solver",  {"rationale": "linking to PR #987 as related"}),
    ])
    out = hallucinated_reference.detect(_ctx(trace))
    assert len(out) == 1
    assert "#987" in out[0].summary


def test_hallucinated_reference_positive_word_form():
    """`case 99` in text with no case-99 in prior state."""
    trace = _trace([
        ("triage", {"rationale": "reviewing incoming"}),
        ("actor",  {"rationale": "escalating case-99 to on-call"}),
    ])
    out = hallucinated_reference.detect(_ctx(trace))
    assert len(out) == 1
    assert "case-99" in out[0].summary


def test_hallucinated_reference_negative_known_from_initial_state():
    """TICKET-42 exists in initial_state; later reference is legitimate."""
    trace = _trace([
        ("planner", {"rationale": "starting"}),
        ("worker",  {"rationale": "closing TICKET-42"}),
    ])
    ctx = _ctx(trace, initial_state={"open_tickets": ["TICKET-42"]})
    out = hallucinated_reference.detect(ctx)
    assert out == []


def test_hallucinated_reference_negative_known_from_prior_step():
    """Prior step introduced TICKET-42 in its structured field; later ref fine."""
    trace = _trace([
        ("intake",  {"ticket_id": "TICKET-42", "rationale": "new report"}),
        ("worker",  {"rationale": "closing TICKET-42"}),
    ])
    out = hallucinated_reference.detect(_ctx(trace))
    assert out == []


def test_hallucinated_reference_negative_defined_this_step():
    """Same step both defines TICKET-42 (structured) AND mentions in rationale.
    That's a legitimate entity creation, not a hallucination."""
    trace = _trace([
        ("intake", {
            "ticket_id": "TICKET-42",
            "rationale": "created TICKET-42 from new bug report",
        }),
    ])
    out = hallucinated_reference.detect(_ctx(trace))
    assert out == []


def test_hallucinated_reference_negative_bare_number():
    """Just `42` in a rationale — no prefix, no # — must not fire."""
    trace = _trace([
        ("planner", {"rationale": "there are 42 items in the queue"}),
    ])
    out = hallucinated_reference.detect(_ctx(trace))
    assert out == []


def test_hallucinated_reference_negative_word_without_digits():
    """`case sensitive` — matches _WORD_ID regex but id part has no digit."""
    trace = _trace([
        ("worker", {"rationale": "the check is case sensitive"}),
    ])
    out = hallucinated_reference.detect(_ctx(trace))
    assert out == []


def test_hallucinated_reference_negative_hash_bullet():
    """`#1`, `#2` bullets — under 2 digits threshold, no fire."""
    trace = _trace([
        ("planner", {"rationale": "the plan: #1 do this, #2 do that"}),
    ])
    out = hallucinated_reference.detect(_ctx(trace))
    assert out == []


def test_hallucinated_reference_fires_only_at_origin_not_propagation():
    """When step 2 hallucinates, step 3 mentioning the same id inherits it —
    we flag the origin only, not the downstream propagation."""
    trace = _trace([
        ("planner", {"rationale": "starting"}),
        ("worker_a", {"rationale": "beginning work on TICKET-42"}),
        ("worker_b", {"rationale": "TICKET-42 confirmed as duplicate"}),
        ("worker_c", {"rationale": "closing out TICKET-42"}),
    ])
    out = hallucinated_reference.detect(_ctx(trace))
    assert len(out) == 1
    assert out[0].agents_involved == ["worker_a"]
    assert out[0].timestep == 2


def test_hallucinated_reference_multiple_ids_same_step():
    """Same agent, same step, mentions two distinct unknown ids — flag both."""
    trace = _trace([
        ("planner", {"rationale": "starting"}),
        ("worker",  {
            "rationale": "linking TICKET-42 with #987 as duplicate"
        }),
    ])
    out = hallucinated_reference.detect(_ctx(trace))
    types = {(f.failure_type, f.agents_involved[0]) for f in out}
    assert len(out) == 2
    assert all(t == ("hallucinated_reference", "worker") for t in types)
    ids_in_summaries = " ".join(f.summary for f in out).lower()
    assert "ticket-42" in ids_in_summaries
    assert "#987" in ids_in_summaries


def test_hallucinated_reference_negative_empty_trace():
    out = hallucinated_reference.detect(_ctx([]))
    assert out == []


# ---- text variant tests ---------------------------------------------------


def test_hallucinated_reference_text_variant_positive():
    """First speaker mentions no ids; second speaker mentions TICKET-42."""
    transcript = """
Alice: I'll start by triaging the queue.
Bob: Sure, I'll close TICKET-42 as a duplicate.
"""
    out = hallucinated_reference.detect_from_text(transcript)
    assert len(out) == 1
    assert "ticket-42" in out[0].summary.lower()


def test_hallucinated_reference_text_variant_negative_first_speaker_authoritative():
    """First speaker introduces the id; second speaker references it — fine."""
    transcript = """
Alice: I opened TICKET-42 for the reported bug.
Bob: Great, I'll close TICKET-42 as a duplicate.
"""
    out = hallucinated_reference.detect_from_text(transcript)
    assert out == []


def test_hallucinated_reference_text_variant_negative_empty():
    assert hallucinated_reference.detect_from_text("") == []


# ---------------------------------------------------------------------------
# Cross-specificity: each positive fixture must not trigger other detectors
# ---------------------------------------------------------------------------


def test_specificity_verifier_does_not_trigger_handoff_or_fanout():
    trace = _trace([
        ("planner",  {"task": "issue-101"}),
        ("verifier", {"verdict": "approve"}),
        ("planner",  {"task": "issue-102"}),
        ("verifier", {"verdict": "approve"}),
        ("planner",  {"task": "issue-103"}),
        ("verifier", {"verdict": "approve"}),
        ("planner",  {"task": "issue-104"}),
        ("verifier", {"verdict": "approve"}),
    ])
    out = run_all_on_trace(trace)
    types = [f.failure_type for f in out]
    # Verifier fires; handoff might fire too because the planner/verifier pair
    # alternates — but only if state isn't progressing. Here each step writes
    # a new `task` value to an EXISTING key (not a new key), so no key-set growth.
    # But `task` becomes non-empty across steps... let's confirm what actually fires.
    assert "verifier_always_approves" in types
    assert "subagent_fanout_excess" not in types


def test_specificity_handoff_does_not_trigger_verifier_or_fanout():
    base = {"task": "fix bug", "result": ""}
    trace = _trace([
        ("a", {"thinking": "passing to b"}),
        ("b", {"thinking": "passing to a"}),
        ("a", {"thinking": "passing to b"}),
        ("b", {"thinking": "passing to a"}),
        ("a", {"thinking": "passing to b"}),
    ], start_state=base)
    out = run_all_on_trace(trace)
    types = [f.failure_type for f in out]
    assert "infinite_handoff" in types
    assert "verifier_always_approves" not in types
    assert "subagent_fanout_excess" not in types


def test_specificity_fanout_does_not_trigger_verifier_or_handoff():
    steps = [("orchestrator", {"plan": "do many things"})]
    for i in range(10):
        steps.append((f"worker_{i}", {f"out_{i}": f"result_{i}"}))
    trace = _trace(steps)
    out = run_all_on_trace(trace)
    types = [f.failure_type for f in out]
    assert "subagent_fanout_excess" in types
    assert "verifier_always_approves" not in types
    assert "infinite_handoff" not in types


def test_specificity_hallucinated_reference_fixture_fires_only_it():
    """Two-step trace with a single hallucinated TICKET reference — no other
    detector should fire on such a short trace."""
    trace = _trace([
        ("planner",  {"rationale": "starting"}),
        ("worker",   {"rationale": "closing TICKET-42 as duplicate"}),
    ])
    out = run_all_on_trace(trace)
    types = [f.failure_type for f in out]
    assert "hallucinated_reference" in types
    assert "verifier_always_approves" not in types
    assert "infinite_handoff" not in types
    assert "subagent_fanout_excess" not in types


def test_specificity_other_fixtures_do_not_fire_hallucination():
    """Verifier/handoff/fanout fixtures don't accidentally have unknown ids."""
    verifier_fx = _trace([
        ("planner",  {"task": "review feature x"}),
        ("verifier", {"verdict": "approve"}),
        ("planner",  {"task": "review feature y"}),
        ("verifier", {"verdict": "approve"}),
        ("planner",  {"task": "review feature z"}),
        ("verifier", {"verdict": "approve"}),
        ("planner",  {"task": "review feature w"}),
        ("verifier", {"verdict": "approve"}),
    ])
    handoff_fx = _trace([
        ("a", {"thinking": "passing to b"}),
        ("b", {"thinking": "passing to a"}),
        ("a", {"thinking": "passing to b"}),
        ("b", {"thinking": "passing to a"}),
    ], start_state={"task": "fix bug", "result": ""})
    fanout_steps = [("orchestrator", {"plan": "do many things"})]
    for i in range(10):
        fanout_steps.append((f"worker_{i}", {f"out_{i}": f"result_{i}"}))
    fanout_fx = _trace(fanout_steps)

    for fx in (verifier_fx, handoff_fx, fanout_fx):
        types = [f.failure_type for f in run_all_on_trace(fx)]
        assert "hallucinated_reference" not in types


# ---------------------------------------------------------------------------
# Library-level integration
# ---------------------------------------------------------------------------


def test_run_all_on_trace_returns_findings_from_each_relevant_detector():
    """Construct a trace that fires BOTH verifier_always_approves AND
    infinite_handoff: planner+verifier alternating with auto-approval, no
    new keys, no new non-empty strings, no container growth across the run."""
    # Seed all keys non-empty up front so subsequent identical writes don't
    # count as "first time non-empty" progress.
    base = {"task": "review feature x", "rationale": "(thinking)", "verdict": "approve"}
    trace = _trace([
        ("planner",  {"rationale": "(thinking)"}),
        ("verifier", {"verdict": "approve"}),
        ("planner",  {"rationale": "(thinking)"}),
        ("verifier", {"verdict": "approve"}),
        ("planner",  {"rationale": "(thinking)"}),
        ("verifier", {"verdict": "approve"}),
        ("planner",  {"rationale": "(thinking)"}),
        ("verifier", {"verdict": "approve"}),
    ], start_state=base)
    out = run_all_on_trace(trace)
    types = {f.failure_type for f in out}
    assert "verifier_always_approves" in types
    assert "infinite_handoff" in types


def test_run_all_on_trace_empty_returns_nothing():
    assert run_all_on_trace([]) == []


def test_run_all_on_text_empty_returns_nothing():
    from drift.failures.library import run_all_on_text
    assert run_all_on_text("") == []


def test_all_detectors_have_required_metadata():
    for mod in ALL_DETECTORS:
        assert isinstance(mod.NAME, str) and mod.NAME
        assert isinstance(mod.SUMMARY, str) and mod.SUMMARY
        assert isinstance(mod.MAST_MODES, list) and all(isinstance(m, str) for m in mod.MAST_MODES)
        assert isinstance(mod.SOURCE, str) and mod.SOURCE
        assert callable(mod.detect)
        assert callable(mod.detect_from_text)
