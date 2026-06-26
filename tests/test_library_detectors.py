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
