"""perturbation_rollouts=N — each perturbed state is run N times and the
result reports k/N instead of pretending one LLM sample is a verdict."""

from __future__ import annotations

import itertools

import pytest

from drift.adapters.langgraph import (
    AdapterResult,
    PerturbationResult,
    drift_test,
    drift_test_async,
)

INIT = {"query": "what is 7 times 8", "reply": "", "attempts": 0, "approved": False}


class _CountingGraph:
    """Deterministic graph; records every invocation."""

    def __init__(self):
        self.calls = 0

    def invoke(self, state, *args, **kwargs):
        self.calls += 1
        out = dict(state)
        out["reply"] = "56"
        return out


class _FlakyGraph:
    """Alternates outcomes on every call so rollouts of the SAME perturbed
    state disagree — the situation N > 1 exists to expose.

    Cycle: unchanged, diverged, crashed, unchanged, …"""

    def __init__(self):
        self.calls = 0
        self._cycle = itertools.cycle(["ok", "diverge", "crash"])

    def invoke(self, state, *args, **kwargs):
        self.calls += 1
        mode = next(self._cycle)
        if mode == "crash":
            raise RuntimeError("flaky upstream")
        # Output is independent of the input, so an "ok" rollout compares
        # equal to the baseline whatever the perturbation touched.
        out = dict(INIT)
        out["reply"] = "56" if mode == "ok" else "I don't know"
        return out


def test_default_is_one_rollout_and_shape_unchanged():
    g = _CountingGraph()
    res = drift_test(graph=g, initial_state=INIT, intensity="light", seed=1)
    assert res.perturbation_rollouts == 1
    assert res.n_intermittent == 0
    for p in res.perturbations:
        assert p.n_rollouts == 1
        assert p.rollouts == []
        assert p.rollout_index == 0
        assert p.outcome_label() == p.outcome
    # one baseline + one run per perturbation
    assert g.calls == 1 + len(res.perturbations)
    assert res.n_graph_runs == g.calls


def test_n_rollouts_runs_each_perturbation_n_times():
    g = _CountingGraph()
    res = drift_test(
        graph=g, initial_state=INIT, intensity="light", seed=1, perturbation_rollouts=3,
    )
    assert res.perturbation_rollouts == 3
    assert g.calls == 1 + 3 * len(res.perturbations)
    assert res.n_graph_runs == g.calls
    for p in res.perturbations:
        assert p.n_rollouts == 3
        assert len(p.rollouts) == 2
        assert sorted(r.rollout_index for r in p.all_rollouts()) == [0, 1, 2]
        # every rollout saw the SAME perturbed input
        for r in p.rollouts:
            assert r.perturbed_initial_state == p.perturbed_initial_state
            assert r.event_name == p.event_name
            assert r.rollouts == []  # no nesting


def test_rollouts_get_distinct_thread_ids_when_checkpointed():
    """Each rollout is its own LangGraph thread when the graph has a
    checkpointer, so repeats never share state."""
    pytest.importorskip("langgraph")
    from typing import TypedDict

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    S = TypedDict("S", {"query": str, "reply": str, "attempts": int, "approved": bool})

    def answer(state):
        return {"reply": "56"}

    sg = StateGraph(S)
    sg.add_node("answer", answer)
    sg.add_edge(START, "answer")
    sg.add_edge("answer", END)
    graph = sg.compile(checkpointer=MemorySaver())

    res = drift_test(
        graph=graph, initial_state=INIT, intensity="light", seed=1,
        max_perturbations=2, perturbation_rollouts=3,
    )
    assert res.perturbations
    for p in res.perturbations:
        ids = [r.thread_id for r in p.all_rollouts()]
        assert all(ids), ids
        assert len(set(ids)) == 3, ids
        assert ids[0].endswith(f"-p{res.perturbations.index(p)}")
        assert ids[1].endswith("r1") and ids[2].endswith("r2")


def test_representative_is_most_severe_and_intermittent_is_flagged():
    g = _FlakyGraph()
    res = drift_test(
        graph=g, initial_state=INIT, intensity="light", seed=1,
        max_perturbations=1, perturbation_rollouts=3,
    )
    assert len(res.perturbations) == 1
    p = res.perturbations[0]
    # baseline consumed "ok"; the 3 rollouts got diverge, crash, ok
    assert p.outcome_counts == {"crashed": 1, "diverged": 1, "unchanged": 1}
    assert p.crashed, "the representative must be the crashed rollout"
    assert p.outcome == "crashed"
    assert p.is_intermittent
    assert res.n_intermittent == 1
    assert res.n_crashed == 1  # top-level counts read "in at least one rollout"
    assert p.outcome_label() == "crashed 1/3 · diverged 1/3 · unchanged 1/3"
    assert any("intermittent" in line for line in res.summary_lines())
    assert any("rollouts" in line for line in res.summary_lines())


def test_consistent_rollouts_are_not_intermittent():
    g = _CountingGraph()
    res = drift_test(
        graph=g, initial_state=INIT, intensity="light", seed=1, perturbation_rollouts=2,
    )
    assert res.n_intermittent == 0
    for p in res.perturbations:
        assert not p.is_intermittent
        counts = p.outcome_counts
        assert sum(counts.values()) == 2
        assert sum(1 for v in counts.values() if v) == 1


def test_finding_rollout_counts_count_rollouts_not_findings():
    p = PerturbationResult(
        event_name="e", event_summary="", perturbed_field="f", pattern_type="x",
        perturbed_initial_state={},
        coordination_findings=[
            {"failure_type": "tool_call_loop"}, {"failure_type": "tool_call_loop"},
        ],
        rollouts=[
            PerturbationResult(
                event_name="e", event_summary="", perturbed_field="f", pattern_type="x",
                perturbed_initial_state={}, rollout_index=1,
                coordination_findings=[{"failure_type": "silent_tool_failure"}],
            ),
            PerturbationResult(
                event_name="e", event_summary="", perturbed_field="f", pattern_type="x",
                perturbed_initial_state={}, rollout_index=2,
                coordination_findings=[{"failure_type": "tool_call_loop"}],
            ),
        ],
    )
    assert p.n_rollouts == 3
    # two findings in rollout 0 count once; rollout 2 adds one → 2 of 3
    assert p.finding_rollout_counts == {"silent_tool_failure": 1, "tool_call_loop": 2}


def test_invalid_rollouts_rejected():
    with pytest.raises(ValueError, match="perturbation_rollouts"):
        drift_test(graph=_CountingGraph(), initial_state=INIT, perturbation_rollouts=0)


def test_async_variant_accepts_rollouts():
    import asyncio

    g = _CountingGraph()
    res = asyncio.run(drift_test_async(
        graph=g, initial_state=INIT, intensity="light", seed=1,
        max_perturbations=2, perturbation_rollouts=2,
    ))
    assert isinstance(res, AdapterResult)
    assert all(p.n_rollouts == 2 for p in res.perturbations)


def test_server_serialises_rollouts_for_the_bundled_graph():
    """POST /api/adapter-demo with perturbation_rollouts=2 → every
    perturbation row carries n_rollouts, outcome_counts and the other
    rollouts in full; the top level reports n_intermittent / n_graph_runs."""
    from fastapi.testclient import TestClient

    from drift.server import create_app

    client = TestClient(create_app())
    resp = client.post("/api/adapter-demo", json={
        "graph_name": "ticket_triage", "intensity": "light", "seed": 3,
        "max_perturbations": 2, "judge": "off", "divergence_mode": "exact",
        "perturbation_rollouts": 2,
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["perturbation_rollouts"] == 2
    assert "n_intermittent" in data
    assert data["n_graph_runs"] == 1 + 2 * len(data["perturbations"])
    for p in data["perturbations"]:
        assert p["n_rollouts"] == 2
        assert set(p["outcome_counts"]) == {"crashed", "diverged", "unchanged"}
        assert sum(p["outcome_counts"].values()) == 2
        assert isinstance(p["is_intermittent"], bool)
        assert "outcome_label" in p and "finding_rollout_counts" in p
        assert len(p["rollouts"]) == 1
        r = p["rollouts"][0]
        assert "rollouts" not in r, "rollouts must not nest"
        assert r["n_rollouts"] == 2 and r["rollout_index"] != p["rollout_index"]
        assert "trace" in r and "coordination_findings" in r


def test_server_rejects_out_of_range_rollouts():
    from fastapi.testclient import TestClient

    from drift.server import create_app

    client = TestClient(create_app())
    resp = client.post("/api/adapter-demo", json={
        "graph_name": "ticket_triage", "perturbation_rollouts": 0,
    })
    assert resp.status_code == 422
