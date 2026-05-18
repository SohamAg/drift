"""Tests for LLM-judged failure detection.

Covers:
  - build_judge factory: each spec resolves to the right class.
  - Window rendering: judge gets the recent slice in compact form.
  - Response parsing: tolerates extraction from prose, drops bad families,
    parses well-formed payloads into FailureRecord with `llm:` prefix.
  - Detector behavior: respects `every`, dedupes via already_reported.
  - End-to-end via drift.run: judge LLM gets called, judge failures appear
    in result.failures alongside deterministic ones, runner's async-detector
    path actually awaits coroutines.
"""
from __future__ import annotations

import asyncio
import json

import pytest

import drift
from drift.failures.base import DetectorContext, FailureRecord
from drift.failures.judge import (
    DEFAULT_JUDGE_SYSTEM,
    JUDGE_FAMILIES,
    JUDGE_PREFIX,
    AnthropicJudge,
    LLMJudgeDetector,
    OpenAIJudge,
    ScriptedMockJudge,
    _parse_judge_response,
    _render_window,
    build_judge,
)
from drift.world import World, WorldState


# ---- a controllable mock judge for tests ---------------------------------


class _CannedJudge:
    """Judge that returns whatever payload the test sets on it.

    `record_calls=True` keeps a list of (system, user) pairs received so
    tests can assert the judge was actually called with the right window.
    """

    def __init__(self, payload: dict | str) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    async def judge(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        if isinstance(self.payload, str):
            return self.payload
        return json.dumps(self.payload)


# ---- build_judge factory -------------------------------------------------


def test_build_judge_off_returns_none():
    assert build_judge("off") is None
    assert build_judge("") is None
    assert build_judge(None) is None  # type: ignore[arg-type]


def test_build_judge_mock_returns_scripted():
    j = build_judge("mock")
    assert isinstance(j, ScriptedMockJudge)


def test_build_judge_anthropic_returns_stub():
    j = build_judge("anthropic", model="claude-haiku-4-5")
    assert isinstance(j, AnthropicJudge)
    assert j.model == "claude-haiku-4-5"


def test_build_judge_rejects_unknown_spec():
    with pytest.raises(ValueError):
        build_judge("gemini")


def test_build_judge_openai_uses_default_model_without_construction(monkeypatch):
    # OpenAI client validates the api key at construction; skip if openai
    # isn't installed locally, and provide a dummy key so we can verify
    # the factory wires the model through.
    try:
        import openai  # noqa: F401
    except ImportError:
        pytest.skip("openai not installed locally")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    j = build_judge("openai", model="gpt-4o-mini")
    assert isinstance(j, OpenAIJudge)
    assert j.model == "gpt-4o-mini"


# ---- window rendering ----------------------------------------------------


def _make_ctx(actions: list[drift.Action], timestep: int, history_state: WorldState | None = None) -> DetectorContext:
    world = World(initial=history_state or WorldState())
    world.begin_step(timestep)
    world.commit_step()
    return DetectorContext(
        timestep=timestep,
        history=world.history,
        actions=actions,
        events=[],
        already_reported=set(),
    )


def test_render_window_trims_to_recent_actions():
    actions = [
        drift.Action(action_id="a000001", timestep=1, agent_name="alice", kind="approve", target_case_id="C1"),
        drift.Action(action_id="a000002", timestep=2, agent_name="bob", kind="reject", target_case_id="C1"),
        drift.Action(action_id="a000003", timestep=10, agent_name="alice", kind="approve", target_case_id="C2"),
    ]
    ctx = _make_ctx(actions, timestep=10)
    rendered = _render_window(ctx, window_steps=3)
    assert "a000003" in rendered
    # Older actions outside the window should be dropped.
    assert "a000001" not in rendered
    assert "a000002" not in rendered
    assert "t=8..10" in rendered


# ---- response parsing ----------------------------------------------------


def test_parse_well_formed_response_emits_failure_records():
    ctx = _make_ctx([], timestep=5)
    raw = json.dumps({
        "failures": [
            {
                "family": "coordination_contradiction",
                "summary": "alice approved while bob rejected case C1",
                "evidence_action_ids": ["a000001", "a000002"],
                "agents_involved": ["alice", "bob"],
            },
        ]
    })
    out = _parse_judge_response(raw, ctx)
    assert len(out) == 1
    f = out[0]
    assert isinstance(f, FailureRecord)
    assert f.failure_type == f"{JUDGE_PREFIX}coordination_contradiction"
    assert f.evidence_action_ids == ["a000001", "a000002"]
    assert f.agents_involved == ["alice", "bob"]
    assert "alice approved" in f.summary


def test_parse_drops_unknown_families():
    ctx = _make_ctx([], timestep=5)
    raw = json.dumps({"failures": [{"family": "made_up_family", "summary": "x"}]})
    assert _parse_judge_response(raw, ctx) == []


def test_parse_extracts_json_wrapped_in_prose():
    # Some models still wrap JSON in explanatory prose; we should cope.
    ctx = _make_ctx([], timestep=5)
    raw = 'Sure, here you go: {"failures": []} (no issues found)'
    assert _parse_judge_response(raw, ctx) == []


def test_parse_returns_empty_on_garbage():
    ctx = _make_ctx([], timestep=5)
    assert _parse_judge_response("", ctx) == []
    assert _parse_judge_response("not json at all", ctx) == []


def test_parse_handles_empty_failures_list():
    ctx = _make_ctx([], timestep=5)
    assert _parse_judge_response('{"failures": []}', ctx) == []


# ---- detector behavior ---------------------------------------------------


def test_detector_skips_when_not_at_cadence():
    judge = _CannedJudge({"failures": []})
    det = LLMJudgeDetector(judge, every=5, window=5)
    ctx = _make_ctx([], timestep=3)
    result = asyncio.run(det(ctx))
    assert result == []
    assert judge.calls == []  # judge was not consulted at t=3


def test_detector_fires_at_cadence_boundary():
    judge = _CannedJudge({"failures": []})
    det = LLMJudgeDetector(judge, every=5, window=5)
    ctx = _make_ctx([], timestep=10)
    asyncio.run(det(ctx))
    assert len(judge.calls) == 1
    system, _ = judge.calls[0]
    assert system == DEFAULT_JUDGE_SYSTEM


def test_detector_dedupes_repeated_findings():
    payload = {"failures": [{
        "family": "state_drift",
        "summary": "policy version drift on case C1",
        "evidence_action_ids": ["a000001"],
        "agents_involved": ["alice"],
    }]}
    judge = _CannedJudge(payload)
    det = LLMJudgeDetector(judge, every=5, window=5)

    reported: set[str] = set()

    def call_at(t: int) -> list[FailureRecord]:
        ctx = DetectorContext(
            timestep=t,
            history=World().history,
            actions=[],
            events=[],
            already_reported=reported,
        )
        return asyncio.run(det(ctx))

    first = call_at(5)
    second = call_at(10)
    assert len(first) == 1
    assert second == []  # same summary -> already reported


def test_detector_distinguishes_different_findings_of_same_type():
    judge = _CannedJudge({})
    det = LLMJudgeDetector(judge, every=5, window=5)
    reported: set[str] = set()

    judge.payload = {"failures": [{
        "family": "gate_bypass",
        "summary": "merger merged PR-1 while security blocked",
        "evidence_action_ids": [], "agents_involved": [],
    }]}
    ctx = DetectorContext(timestep=5, history=World().history, actions=[], events=[], already_reported=reported)
    assert len(asyncio.run(det(ctx))) == 1

    judge.payload = {"failures": [{
        "family": "gate_bypass",
        "summary": "merger merged PR-2 while security blocked",
        "evidence_action_ids": [], "agents_involved": [],
    }]}
    ctx2 = DetectorContext(timestep=10, history=World().history, actions=[], events=[], already_reported=reported)
    # Different summary -> different fingerprint -> reports again.
    assert len(asyncio.run(det(ctx2))) == 1


def test_judge_families_match_skill_md():
    # Hard-coded list as a safety net — if someone edits the families,
    # the test forces them to update both places consciously.
    assert JUDGE_FAMILIES == (
        "coordination_contradiction",
        "grounding_failure",
        "state_drift",
        "emergent_decay",
        "gate_bypass",
    )


# ---- scripted mock judge -------------------------------------------------


def test_scripted_mock_judge_fires_placeholder_once_per_instance():
    j = ScriptedMockJudge()
    a = asyncio.run(j.judge(system="x", user="y"))
    b = asyncio.run(j.judge(system="x", user="y"))
    a_failures = json.loads(a)["failures"]
    b_failures = json.loads(b)["failures"]
    assert len(a_failures) == 1
    assert a_failures[0]["family"] == "emergent_decay"
    assert "placeholder" in a_failures[0]["summary"].lower()
    assert b_failures == []


# ---- end-to-end via drift.run --------------------------------------------


def test_drift_run_with_canned_judge_produces_llm_failures():
    judge = _CannedJudge({"failures": [{
        "family": "coordination_contradiction",
        "summary": "agents disagreed on case C1",
        "evidence_action_ids": [],
        "agents_involved": ["a", "b"],
    }]})

    @drift.agent(role="alice")
    async def alice(state, memory):
        return drift.Action(kind="no_op")

    @drift.agent(role="bob")
    async def bob(state, memory):
        return drift.Action(kind="no_op")

    result = drift.run(
        agents=[alice, bob],
        steps=10,           # judge fires at t=5 and t=10
        seed=1,
        judge_llm=judge,
        judge_every=5,
    )
    # Judge was called twice (t=5 and t=10).
    assert len(judge.calls) == 2
    # But dedupe keeps the failure list to 1 (same summary both times).
    llm_failures = [f for f in result.failures if f.failure_type.startswith(JUDGE_PREFIX)]
    assert len(llm_failures) == 1
    assert llm_failures[0].failure_type == f"{JUDGE_PREFIX}coordination_contradiction"


def test_drift_run_without_judge_makes_no_calls():
    # Sanity: if judge_llm is None, nothing async happens in the detector path.
    @drift.agent(role="alice")
    async def alice(state, memory):
        return drift.Action(kind="no_op")

    result = drift.run(agents=[alice], steps=10, seed=1)
    assert all(not f.failure_type.startswith(JUDGE_PREFIX) for f in result.failures)


def test_runner_awaits_async_detectors_alongside_sync():
    """The async detector must coexist with the sync detector pipeline —
    sync detectors keep firing, async detector fires too."""
    judge = _CannedJudge({"failures": [{
        "family": "emergent_decay",
        "summary": "sentiment trending bad",
        "evidence_action_ids": [], "agents_involved": [],
    }]})

    @drift.agent(role="alice")
    async def alice(state, memory):
        return drift.Action(kind="no_op")

    # Force a deterministic detector to also fire by using auto-chaos that
    # clears open_cases mid-run AND an agent that references a case.
    class _State(drift.WorldState):
        open_cases: dict = {}

    @drift.agent(role="ref")
    async def ref(state, memory):
        return drift.Action(kind="approve", target_case_id="missing-case")

    result = drift.run(
        agents=[ref],
        state=_State(),
        steps=10,
        seed=2,
        judge_llm=judge,
        judge_every=5,
    )
    types = {f.failure_type for f in result.failures}
    # Deterministic hallucinated_reference should still fire.
    assert "hallucinated_reference" in types
    # And the judge should have fired its emergent_decay too.
    assert f"{JUDGE_PREFIX}emergent_decay" in types
