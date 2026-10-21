"""Tests for drift.target_builder — the Build tab's codegen backend.

Positive tests write the generated source to a real temp .py file and
import it via importlib (matching how the server actually loads a saved
generated target — the same path _build_custom_graph uses for every
other target), then build() a real LangGraph graph (no LLM call —
build() just wires nodes/edges, nothing invokes until graph.invoke()).

A bare exec(source, {}) was tried first and rejected: the generated
module uses `from __future__ import annotations` + a TypedDict with
Annotated field reducers, and typing.get_type_hints() resolves those
forward-reference annotations against sys.modules[cls.__module__] —
which for a class defined via bare exec() into a throwaway dict
resolves to the test process's own __main__ module, not the dict that
was actually passed in, so lookups for names the generated source itself
imports (Annotated, operator) spuriously fail. Real imports vs disk
don't hit this: a real module is registered in sys.modules under its
own name with its own __dict__.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import uuid

import pytest

import drift.target_builder as target_builder_module
from drift.target_builder import (
    TOPOLOGY_MIN_AGENTS,
    ScenarioProposalError,
    TargetBuilderError,
    _parse_proposal,
    generate_target_source,
    propose_target,
    slugify_agent_name,
    validate_target_name,
)


def _exec_build(source: str, tmp_path):
    path = tmp_path / f"generated_{uuid.uuid4().hex}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    graph = module.build()
    state = module.initial_state("a test query")
    return graph, state, module


SUPERVISOR_AGENTS = [
    {"name": "coordinator", "role": "Break the task into subtasks."},
    {"name": "researcher", "role": "Research facts."},
    {"name": "writer", "role": "Write the final answer."},
]
SWARM_AGENTS = [
    {"name": "intake", "role": "Understand the request."},
    {"name": "solver", "role": "Solve it."},
]
PIPELINE_AGENTS = [
    {"name": "draft", "role": "Draft a response."},
    {"name": "edit", "role": "Edit for clarity."},
]
LOOP_AGENTS = [
    {"name": "generator", "role": "Generate a solution."},
    {"name": "critic", "role": "Review it; say APPROVED or REVISE."},
]
FANOUT_AGENTS = [
    {"name": "dispatcher", "role": "Split the task."},
    {"name": "worker_a", "role": "Cover angle A."},
    {"name": "worker_b", "role": "Cover angle B."},
    {"name": "merge", "role": "Combine findings."},
]


@pytest.mark.parametrize(
    "topology,agents",
    [
        ("supervisor", SUPERVISOR_AGENTS),
        ("swarm", SWARM_AGENTS),
        ("pipeline", PIPELINE_AGENTS),
        ("loop", LOOP_AGENTS),
        ("fanout", FANOUT_AGENTS),
    ],
)
def test_generated_source_compiles_to_a_real_graph(topology, agents, tmp_path):
    source = generate_target_source(topology, agents)
    graph, state, module = _exec_build(source, tmp_path)
    assert type(graph).__name__ == "CompiledStateGraph"
    assert state == {"task": "a test query", "notes": {}, "_iterations": 0}
    assert callable(module.build)
    assert callable(module.initial_state)


def test_generated_source_is_valid_python_syntax_standalone():
    """compile() alone, independent of exec — catches syntax errors the
    exec-based tests above would also catch, but isolates the failure
    mode (syntax vs. runtime wiring error) when something breaks."""
    for topology, agents in [
        ("supervisor", SUPERVISOR_AGENTS), ("swarm", SWARM_AGENTS),
        ("pipeline", PIPELINE_AGENTS), ("loop", LOOP_AGENTS), ("fanout", FANOUT_AGENTS),
    ]:
        source = generate_target_source(topology, agents)
        compile(source, f"<generated:{topology}>", "exec")  # raises SyntaxError if bad


def test_user_text_with_quotes_and_newlines_embeds_safely(tmp_path):
    """Role text containing quotes/newlines/backslashes must not break the
    generated source — proves safe embedding, not raw interpolation."""
    agents = [
        {"name": "tricky", "role": 'Say "hello" and don\'t\nuse semicolons; or \\ backslashes.'},
        {"name": "other", "role": "Respond normally."},
    ]
    source = generate_target_source("pipeline", agents)
    graph, state, module = _exec_build(source, tmp_path)
    assert type(graph).__name__ == "CompiledStateGraph"


def test_user_text_with_triple_quotes_embeds_safely(tmp_path):
    """A literal \"\"\" in role text must not prematurely terminate the
    generated module's docstring (would otherwise be a SyntaxError)."""
    agents = [
        {"name": "tricky", "role": 'Ends the """docstring""" early if unescaped.'},
        {"name": "other", "role": "Respond normally."},
    ]
    source = generate_target_source("pipeline", agents)
    graph, state, module = _exec_build(source, tmp_path)
    assert type(graph).__name__ == "CompiledStateGraph"


def test_fanout_wires_dispatcher_to_each_worker_and_each_worker_to_merge(tmp_path):
    source = generate_target_source("fanout", FANOUT_AGENTS)
    graph, _, _ = _exec_build(source, tmp_path)
    g = graph.get_graph()
    edge_pairs = {(e.source, e.target) for e in g.edges}
    assert ("dispatcher", "worker_a") in edge_pairs
    assert ("dispatcher", "worker_b") in edge_pairs
    assert ("worker_a", "merge") in edge_pairs
    assert ("worker_b", "merge") in edge_pairs


def test_loop_wires_a_conditional_edge_back_to_the_generator(tmp_path):
    source = generate_target_source("loop", LOOP_AGENTS)
    graph, _, _ = _exec_build(source, tmp_path)
    g = graph.get_graph()
    edge_pairs = {(e.source, e.target) for e in g.edges}
    assert ("generator", "critic") in edge_pairs
    # conditional edge back to generator is present among critic's outgoing edges
    assert any(src == "critic" and dst == "generator" for src, dst in edge_pairs)


def test_supervisor_adds_a_synthesis_node_after_specialists(tmp_path):
    source = generate_target_source("supervisor", SUPERVISOR_AGENTS)
    graph, _, _ = _exec_build(source, tmp_path)
    g = graph.get_graph()
    assert "synthesis" in g.nodes
    edge_pairs = {(e.source, e.target) for e in g.edges}
    assert ("writer", "synthesis") in edge_pairs


@pytest.mark.parametrize("topology", list(TOPOLOGY_MIN_AGENTS))
def test_too_few_agents_raises(topology):
    min_n = TOPOLOGY_MIN_AGENTS[topology]
    too_few = [{"name": f"a{i}", "role": "does something"} for i in range(min_n - 1)]
    with pytest.raises(TargetBuilderError):
        generate_target_source(topology, too_few)


def test_loop_rejects_more_than_two_agents():
    with pytest.raises(TargetBuilderError):
        generate_target_source("loop", LOOP_AGENTS + [{"name": "extra", "role": "x"}])


def test_unknown_topology_raises():
    with pytest.raises(TargetBuilderError):
        generate_target_source("not_a_real_topology", SWARM_AGENTS)


def test_missing_role_raises():
    with pytest.raises(TargetBuilderError):
        generate_target_source("pipeline", [
            {"name": "a", "role": "does something"},
            {"name": "b", "role": ""},
        ])


def test_duplicate_agent_names_after_slugifying_raises():
    with pytest.raises(TargetBuilderError):
        generate_target_source("pipeline", [
            {"name": "my-agent", "role": "x"},
            {"name": "my_agent", "role": "y"},  # slugifies to the same identifier
        ])


def test_slugify_agent_name():
    assert slugify_agent_name("Research Agent") == "research_agent"
    assert slugify_agent_name("worker-1") == "worker_1"
    with pytest.raises(TargetBuilderError):
        slugify_agent_name("1_starts_with_digit")
    with pytest.raises(TargetBuilderError):
        slugify_agent_name("!!!")


def test_validate_target_name():
    assert validate_target_name("My Cool System") == "my_cool_system"
    with pytest.raises(TargetBuilderError):
        validate_target_name("a")  # too short after slugifying


# ---------------------------------------------------------------------------
# AI-assisted proposal (propose_target) — the LLM call itself is mocked via
# monkeypatching _call_proposal_llm, never a real API call. End-to-end proof
# that a real proposal flows into the real generator was done manually
# during development (scenario "how would a new sci-fi action movie do in
# theaters" -> fanout topology, 5 agents, compiled and ran for real).
# ---------------------------------------------------------------------------


VALID_PROPOSAL_JSON = """{
    "topology": "fanout",
    "rationale": "Several independent angles can be explored in parallel.",
    "suggested_name": "movie_success",
    "suggested_query": "How would a new sci-fi action movie do in theaters?",
    "agents": [
        {"name": "dispatcher", "role": "Split the question into angles."},
        {"name": "marketing", "role": "Assess marketing strength."},
        {"name": "star_power", "role": "Assess cast/director draw."},
        {"name": "merge", "role": "Combine all findings into one verdict."}
    ]
}"""


def test_parse_proposal_valid():
    result = _parse_proposal(VALID_PROPOSAL_JSON)
    assert result.topology == "fanout"
    assert result.suggested_name == "movie_success"
    assert len(result.agents) == 4
    assert result.agents[0] == {"name": "dispatcher", "role": "Split the question into angles."}


def test_parse_proposal_rejects_invalid_json():
    with pytest.raises(ScenarioProposalError):
        _parse_proposal("not json at all")


def test_parse_proposal_rejects_non_object_json():
    with pytest.raises(ScenarioProposalError):
        _parse_proposal("[1, 2, 3]")


def test_parse_proposal_rejects_unknown_topology():
    with pytest.raises(ScenarioProposalError):
        _parse_proposal('{"topology": "made_up", "agents": [{"name": "a", "role": "b"}]}')


def test_parse_proposal_rejects_empty_agent_list():
    with pytest.raises(ScenarioProposalError):
        _parse_proposal('{"topology": "pipeline", "agents": []}')


def test_parse_proposal_skips_malformed_agent_entries_but_keeps_valid_ones():
    raw = json.dumps({
        "topology": "pipeline",
        "agents": [
            {"name": "good", "role": "does something"},
            {"name": "", "role": "missing a name"},
            "not even a dict",
            {"name": "also_good", "role": "does something else"},
        ],
    })
    result = _parse_proposal(raw)
    assert [a["name"] for a in result.agents] == ["good", "also_good"]


def test_parse_proposal_defaults_missing_optional_fields_to_empty_string():
    raw = json.dumps({"topology": "swarm", "agents": [{"name": "a", "role": "b"}]})
    result = _parse_proposal(raw)
    assert result.rationale == ""
    assert result.suggested_name == ""
    assert result.suggested_query == ""


def test_propose_target_rejects_empty_scenario():
    # No pytest-asyncio in this project (see module docstring note) — every
    # other async test here goes through asyncio.run(), matching how the
    # rest of the codebase tests async code (drift_test wraps
    # drift_test_async the same way).
    with pytest.raises(ScenarioProposalError):
        asyncio.run(propose_target("   "))


def test_propose_target_parses_the_llm_response(monkeypatch: pytest.MonkeyPatch):
    async def fake_call(system, user, model):
        assert "movie" in user.lower()
        return VALID_PROPOSAL_JSON
    monkeypatch.setattr(target_builder_module, "_call_proposal_llm", fake_call)

    result = asyncio.run(propose_target("how would a new sci-fi movie do in theaters"))
    assert result.topology == "fanout"
    assert len(result.agents) == 4


def test_propose_target_propagates_llm_call_failures(monkeypatch: pytest.MonkeyPatch):
    async def failing_call(system, user, model):
        raise ScenarioProposalError("LLM call failed (RuntimeError): no API key")
    monkeypatch.setattr(target_builder_module, "_call_proposal_llm", failing_call)

    with pytest.raises(ScenarioProposalError):
        asyncio.run(propose_target("any scenario"))


def test_proposed_agents_feed_directly_into_generate_target_source():
    """The contract that matters: a parsed proposal's (topology, agents)
    must be exactly what generate_target_source() already accepts — no
    adapter/transform layer in between."""
    proposal = _parse_proposal(VALID_PROPOSAL_JSON)
    source = generate_target_source(proposal.topology, proposal.agents)
    assert "def build(" in source
    assert "dispatcher" in source and "merge" in source
