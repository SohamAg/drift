"""Tool-dispatched sub-agents get a name.

deepagents (and any graph that starts a sub-agent from inside a tool call)
streams the sub-agent's steps under the PARENT node's namespace:
`tools/model`, `tools/tools`. Three sub-agents, one label. The dispatching
tool call in the root conversation knows who it called
(`task(description=..., subagent_type="critic")`), and the subgraph's own
first human message is that description — so the two can be joined."""

from __future__ import annotations

from drift.adapters.langgraph import _infer_subagent, _NativeTraceBuilder
from drift.failures.library.base import TraceStep, from_adapter_trace


def _ai(tool_calls):
    return {"type": "ai", "content": "", "tool_calls": tool_calls}


def _task(cid, description, who):
    return {"id": cid, "name": "task", "args": {"description": description, "subagent_type": who}}


def _tool(cid, text="done"):
    return {"type": "tool", "content": text, "tool_call_id": cid}


def _sub(first_human):
    return {"messages": [{"type": "human", "content": first_human}]}


ROOT_ONE = {"messages": [
    {"type": "human", "content": "research X"},
    _ai([_task("c1", "Research key facts about X.", "researcher")]),
]}


def test_single_pending_call_names_the_subgraph():
    assert _infer_subagent(ROOT_ONE, _sub("Research key facts about X.")) == "researcher"


def test_single_pending_call_without_text_match_still_names():
    # Sub-agent rewrote the task into a system+human pair we can't match on,
    # but there is exactly one candidate — unambiguous.
    assert _infer_subagent(ROOT_ONE, {"messages": [{"type": "human", "content": "Hi"}]}) == "researcher"


def test_parallel_dispatch_is_disambiguated_by_task_text():
    root = {"messages": [
        {"type": "human", "content": "compare"},
        _ai([
            _task("c1", "Research LangGraph.", "researcher"),
            _task("c2", "Critique the plan.", "critic"),
        ]),
    ]}
    assert _infer_subagent(root, _sub("Research LangGraph.")) == "researcher"
    assert _infer_subagent(root, _sub("Critique the plan.")) == "critic"
    # template wrapper around the task text still matches
    assert _infer_subagent(root, _sub("Task: Critique the plan.\nBe brief.")) == "critic"


def test_ambiguous_parallel_dispatch_returns_empty_not_a_guess():
    root = {"messages": [
        _ai([
            _task("c1", "Do A.", "researcher"),
            _task("c2", "Do B.", "critic"),
        ]),
    ]}
    assert _infer_subagent(root, _sub("Something unrelated")) == ""
    assert _infer_subagent(root, None) == ""


def test_same_name_parallel_dispatch_resolves_to_that_name():
    root = {"messages": [_ai([
        _task("c1", "Research A.", "researcher"),
        _task("c2", "Research B.", "researcher"),
    ])]}
    assert _infer_subagent(root, _sub("unmatched")) == "researcher"


def test_answered_calls_are_not_candidates():
    root = {"messages": [
        _ai([_task("c1", "Research A.", "researcher")]),
        _tool("c1"),
        _ai([_task("c2", "Critique A.", "critic")]),
    ]}
    assert _infer_subagent(root, _sub("nope")) == "critic"


def test_specific_tool_name_is_the_agent_when_args_do_not_name_one():
    root = {"messages": [_ai([
        {"id": "c1", "name": "research_agent", "args": {"query": "LangGraph"}},
    ])]}
    assert _infer_subagent(root, _sub("LangGraph")) == "research_agent"


def test_generic_dispatch_tool_without_a_name_arg_stays_anonymous():
    root = {"messages": [_ai([
        {"id": "c1", "name": "delegate", "args": {"task": "do it"}},
    ])]}
    assert _infer_subagent(root, _sub("do it")) == ""


def test_string_json_args_are_parsed():
    root = {"messages": [_ai([
        {"id": "c1", "name": "task", "args": '{"description": "Do A.", "subagent_type": "critic"}'},
    ])]}
    assert _infer_subagent(root, _sub("Do A.")) == "critic"


def test_no_pending_calls_means_no_name():
    assert _infer_subagent({"messages": [{"type": "human", "content": "x"}]}, _sub("x")) == ""
    assert _infer_subagent({}, _sub("x")) == ""


def test_builder_stamps_subagent_on_subgraph_steps_only():
    """Feed the builder a deepagents-shaped stream: root model fans out two
    task calls, two sub-agents run interleaved under `tools:<task>`, then the
    tools node commits both results."""
    trace: list[dict] = []
    b = _NativeTraceBuilder({"messages": [{"type": "human", "content": "compare"}]}, trace, None)
    ai = _ai([
        _task("c1", "Research LangGraph.", "researcher"),
        _task("c2", "Critique LangGraph.", "critic"),
    ])
    root_after_model = {"messages": [{"type": "human", "content": "compare"}, ai]}
    b.feed(((), "updates", {"model": {"messages": [ai]}}))
    b.feed(((), "values", root_after_model))
    ns = ("tools:abc123",)
    b.feed((ns, "updates", {"model": {"messages": [{"type": "ai", "content": "LangGraph is a graph."}]}}))
    b.feed((ns, "values", {"messages": [
        {"type": "human", "content": "Research LangGraph."},
        {"type": "ai", "content": "LangGraph is a graph."},
    ]}))
    b.feed((ns, "updates", {"model": {"messages": [{"type": "ai", "content": "Too shallow."}]}}))
    b.feed((ns, "values", {"messages": [
        {"type": "human", "content": "Critique LangGraph."},
        {"type": "ai", "content": "Too shallow."},
    ]}))
    root_after_tools = {"messages": root_after_model["messages"] + [_tool("c1", "LangGraph is a graph."), _tool("c2", "Too shallow.")]}
    b.feed(((), "updates", {"tools": {"messages": [_tool("c1", "LangGraph is a graph."), _tool("c2", "Too shallow.")]}}))
    b.feed(((), "values", root_after_tools))
    b.finish()

    nodes = [(r["node"], r.get("subagent")) for r in trace]
    assert nodes == [
        ("model", None),
        ("tools/model", "researcher"),
        ("tools/model", "critic"),
        ("tools", None),
    ], nodes

    ctx = from_adapter_trace(trace)
    assert [s.subagent for s in ctx.steps] == ["", "researcher", "critic", ""]
    assert ctx.steps[1].actor == "researcher (tools/model)"
    assert ctx.steps[0].actor == "model"


def test_tracestep_actor_defaults_to_agent():
    assert TraceStep(step=1, agent="supervisor", kind="node:supervisor").actor == "supervisor"


def test_subgraph_tail_step_without_values_chunk_inherits_the_name():
    """deepagents' `after_model` middleware step is the subgraph's last; its
    values chunk never comes (the parent's does). It still gets the name."""
    trace: list[dict] = []
    b = _NativeTraceBuilder({"messages": [{"type": "human", "content": "go"}]}, trace, None)
    ai = _ai([_task("c1", "Research A.", "researcher")])
    b.feed(((), "updates", {"model": {"messages": [ai]}}))
    b.feed(((), "values", {"messages": [{"type": "human", "content": "go"}, ai]}))
    ns = ("tools:t1",)
    b.feed((ns, "updates", {"model": {"messages": [{"type": "ai", "content": "A is…"}]}}))
    b.feed((ns, "values", {"messages": [{"type": "human", "content": "Research A."}, {"type": "ai", "content": "A is…"}]}))
    b.feed((ns, "updates", {"after_model": {"_value": None}}))  # no values chunk follows
    b.feed(((), "updates", {"tools": {"messages": [_tool("c1", "A is…")]}}))
    b.feed(((), "values", {"messages": [{"type": "human", "content": "go"}, ai, _tool("c1", "A is…")]}))
    b.finish()
    assert [(r["node"], r.get("subagent")) for r in trace] == [
        ("model", None), ("tools/model", "researcher"), ("tools/after_model", "researcher"), ("tools", None),
    ]
    assert trace[2]["state_after"] is not None and "sub_state_after" not in trace[2]
