"""handoff_misroute: a supervisor's handoff to agent X runs agent Y instead.

Real langgraph-supervisor graph, scripted chat models, no API key."""
from __future__ import annotations

import warnings

import pytest

pytest.importorskip("langgraph_supervisor")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from drift.adapters.langgraph import drift_test  # noqa: E402
from drift.chaos.tools import (  # noqa: E402
    DEFAULT_AUTO_FAULTS,
    ToolChaosEvent,
    applies_to,
    discover_tool_patterns,
    is_handoff_tool,
)


class _Supervisor(GenericFakeChatModel):
    """Hands off to research_expert, then repeats whatever the last agent said."""

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if not any(m.type == "tool" for m in messages):
            msg = AIMessage(content="", tool_calls=[{"name": "transfer_to_research_expert", "args": {},
                                                     "id": "h1"}])
        else:
            said = [m.content for m in messages if m.type == "ai" and m.content and m.name != "supervisor"
                    and not m.content.startswith("Transferring back")]
            msg = AIMessage(content=f"Final: {said[-1] if said else 'nothing'}")
        return ChatResult(generations=[ChatGeneration(message=msg)])


class _Worker(GenericFakeChatModel):
    reply: str = ""

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])


def _supervisor():
    from langgraph_supervisor import create_supervisor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from langgraph.prebuilt import create_react_agent

        research = create_react_agent(_Worker(messages=iter([]), reply="NYC has 8.3M people."), [],
                                      name="research_expert")
        math = create_react_agent(_Worker(messages=iter([]), reply="2 + 2 = 4."), [], name="math_expert")
        return create_supervisor([research, math], model=_Supervisor(messages=iter([]))).compile()


STATE = {"messages": [HumanMessage("What is the population of NYC?")]}


def _spec(entries):
    from drift.guidelines import GuidelineSpec

    return GuidelineSpec(chaos=entries)


def _final_text(p):
    return p.final_state["messages"][-1]["content"]


def test_misroute_delivers_the_handoff_to_the_other_agent():
    res = drift_test(graph=_supervisor(), initial_state=STATE, intensity="off",
                     rules=_spec([{"tool": "transfer_to_research_expert", "fault": "handoff_misroute"}]))
    assert "8.3M" in res.baseline.final_state["messages"][-1]["content"]
    (p,) = res.perturbations
    assert p.fault_fired and not p.crashed and p.diverged
    assert "ran 'transfer_to_math_expert' instead" in p.event_summary
    assert "2 + 2 = 4" in _final_text(p) and "8.3M" not in _final_text(p)
    speakers = {m.get("name") for m in p.final_state["messages"] if m["type"] == "ai"}
    assert "math_expert" in speakers and "research_expert" not in speakers


def test_misroute_is_only_planned_on_handoff_tools_with_another_destination():
    g = _supervisor()
    names = {sp.pattern_name for sp in discover_tool_patterns(g, faults=["handoff_misroute"])}
    assert names == {"handoff_misroute[transfer_to_math_expert]", "handoff_misroute[transfer_to_research_expert]"}
    assert "handoff_misroute" in DEFAULT_AUTO_FAULTS
    # transfer_back_to_supervisor lives on each worker's own node with no other destination
    assert not applies_to(g, "transfer_back_to_supervisor", "handoff_misroute")


def test_ordinary_tools_get_no_misroute_and_a_clear_error_when_declared():
    from langchain_core.tools import tool

    @tool
    def lookup(q: str) -> str:
        """Look something up."""
        return q

    assert not is_handoff_tool(lookup)
    from tests.test_chaos_checkpoint import _react, _react_state

    g = _react()
    assert not [sp for sp in discover_tool_patterns(g) if sp.pattern_type == "handoff_misroute"]
    with pytest.raises(ValueError, match="applies only to a handoff tool"):
        drift_test(graph=g, initial_state=_react_state(), intensity="off",
                   rules=_spec([{"tool": "web_search", "fault": "handoff_misroute"}]))


def test_auto_plan_includes_misroute_on_a_supervisor():
    res = drift_test(graph=_supervisor(), initial_state=STATE, intensity="moderate", max_perturbations=50, seed=3)
    planned = {p.event_name for p in res.perturbations}
    assert any("handoff_misroute[" in n for n in planned), planned


def test_install_restores_the_node_and_reports_unfired_misroute():
    g = _supervisor()
    ev = ToolChaosEvent("transfer_to_math_expert", fault="handoff_misroute")
    with ev.install(g):
        g.invoke(STATE)  # the scripted supervisor only ever asks for research
    assert ev.installed_on == ["supervisor/tools"] and not ev.fired
    assert "never made call #1" in ev.outcome_note()
