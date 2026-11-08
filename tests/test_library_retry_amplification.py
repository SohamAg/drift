"""retry_amplification — the graph did ≥2× the baseline's work under a perturbation."""
from __future__ import annotations

from drift.failures.library import ALL_DETECTORS, retry_amplification
from drift.failures.library.base import from_adapter_trace


def _ai(text="", calls=None):
    m = {"type": "ai", "content": text}
    if calls:
        m["tool_calls"] = [{"name": n, "args": a, "id": i} for (n, a, i) in calls]
    return m


def _tool(text, call_id, status=None):
    m = {"type": "tool", "content": text, "tool_call_id": call_id}
    if status:
        m["status"] = status
    return m


def _human(text):
    return {"type": "human", "content": text}


INIT = {"messages": [_human("What's the weather in Paris?")]}


def _trace(steps, initial=INIT):
    running = list(initial.get("messages", []))
    out = []
    for i, (node, msgs) in enumerate(steps, start=1):
        running = running + msgs
        out.append({"step": i, "node": node, "update": {"messages": msgs},
                    "state_after": {"messages": list(running)}})
    return out


def _final_state(trace):
    return trace[-1]["state_after"]


def _healthy_steps(n_calls=1):
    steps = []
    for k in range(n_calls):
        steps.append(("agent", [_ai(calls=[("get_weather", {"city": "Paris"}, f"c{k}")])]))
        steps.append(("tools", [_tool("18°C", f"c{k}")]))
    steps.append(("agent", [_ai("It is 18°C in Paris.")]))
    return steps


def _storm_steps(n_calls):
    steps = []
    for k in range(n_calls):
        steps.append(("agent", [_ai(calls=[("get_weather", {"city": "Paris", "try": k}, f"s{k}")])]))
        steps.append(("tools", [_tool("Error: 503", f"s{k}", "error")]))
    steps.append(("agent", [_ai("It is probably 18°C in Paris.")]))
    return steps


def test_registered_in_library():
    assert retry_amplification in ALL_DETECTORS


def test_silent_without_a_baseline():
    trace = _trace(_storm_steps(6))
    ctx = from_adapter_trace(trace, initial_state=INIT, baseline_state=None)
    assert retry_amplification.detect(ctx) == []


def test_silent_when_work_matches_baseline():
    base = _trace(_healthy_steps(1))
    pert = _trace(_healthy_steps(1))
    ctx = from_adapter_trace(pert, initial_state=INIT, baseline_state=_final_state(base))
    assert retry_amplification.detect(ctx) == []


def test_fires_on_a_retry_storm_and_names_both_metrics():
    base = _trace(_healthy_steps(1))           # 1 call, 2 turns
    pert = _trace(_storm_steps(5))             # 5 calls, 6 turns
    ctx = from_adapter_trace(pert, initial_state=INIT, baseline_state=_final_state(base))
    findings = retry_amplification.detect(ctx)
    assert len(findings) == 1
    f = findings[0]
    assert f.failure_type == "retry_amplification"
    assert "5 tool calls vs 1 at baseline (5.0×)" in f.summary
    assert "6 LLM turns vs 2 at baseline (3.0×)" in f.summary
    assert f.agents_involved == ["agent"]
    assert f.timestep == len(pert)


def test_small_absolute_bump_is_not_amplification():
    """1 → 2 calls is 2.0× but only +1 — noise, not a storm."""
    base = _trace(_healthy_steps(1))
    pert = _trace(_healthy_steps(2))
    ctx = from_adapter_trace(pert, initial_state=INIT, baseline_state=_final_state(base))
    assert retry_amplification.detect(ctx) == []


def test_ratio_below_two_is_not_amplification_even_if_absolute_is_large():
    base = _trace(_healthy_steps(4))   # 4 calls
    pert = _trace(_healthy_steps(7))   # 7 calls: +3 but 1.75×
    ctx = from_adapter_trace(pert, initial_state=INIT, baseline_state=_final_state(base))
    assert retry_amplification.detect(ctx) == []


def test_baseline_with_no_tool_calls_fires_on_three_or_more():
    base = _trace([("agent", [_ai("Paris is sunny.")])])
    pert = _trace(_storm_steps(3))
    ctx = from_adapter_trace(pert, initial_state=INIT, baseline_state=_final_state(base))
    findings = retry_amplification.detect(ctx)
    assert len(findings) == 1 and "baseline made none" in findings[0].summary


def test_drift_injected_call_and_initial_history_are_excluded_on_both_sides():
    """The chaos pattern seeds an AI call + error result in the initial
    state; the user's own history may hold earlier AI turns. Neither is
    work the GRAPH did, on either side."""
    init = {"messages": [
        _human("Earlier question"),
        _ai("Earlier answer"),
        _human("What's the weather in Paris?"),
        _ai(calls=[("get_weather", {"city": "Paris"}, "drift_call_err")]),
        _tool("Error: connection reset", "drift_call_err", "error"),
    ]}
    base_init = {"messages": init["messages"][:3]}
    base = _trace(_healthy_steps(1), base_init)
    pert = _trace(_healthy_steps(1), init)
    ctx = from_adapter_trace(pert, initial_state=init, baseline_state=_final_state(base))
    assert retry_amplification.detect(ctx) == []


def test_custom_thresholds():
    base = _trace(_healthy_steps(1))
    pert = _trace(_healthy_steps(2))
    ctx = from_adapter_trace(pert, initial_state=INIT, baseline_state=_final_state(base))
    assert retry_amplification.detect(ctx, min_ratio=1.5, min_extra=1)


def test_text_variant_is_silent_by_design():
    assert retry_amplification.detect_from_text("[tool] Error\n[agent] retrying\n[agent] retrying") == []


def _sub(node, msgs):
    """A subgraph step: root state does NOT grow (the parent commits later)."""
    return (node, msgs, True)


def _supervisor_trace(n_inner_calls, initial=INIT):
    """langgraph-supervisor shape with output_mode="last_message": the
    sub-agent's tool calls happen in math_expert/* steps and never reach the
    root messages — only its last message does. Counting the final state
    would miss every one of them."""
    running = list(initial["messages"])
    out = []
    step = 0

    def root(node, msgs):
        nonlocal running, step
        step += 1
        running = running + msgs
        out.append({"step": step, "node": node, "update": {"messages": msgs},
                    "state_after": {"messages": list(running)}})

    def inner(node, msgs):
        nonlocal step
        step += 1
        out.append({"step": step, "node": node, "namespace": [node.split("/")[0]],
                    "update": {"messages": msgs}, "state_after": {"messages": list(running)}})

    root("supervisor", [_ai(calls=[("transfer_to_math_expert", {}, "h1")]), _tool("ok", "h1")])
    for k in range(n_inner_calls):
        inner("math_expert/agent", [_ai(calls=[("multiply", {"a": 7, "b": 8, "try": k}, f"m{k}")])])
        inner("math_expert/tools", [_tool("Error: 503" if k < n_inner_calls - 1 else "56", f"m{k}")])
    last = _ai("7 times 8 is 56.")
    inner("math_expert/agent", [last])
    root("math_expert", [last, _ai(calls=[("transfer_back_to_supervisor", {}, "h2")]), _tool("ok", "h2")])
    root("supervisor", [_ai("The answer is 56.")])
    return out


def test_sub_agent_work_hidden_from_root_state_is_counted_when_baseline_trace_is_known():
    base = _supervisor_trace(1)   # 1 multiply + 2 handoffs = 3 calls
    pert = _supervisor_trace(5)   # 5 multiplies + 2 handoffs = 7 calls
    # Final states are IDENTICAL in shape (last_message mode) — the fallback
    # path cannot see the storm…
    ctx_state_only = from_adapter_trace(pert, initial_state=INIT, baseline_state=_final_state(base))
    assert retry_amplification.detect(ctx_state_only) == []
    # …the trace-aware path can.
    ctx = from_adapter_trace(
        pert, initial_state=INIT, baseline_state=_final_state(base),
        baseline_trace=base, baseline_initial_state=INIT,
    )
    findings = retry_amplification.detect(ctx)
    assert len(findings) == 1, findings
    assert "7 tool calls vs 3 at baseline (2.3×)" in findings[0].summary
    assert findings[0].agents_involved[0] == "math_expert"


def test_adapter_passes_the_baseline_trace_to_perturbations():
    """End to end: the adapter supplies baseline_trace for every perturbation,
    so a graph that storms only under chaos is caught."""
    from drift.adapters.langgraph import drift_test

    class _StormOnPerturbation:
        """Streams a 1-call run normally; 5 retries when the query was touched."""

        def stream(self, state, *args, **kwargs):
            touched = state.get("query") != "What's the weather in Paris?"
            msgs = list(state.get("messages") or [])
            n = 5 if touched else 1
            for k in range(n):
                upd = {"messages": [_ai(calls=[("get_weather", {"try": k}, f"c{k}")])]}
                msgs += upd["messages"]
                yield {"agent": upd}
                upd = {"messages": [_tool("Error: 503" if k < n - 1 else "18°C", f"c{k}")]}
                msgs += upd["messages"]
                yield {"tools": upd}
            upd = {"messages": [_ai("18°C in Paris.")]}
            yield {"agent": upd}

        def invoke(self, state, *args, **kwargs):
            out = dict(state)
            msgs = list(state.get("messages") or [])
            for chunk in self.stream(state):
                for upd in chunk.values():
                    msgs += upd["messages"]
            out["messages"] = msgs
            return out

    res = drift_test(
        graph=_StormOnPerturbation(),
        initial_state={"query": "What's the weather in Paris?", "messages": [_human("What's the weather in Paris?")]},
        intensity="exhaustive", seed=1, divergence_mode="off",
    )
    touched = [p for p in res.perturbations if p.perturbed_field == "query" and not p.crashed]
    assert touched, [p.event_name for p in res.perturbations]
    for p in touched:
        types = [f["failure_type"] for f in p.coordination_findings]
        assert "retry_amplification" in types, (p.event_name, types)
    assert not any(f["failure_type"] == "retry_amplification" for f in res.baseline.coordination_findings)
