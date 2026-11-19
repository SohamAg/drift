"""Smoke tests for the evidence-pack report generator (`drift report`)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from drift.report import load_run, render_report, write_report

MULTI_SHAPE = {
    "config": {"model": "gpt-4o-mini", "intensity": "aggressive", "use_judge": True},
    "elapsed_seconds": 12.5,
    "per_question": [
        {
            "question": "Do the thing & verify <it>.",
            "category": "multi_3step",
            "baseline_crashed": False,
            "baseline_trace_steps": 3,
            "baseline_unique_agents": ["supervisor", "worker"],
            "baseline_judge": [
                {"type": "llm:coordination_contradiction",
                 "summary": "Supervisor re-issued an answered request."}
            ],
            "baseline_coord": [],
            "n_perturbations": 2,
            "perturbations": [
                {"event": "AutoChaos.clear_list[messages]", "field": "messages",
                 "pattern": "clear_list", "tags": ["DIVERGE"],
                 "divergence": "messages: a -> b", "judge_findings": [],
                 "coordination_findings": []},
                {"event": "AutoChaos.duplicate_list_entry[messages]",
                 "field": "messages", "pattern": "duplicate_list_entry",
                 "tags": ["UNCHANGED"], "divergence": "",
                 "judge_findings": [], "coordination_findings": []},
            ],
        }
    ],
}

FLAT_SHAPE = {
    "question": "Research X.",
    "model": "gpt-4o-mini",
    "baseline_crashed": False,
    "baseline_trace_steps": 8,
    "baseline_unique_agents": ["model", "tools"],
    "baseline_judge": [],
    "baseline_coord": [],
    "_elapsed_s": 55.0,
    "perturbations": [
        {"event": "AutoChaos.clear_list[messages]", "crashed": False,
         "diverged": True, "judge_findings": [],
         "coord_findings": [{"type": "stale_state_reference", "summary": "s"}]},
    ],
}

FORK_PROOF = {
    "mas": "demo",
    "finding_type": "hallucinated_reference",
    "fork_step": 1,
    "edits": {"ticket_id": "TICKET-42"},
    "fork_closed": True,
    "top_closed": True,
    "duration_s": 0.01,
}


@pytest.fixture()
def multi_path(tmp_path: Path) -> Path:
    p = tmp_path / "multi.json"
    p.write_text(json.dumps(MULTI_SHAPE), encoding="utf-8")
    return p


@pytest.fixture()
def flat_path(tmp_path: Path) -> Path:
    p = tmp_path / "flat.json"
    p.write_text(json.dumps(FLAT_SHAPE), encoding="utf-8")
    return p


def test_load_run_multi_shape(multi_path: Path) -> None:
    run = load_run(multi_path)
    assert run["summary"]["n_scenarios"] == 1
    assert run["summary"]["n_perturbations"] == 2
    assert run["summary"]["n_diverged"] == 1
    assert run["summary"]["n_unchanged"] == 1
    assert run["summary"]["n_judge_findings"] == 1
    assert run["summary"]["agents"] == ["supervisor", "worker"]
    assert len(run["source"]["sha256"]) == 64


def test_load_run_flat_shape(flat_path: Path) -> None:
    run = load_run(flat_path)
    assert run["summary"]["n_scenarios"] == 1
    assert run["summary"]["n_diverged"] == 1
    assert run["summary"]["n_coord_findings"] == 1
    assert run["config"]["model"] == "gpt-4o-mini"


def test_render_escapes_html(multi_path: Path) -> None:
    html_text = render_report(load_run(multi_path))
    assert "&amp;" in html_text and "&lt;it&gt;" in html_text
    assert "<it>" not in html_text


def test_render_includes_fork_section(multi_path: Path) -> None:
    html_text = render_report(load_run(multi_path), fork=FORK_PROOF)
    assert "Counterfactual proof" in html_text
    assert "CLOSED" in html_text
    assert "TICKET-42" in html_text


def test_write_report_default_out(multi_path: Path) -> None:
    out = write_report(multi_path)
    assert out.name == "multi.report.html"
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "Reliability evidence" in text


def test_no_findings_renders_clean(flat_path: Path, tmp_path: Path) -> None:
    data = json.loads(flat_path.read_text(encoding="utf-8"))
    data["perturbations"] = []
    p = tmp_path / "clean.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    html_text = render_report(load_run(p))
    assert "No coordination findings" in html_text


# --------------------------------------------------------------------------
# Non-run rejection — keeps unrelated JSON in results/ out of aggregate views
# --------------------------------------------------------------------------

def test_rejects_non_drift_json(tmp_path: Path) -> None:
    """Judge-eval dumps and other stray JSON must not parse as runs."""
    from drift.report import NotADriftRun

    p = tmp_path / "judge_eval.json"
    p.write_text(json.dumps({
        "path": "x", "example": "y", "n_actions": 3,
        "failures_by_type": {}, "judge_calls": 2,
    }), encoding="utf-8")
    with pytest.raises(NotADriftRun):
        load_run(p)


def test_rejects_non_object_json(tmp_path: Path) -> None:
    from drift.report import NotADriftRun

    p = tmp_path / "list.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(NotADriftRun):
        load_run(p)


def test_accepts_flat_run_with_only_baseline_marker(tmp_path: Path) -> None:
    """A run with no perturbations is still a run if it has baseline markers."""
    p = tmp_path / "baseline_only.json"
    p.write_text(json.dumps({
        "question": "q", "baseline_crashed": False,
        "baseline_trace_steps": 4, "baseline_judge": [], "baseline_coord": [],
    }), encoding="utf-8")
    run = load_run(p)
    assert run["summary"]["n_runs"] == 1
    assert run["summary"]["n_perturbations"] == 0


def test_web_and_cli_shaped_result_renders_event_names_and_rollouts(tmp_path: Path) -> None:
    """`drift run --json` / the web UI write the shaped AdapterResult dict.
    The report used to render blank event cells for it."""
    import json

    from drift.adapters.langgraph import drift_test
    from drift.adapters.serialize import shape_adapter_result
    from drift.report import load_run, render_report

    class _G:
        def invoke(self, state, *a, **k):
            out = dict(state)
            if out.get("approved"):
                raise RuntimeError("boom")
            out["reply"] = "ok"
            return out

    res = drift_test(
        graph=_G(), initial_state={"query": "hello", "approved": False, "reply": ""},
        intensity="exhaustive", seed=1, perturbation_rollouts=2,
    )
    shaped = shape_adapter_result(
        res, graph_meta={"name": "stub", "description": "", "agents": ["a"]}, seed=1,
        judge="off", judge_model=None, user_guidelines=[],
    )
    path = tmp_path / "cli.json"
    path.write_text(json.dumps(shaped), encoding="utf-8")
    run = load_run(path)
    q = run["questions"][0]
    assert q["question"] == "hello" and q["category"] == "stub" and q["agents"] == ["a"]
    assert run["summary"]["n_crashed"] == shaped["n_crashed"]
    events = [p["event"] for p in q["perturbations"]]
    assert events and all(events), events
    assert all(p["n_rollouts"] == 2 for p in q["perturbations"])
    html = render_report(run)
    assert "flip_bool[approved]" in html
    assert "2/2" in html  # outcome label rendered
    assert run["config"]["perturbation_rollouts"] == 2
    # graph invocations, not rows: baseline + 2 rollouts per perturbation
    assert run["summary"]["n_runs"] == shaped["n_graph_runs"] == 1 + 2 * len(shaped["perturbations"])


# ---- gate section -------------------------------------------------------------------------

SHAPED_CRASH = {
    "graph_name": "app.build", "intensity": "moderate", "seed": 7,
    "baseline": {"crashed": False, "coordination_findings": [], "judge_findings": [], "trace": []},
    "perturbations": [
        {"event_name": "AutoChaos.corrupt_string[priority]", "crashed": True, "diverged": False,
         "error_type": "KeyError", "error": "'x'", "coordination_findings": [], "judge_findings": [],
         "trace": []},
        {"event_name": "AutoChaos.flip_bool[escalated]", "crashed": False, "diverged": False,
         "coordination_findings": [], "judge_findings": [], "trace": []},
    ],
}


def test_report_shows_gate_fail_with_violations(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(json.dumps(SHAPED_CRASH), encoding="utf-8")
    out = write_report(p, fail_on=["crash,diverge"])
    text = out.read_text(encoding="utf-8")
    assert 'class="eyebrow">Gate<' in text and ">FAIL<" in text
    assert "1/2 perturbation(s) crashed" in text
    assert "AutoChaos.corrupt_string[priority]" in text


def test_report_gate_pass_and_note(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(json.dumps(SHAPED_CRASH), encoding="utf-8")
    text = write_report(p, fail_on=["diverge", "new_crash"]).read_text(encoding="utf-8")
    assert ">PASS<" in text and "no previous result" in text


def test_report_without_rules_has_no_gate_section(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(json.dumps(SHAPED_CRASH), encoding="utf-8")
    assert 'class="eyebrow">Gate<' not in write_report(p).read_text(encoding="utf-8")


def test_report_gate_refuses_multi_scenario_files(multi_path: Path) -> None:
    with pytest.raises(ValueError, match="single drift run"):
        write_report(multi_path, fail_on=["crash"])


def test_cli_report_fail_on(tmp_path: Path, capsys) -> None:
    from drift.cli import main

    p = tmp_path / "r.json"
    p.write_text(json.dumps(SHAPED_CRASH), encoding="utf-8")
    out = tmp_path / "r.html"
    assert main(["report", str(p), "--out", str(out), "--fail-on", "crash"]) == 0  # a report, not a gate
    assert ">FAIL<" in out.read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown gate rule"):
        main(["report", str(p), "--fail-on", "nope"])


# ---- guideline rules section + methodology timing ------------------------------------------

def _with_rules():
    d = json.loads(json.dumps(SHAPED_CRASH))
    d["guideline_rules"] = [
        {"id": "no-email", "forbid_tool": "send_*", "why": "only the coordinator emails"},
        {"id": "bounded", "max_steps": 20, "severity": "medium"},
        {"id": "broken", "state": "missing_field > 0"},
    ]
    d["perturbations"][1]["coordination_findings"] = [
        {"failure_type": "guideline:no-email", "summary": "[high] send_email called"},
    ]
    d["baseline"]["coordination_findings"] = [
        {"failure_type": "guideline:broken:error", "summary": "unknown name missing_field"},
    ]
    return d


def test_guideline_section_lists_violated_held_and_unevaluable_rules(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(json.dumps(_with_rules()), encoding="utf-8")
    text = write_report(p).read_text(encoding="utf-8")
    assert 'class="eyebrow">Guideline rules<' in text
    assert "violated on 1 of 3 run(s)" in text and "AutoChaos.flip_bool[escalated]" in text
    assert "held on all 3 run(s)" in text          # bounded: checked, never fired
    assert "could not evaluate" in text and "unknown name missing_field" in text
    assert "only the coordinator emails" in text and "send_*" in text


def test_no_guideline_section_without_rules(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(json.dumps(SHAPED_CRASH), encoding="utf-8")
    assert "Guideline rules<" not in write_report(p).read_text(encoding="utf-8")


def test_methodology_states_the_timing_actually_used(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(json.dumps(SHAPED_CRASH), encoding="utf-8")
    assert "applied to the initial state" in write_report(p).read_text(encoding="utf-8")
    d = json.loads(json.dumps(SHAPED_CRASH))
    for x in d["perturbations"]:
        x["chaos_timing"] = "mid_run"
    p.write_text(json.dumps(d), encoding="utf-8")
    text = write_report(p).read_text(encoding="utf-8")
    assert "applied mid-run" in text and "applied to the initial state" not in text


def test_report_labels_model_and_flow_layers(tmp_path: Path) -> None:
    d = json.loads(json.dumps(SHAPED_CRASH))
    d["perturbations"][0].update(fault_layer="flow", fault_fired=True, chaos_timing="mid_run", fork_step=0)
    d["perturbations"][1].update(fault_layer="model", fault_fired=False, chaos_timing="mid_run")
    p = tmp_path / "r.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    text = write_report(p).read_text(encoding="utf-8")
    assert ">control flow<" in text and "mid-run @ 0" in text
    assert "model · not applied" in text
    assert "taxonomy judge" not in text  # the old subtitle overstated the judge's role


def test_layer_table_counts_each_layer_and_keeps_unapplied_faults_apart(tmp_path: Path) -> None:
    from drift.report import layer_summary, load_run

    d = json.loads(json.dumps(SHAPED_CRASH))  # 1 state crash, 1 state unchanged
    d["perturbations"] += [
        {"event_name": "AutoChaos.tool_error[web_search]", "fault_layer": "tool", "fault_fired": True,
         "crashed": False, "diverged": True, "coordination_findings": [{"type": "silent_tool_failure"}],
         "judge_findings": [], "trace": []},
        {"event_name": "AutoChaos.skip_node[math]", "fault_layer": "flow", "fault_fired": False,
         "crashed": False, "diverged": False, "coordination_findings": [], "judge_findings": [], "trace": []},
    ]
    p = tmp_path / "r.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    rows = {r["layer"]: r for r in layer_summary(load_run(p)["questions"])}
    assert [*rows] == ["state", "tool", "flow"]
    assert rows["state"] == {"layer": "state", "n": 2, "not_applied": 0, "crashed": 1, "diverged": 0,
                                  "unchanged": 1, "with_findings": 0}
    assert rows["tool"]["diverged"] == 1 and rows["tool"]["with_findings"] == 1
    assert rows["flow"]["not_applied"] == 1 and rows["flow"]["unchanged"] == 0  # not counted as coping
    text = write_report(p).read_text(encoding="utf-8")
    assert "By fault layer" in text and "Control flow" in text and "+1 not applied" in text


def test_single_layer_run_has_no_layer_table(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(json.dumps(SHAPED_CRASH), encoding="utf-8")
    assert "By fault layer" not in write_report(p).read_text(encoding="utf-8")
