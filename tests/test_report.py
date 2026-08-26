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
