"""Gate rules that compare against a previous result: new_finding, new_crash, regression."""
from __future__ import annotations

import json

import pytest

from drift import gate
from drift.regression import snapshot_from_shaped


def _p(name, *, crashed=False, diverged=False, findings=(), duration=0.1):
    return {"event_name": name, "crashed": crashed, "diverged": diverged, "duration_s": duration,
            "coordination_findings": [{"failure_type": t, "summary": t} for t in findings],
            "judge_findings": []}


def _run(perts, *, base_findings=(), base_crashed=False):
    return {"graph_name": "app.build", "intensity": "moderate", "seed": 7,
            "baseline": {"crashed": base_crashed, "duration_s": 0.2,
                         "coordination_findings": [{"failure_type": t} for t in base_findings]},
            "perturbations": perts}


PREV = _run([_p("a"), _p("b", findings=["silent_tool_failure"]), _p("c", crashed=True)])


def test_rules_parse():
    assert gate.parse_rules(["new_finding,new_crash", "regression"]) == ["new_finding", "new_crash", "regression"]


def test_without_previous_the_rules_are_skipped_with_a_note():
    cur = _run([_p("a", crashed=True)])
    rep = gate.evaluate(cur, ["new_crash", "new_finding"])
    assert rep.passed
    assert rep.notes and "no previous result" in rep.notes[0]
    assert any("note" in line for line in rep.lines())


def test_new_finding_flags_only_types_absent_before():
    cur = _run([_p("a", findings=["silent_tool_failure", "tool_call_loop"]), _p("b")])
    rep = gate.evaluate(cur, ["new_finding"], previous=PREV)
    assert [v.rule for v in rep.violations] == ["new_finding"]
    assert "tool_call_loop" in rep.violations[0].summary
    assert "silent_tool_failure" not in rep.violations[0].summary
    assert rep.violations[0].perturbations == ["a"]
    # an existing finding type on its own is not new
    same = _run([_p("b", findings=["silent_tool_failure"])])
    assert gate.evaluate(same, ["new_finding"], previous=PREV).passed


def test_new_crash_matches_by_event_name_and_ignores_known_crashes():
    cur = _run([_p("a", crashed=True), _p("c", crashed=True), _p("zz_new", crashed=True)])
    rep = gate.evaluate(cur, ["new_crash"], previous=PREV)
    v = rep.violations[0]
    # c crashed before (known), zz_new did not exist before (not a regression of a known input)
    assert v.perturbations == ["a"]
    assert not v.baseline


def test_new_crash_on_the_baseline():
    rep = gate.evaluate(_run([_p("a")], base_crashed=True), ["new_crash"], previous=PREV)
    v = next(v for v in rep.violations if v.rule == "new_crash")
    assert v.baseline and "baseline now crashes" in v.summary


def test_regression_uses_the_regression_module_thresholds():
    cur = _run([_p("a", crashed=True), _p("b", crashed=True, findings=["silent_tool_failure"]), _p("c", crashed=True)])
    rep = gate.evaluate(cur, ["regression"], previous=PREV)
    assert rep.violations and "crash rate up" in rep.violations[0].summary
    assert gate.evaluate(PREV, ["regression"], previous=PREV).passed


def test_snapshot_from_shaped_recounts():
    s = snapshot_from_shaped(PREV)
    assert (s.n_crashed, s.n_diverged, s.n_unchanged, s.n_perturbations) == (1, 0, 2, 3)
    assert s.coordination_finding_types == {"silent_tool_failure": 1}


def test_cli_against_missing_file_is_a_note_not_an_error(tmp_path, capsys):
    from drift.cli import main

    cur = tmp_path / "cur.json"
    cur.write_text(json.dumps(_run([_p("a")])), encoding="utf-8")
    assert main(["gate", str(cur), "--fail-on", "new_crash", "--against", str(tmp_path / "nope.json")]) == 0
    assert "no previous result" in capsys.readouterr().out


def test_cli_against_flags_the_regression(tmp_path, capsys):
    from drift.cli import main

    prev, cur = tmp_path / "prev.json", tmp_path / "cur.json"
    prev.write_text(json.dumps(PREV), encoding="utf-8")
    cur.write_text(json.dumps(_run([_p("a", crashed=True)])), encoding="utf-8")
    assert main(["gate", str(cur), "--fail-on", "new_crash", "--against", str(prev)]) == 1
    assert "[new_crash]" in capsys.readouterr().out


def test_cli_against_rejects_non_results(tmp_path):
    from drift.cli import main

    cur, bad = tmp_path / "cur.json", tmp_path / "bad.json"
    cur.write_text(json.dumps(_run([_p("a")])), encoding="utf-8")
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        main(["gate", str(cur), "--fail-on", "new_crash", "--against", str(bad)])
    assert "not a drift result" in str(ei.value)


def test_server_gate_accepts_previous():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from drift.server import create_app

    client = TestClient(create_app())
    cur = _run([_p("a", crashed=True)])
    r = client.post("/api/gate", json={"result": cur, "rules": ["new_crash"], "previous": PREV})
    assert r.status_code == 200 and r.json()["passed"] is False
    r = client.post("/api/gate", json={"result": cur, "rules": ["new_crash"]})
    assert r.json()["passed"] is True and r.json()["notes"]


def test_new_event_crash_catches_crashes_new_crash_cannot_match():
    cur = _run([_p("a", crashed=True), _p("c", crashed=True), _p("zz_new", crashed=True), _p("yy_new")])
    rep = gate.evaluate(cur, ["new_event_crash"], previous=PREV)
    (v,) = rep.violations
    assert v.rule == "new_event_crash" and v.perturbations == ["zz_new"]  # a and c existed before
    assert gate.evaluate(_run([_p("zz_new")]), ["new_event_crash"], previous=PREV).passed


def test_new_crash_alone_notes_the_crashes_it_could_not_judge():
    cur = _run([_p("zz_new", crashed=True)])
    rep = gate.evaluate(cur, ["new_crash"], previous=PREV)
    assert rep.passed
    assert any("new_event_crash" in n and "zz_new" in n for n in rep.notes)
    # with both rules there is nothing left unjudged, so no note
    both = gate.evaluate(cur, ["new_crash", "new_event_crash"], previous=PREV)
    assert not both.passed and not any("could not judge" in n for n in both.notes)


def test_new_event_crash_is_skipped_without_previous():
    rep = gate.evaluate(_run([_p("zz", crashed=True)]), ["new_event_crash"])
    assert rep.passed and "new_event_crash" in rep.notes[0]
