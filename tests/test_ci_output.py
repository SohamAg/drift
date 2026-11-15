"""CI renderings: JUnit XML (and the CLI flags that write them)."""
from __future__ import annotations

import xml.etree.ElementTree as ET

from drift import gate
from drift.ci_output import to_junit


def _shaped(perts, *, base_findings=(), base_crashed=False):
    return {
        "graph_name": "app.build",
        "intensity": "moderate",
        "seed": 7,
        "chaos_timing": "initial",
        "baseline": {
            "crashed": base_crashed, "error_type": "RuntimeError" if base_crashed else None,
            "error": "boom" if base_crashed else None, "duration_s": 0.5,
            "coordination_findings": [{"failure_type": t, "summary": "s"} for t in base_findings],
            "judge_findings": [],
        },
        "perturbations": perts,
    }


def _p(name, **kw):
    d = {"event_name": name, "crashed": False, "diverged": False, "coordination_findings": [],
         "judge_findings": [], "duration_s": 0.1}
    d.update(kw)
    return d


def _parse(xml: str):
    root = ET.fromstring(xml)
    suite = root.find("testsuite")
    cases = {c.get("name"): c for c in suite.findall("testcase")}
    return suite, cases


def test_one_testcase_per_perturbation_plus_baseline():
    shaped = _shaped([_p("a"), _p("b", diverged=True, divergence_summary="reply changed")])
    rep = gate.evaluate(shaped, ["crash"])
    suite, cases = _parse(to_junit(shaped, rep))
    assert set(cases) == {"baseline (unperturbed)", "a", "b"}
    assert suite.get("tests") == "3" and suite.get("failures") == "0"
    # the divergence is recorded but is not a failure: no rule asked about it
    assert cases["b"].find("failure") is None
    assert "reply changed" in cases["b"].find("system-out").text


def test_failures_match_the_gate_exactly():
    shaped = _shaped([_p("crashy", crashed=True, error_type="KeyError", error="x"), _p("fine")])
    rep = gate.evaluate(shaped, ["crash"])
    suite, cases = _parse(to_junit(shaped, rep))
    assert suite.get("failures") == "1"
    f = cases["crashy"].find("failure")
    assert f is not None and "[crash]" in f.get("message")
    assert cases["fine"].find("failure") is None
    props = {p.get("name"): p.get("value") for p in suite.find("properties")}
    assert props["drift.gate"] == "fail" and props["drift.rules"] == "crash"


def test_baseline_findings_fail_the_baseline_case():
    shaped = _shaped([_p("a")], base_findings=["silent_tool_failure"])
    rep = gate.evaluate(shaped, ["finding"])
    _, cases = _parse(to_junit(shaped, rep))
    assert cases["baseline (unperturbed)"].find("failure") is not None
    assert cases["a"].find("failure") is None


def test_run_level_violations_get_their_own_case():
    shaped = _shaped([])
    rep = gate.evaluate(shaped, ["crash"])
    suite, cases = _parse(to_junit(shaped, rep))
    assert "gate: no_perturbations" in cases
    assert suite.get("failures") == "1"


def test_xml_escapes_hostile_text():
    shaped = _shaped([_p('<inject "x" & y>', crashed=True, error_type="E", error="</failure><x>")])
    rep = gate.evaluate(shaped, ["crash"])
    _, cases = _parse(to_junit(shaped, rep))  # parses → escaped correctly
    assert '<inject "x" & y>' in cases


def test_cli_gate_writes_junit(tmp_path):
    import json

    from drift.cli import main

    res = tmp_path / "r.json"
    res.write_text(json.dumps(_shaped([_p("c", crashed=True, error_type="E", error="e")])), encoding="utf-8")
    out = tmp_path / "reports" / "drift.xml"
    assert main(["gate", str(res), "--fail-on", "crash", "--junit", str(out)]) == 1
    suite, cases = _parse(out.read_text(encoding="utf-8"))
    assert suite.get("failures") == "1" and cases["c"].find("failure") is not None
