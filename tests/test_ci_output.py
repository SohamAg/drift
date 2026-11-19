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


# ---- Markdown summary -------------------------------------------------------------------

from drift.ci_output import to_markdown, write_github_step_summary  # noqa: E402


def test_markdown_verdict_and_violation_table():
    shaped = _shaped([_p("crashy", crashed=True, error_type="E", error="e"), _p("fine")])
    md = to_markdown(shaped, gate.evaluate(shaped, ["crash"]))
    assert md.startswith("## drift · `app.build` — ❌ FAIL")
    assert "| `crash` | 1/2 perturbation(s) crashed |" in md
    # flagged perturbations are listed first and marked
    rows = [l for l in md.splitlines() if l.startswith("| ") and ("`crashy`" in l or "`fine`" in l)]
    assert rows[0].startswith("| ❌ | `crashy`")


def test_markdown_pass_and_no_rules():
    shaped = _shaped([_p("a")])
    assert "✅ PASS" in to_markdown(shaped, gate.evaluate(shaped, ["crash"]))
    assert "no gate rules" in to_markdown(shaped, gate.evaluate(shaped, []))


def test_markdown_escapes_pipes_in_cells():
    shaped = _shaped([_p("a|b", diverged=True)])
    md = to_markdown(shaped, gate.evaluate(shaped, ["diverge"]))
    assert r"a\|b" in md


def test_step_summary_only_when_env_set(tmp_path):
    assert write_github_step_summary("x", env={}) is None
    target = tmp_path / "summary.md"
    target.write_text("before\n", encoding="utf-8")
    assert write_github_step_summary("after\n", env={"GITHUB_STEP_SUMMARY": str(target)}) == target
    assert target.read_text(encoding="utf-8") == "before\nafter\n"  # appends, never truncates


def test_cli_gate_appends_to_github_step_summary(tmp_path, monkeypatch):
    import json

    from drift.cli import main

    res = tmp_path / "r.json"
    res.write_text(json.dumps(_shaped([_p("a")])), encoding="utf-8")
    step = tmp_path / "step.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step))
    assert main(["gate", str(res), "--fail-on", "crash", "--summary", str(tmp_path / "s.md")]) == 0
    assert "✅ PASS" in step.read_text(encoding="utf-8")
    assert (tmp_path / "s.md").read_text(encoding="utf-8") == step.read_text(encoding="utf-8")
    step.unlink()
    assert main(["gate", str(res), "--fail-on", "crash", "--no-step-summary"]) == 0
    assert not step.exists()


def test_unapplied_fault_is_skipped_not_passed_and_layers_group_by_classname():
    shaped = _shaped([
        _p("AutoChaos.flip_bool[x]"),
        _p("AutoChaos.tool_error[web]", fault_layer="tool", fault_fired=True),
        _p("AutoChaos.skip_node[math]", fault_layer="flow", fault_fired=False,
           event_summary="flow-layer fault skip_node on 'math' — not applied: the baseline never ran 'math'"),
    ])
    suite, cases = _parse(to_junit(shaped, gate.evaluate(shaped, ["crash"])))
    assert suite.get("skipped") == "1" and suite.get("failures") == "0"
    sk = cases["AutoChaos.skip_node[math]"].find("skipped")
    assert sk is not None and sk.get("message") == "flow fault not applied: the baseline never ran 'math'"
    assert cases["AutoChaos.tool_error[web]"].find("skipped") is None
    assert cases["AutoChaos.tool_error[web]"].get("classname").endswith(".tool")
    assert cases["AutoChaos.flip_bool[x]"].get("classname").endswith(".state")
    assert cases["baseline (unperturbed)"].get("classname") == "drift.app.build"  # no layer suffix


def test_markdown_says_not_applied_rather_than_unchanged():
    from drift.ci_output import to_markdown

    shaped = _shaped([_p("AutoChaos.skip_node[math]", fault_layer="flow", fault_fired=False), _p("b")])
    md = to_markdown(shaped, gate.evaluate(shaped, ["crash"]))
    row = next(line for line in md.splitlines() if "skip_node[math]" in line)
    assert "not applied" in row and "unchanged" not in row
