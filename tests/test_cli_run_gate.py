"""`drift run` and `drift gate` — the CI path."""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from drift import gate
from drift.cli import main

pytest.importorskip("langgraph")

GRAPH_SRC = textwrap.dedent('''
    from typing import TypedDict
    from langgraph.graph import END, START, StateGraph

    class S(TypedDict):
        query: str
        attempts: int
        approved: bool
        reply: str

    def answer(state):
        if state["approved"]:
            raise RuntimeError("approved before review")  # flip_bool[approved] trips this
        return {"reply": f"answer to {state['query']}"}

    def build(model=None):
        sg = StateGraph(S)
        sg.add_node("answer", answer)
        sg.add_edge(START, "answer")
        sg.add_edge("answer", END)
        return sg.compile()

    def initial_state(query="hello"):
        return {"query": query, "attempts": 0, "approved": False, "reply": ""}
''')


@pytest.fixture
def graph_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    name = "drift_cli_test_graph"
    (tmp_path / f"{name}.py").write_text(GRAPH_SRC, encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # `drift run` adds the CWD to sys.path
    sys.modules.pop(name, None)
    yield name
    sys.modules.pop(name, None)
    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))


def test_run_prints_verdicts_and_writes_ui_shaped_json(graph_module: str, tmp_path: Path, capsys) -> None:
    out = tmp_path / "out" / "res.json"
    code = main(["run", graph_module, "--preset", "quick", "--json", str(out), "--seed", "3"])
    assert code == 0
    text = capsys.readouterr().out
    assert "drift × graph" in text and "crashed" in text
    data = json.loads(out.read_text(encoding="utf-8"))
    # the web UI's Results tab reads exactly these keys
    for key in ("graph_name", "baseline", "perturbations", "n_crashed", "n_diverged", "summary_lines",
                "perturbation_rollouts", "intensity", "seed"):
        assert key in data, key
    assert data["graph_name"] == f"{graph_module}.build"
    assert data["graph_agents"] == ["answer"]
    assert data["perturbations"], "quick preset should schedule perturbations on a 4-field state"


def test_run_knobs_override_preset(graph_module: str, tmp_path: Path) -> None:
    out = tmp_path / "res.json"
    code = main(["run", graph_module, "--rollouts", "2", "--max-perturbations", "2", "--json", str(out), "-q"])
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["perturbation_rollouts"] == 2
    assert len(data["perturbations"]) <= 2
    assert all(p["n_rollouts"] == 2 for p in data["perturbations"])


def test_run_fail_on_crash_flips_exit_code(graph_module: str, capsys) -> None:
    # exhaustive chaos always includes flip_bool[approved] → the graph raises.
    code = main(["run", graph_module, "--intensity", "exhaustive", "--fail-on", "crash", "-q"])
    out = capsys.readouterr().out
    assert code == 1, out
    assert "gate: FAIL" in out and "[crash]" in out


def test_run_passes_when_rule_not_triggered(graph_module: str, capsys) -> None:
    code = main(["run", graph_module, "--intensity", "exhaustive", "--fail-on", "finding:injected_instruction_followed"])
    assert code == 0
    assert "gate: PASS" in capsys.readouterr().out


def test_run_rejects_unknown_rule(graph_module: str) -> None:
    with pytest.raises(SystemExit) as ei:
        main(["run", graph_module, "--fail-on", "explode"])
    assert "unknown gate rule" in str(ei.value)


def test_run_requires_a_module_or_target(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["run"])


def test_run_bad_module_is_a_clean_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as ei:
        main(["run", "no_such_module_xyz"])
    assert "could not import" in str(ei.value)


def test_gate_on_saved_json(graph_module: str, tmp_path: Path, capsys) -> None:
    out = tmp_path / "res.json"
    assert main(["run", graph_module, "--intensity", "exhaustive", "--json", str(out), "-q"]) == 0
    assert main(["gate", str(out), "--fail-on", "crash"]) == 1
    assert "FAIL" in capsys.readouterr().out
    assert main(["gate", str(out), "--fail-on", "max_crash_rate=1.0,finding:tool_call_loop"]) == 0
    assert "PASS" in capsys.readouterr().out


def test_gate_rejects_non_result_files(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        main(["gate", str(p), "--fail-on", "crash"])
    assert "not a drift result" in str(ei.value)


# ---- gate module ------------------------------------------------------------------------


def _shaped(perts, baseline_findings=()):
    return {
        "baseline": {"coordination_findings": [{"failure_type": t} for t in baseline_findings], "judge_findings": []},
        "perturbations": perts,
        "n_judge_findings": 0,
    }


def _p(name, crashed=False, diverged=False, findings=(), crash_class="", intermittent=False):
    return {
        "event_name": name, "crashed": crashed, "diverged": diverged, "crash_class": crash_class,
        "coordination_findings": [{"failure_type": t} for t in findings], "judge_findings": [],
        "is_intermittent": intermittent, "n_rollouts": 3 if intermittent else 1,
    }


def test_parse_rules_splits_and_validates():
    assert gate.parse_rules(["crash, diverge", "finding:x"]) == ["crash", "diverge", "finding:x"]
    assert gate.parse_rules(None) == []
    with pytest.raises(ValueError):
        gate.parse_rules(["max_crash_rate=2"])
    with pytest.raises(ValueError):
        gate.parse_rules(["max_crash_rate=abc"])
    with pytest.raises(ValueError):
        gate.parse_rules(["finding:"])


def test_evaluate_each_rule():
    shaped = _shaped([
        _p("a", crashed=True, crash_class="recursion_limit"),
        _p("b", diverged=True, findings=["tool_call_loop"]),
        _p("c", intermittent=True),
        _p("d"),
    ], baseline_findings=["verifier_always_approves"])
    r = gate.evaluate(shaped, ["crash", "recursion_limit", "diverge", "intermittent", "finding",
                               "finding:tool_call_loop", "finding:silent_tool_failure", "baseline_finding",
                               "judge", "max_crash_rate=0.5", "max_diverge_rate=0.1"])
    fired = {v.rule for v in r.violations}
    assert fired == {"crash", "recursion_limit", "diverge", "intermittent", "finding",
                     "finding:tool_call_loop", "baseline_finding", "max_diverge_rate=0.1"}
    assert not r.passed
    by = {v.rule: v for v in r.violations}
    assert by["crash"].perturbations == ["a"]
    assert by["finding:tool_call_loop"].perturbations == ["b"]
    assert "unperturbed" in by["baseline_finding"].summary
    assert "25%" in by["max_diverge_rate=0.1"].summary


def test_evaluate_passes_cleanly_and_reports():
    r = gate.evaluate(_shaped([_p("a"), _p("b")]), ["crash", "diverge", "finding"])
    assert r.passed and r.lines()[0].startswith("gate: PASS")
    assert gate.evaluate(_shaped([]), []).lines()[0].startswith("gate: no rules")


def test_presets_match_the_web_ui():
    """web/app.js ADAPTER_PRESETS and drift.gate.PRESETS must agree."""
    import re

    js = (Path(__file__).resolve().parents[1] / "web" / "app.js").read_text(encoding="utf-8")
    block = js[js.index("const ADAPTER_PRESETS = {"): js.index("};", js.index("const ADAPTER_PRESETS = {"))]
    for name, knobs in gate.PRESETS.items():
        m = re.search(name + r":\s*\{(.*?)\}", block, re.S)
        assert m, name
        body = m.group(1)
        for k, v in knobs.items():
            want = f"'{v}'" if isinstance(v, str) else str(v)
            assert re.search(rf"\b{k}:\s*{re.escape(want)}", body), (name, k, v, body)


def test_server_gate_endpoint_matches_the_cli_evaluator():
    from fastapi.testclient import TestClient

    from drift.server import create_app

    client = TestClient(create_app())
    run = client.post("/api/adapter-demo", json={
        "graph_name": "ticket_triage", "intensity": "exhaustive", "judge": "off", "divergence_mode": "exact",
    }).json()
    r = client.post("/api/gate", json={"result": run, "rules": ["crash", "finding:tool_call_loop"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rules"] == ["crash", "finding:tool_call_loop"]
    assert body["n_perturbations"] == len(run["perturbations"])
    expect = gate.evaluate(run, ["crash", "finding:tool_call_loop"])
    assert body["passed"] == expect.passed
    assert [v["rule"] for v in body["violations"]] == [v.rule for v in expect.violations]
    bad = client.post("/api/gate", json={"result": run, "rules": ["nonsense"]})
    assert bad.status_code == 400 and "unknown gate rule" in bad.json()["detail"]


def test_targets_lists_the_registry(capsys) -> None:
    assert main(["targets"]) == 0
    out = capsys.readouterr().out
    assert "supervisor_basic" in out and "deepagents_tooldispatch" in out
    assert "drift run --target" in out


def test_targets_json(capsys) -> None:
    import json as _json

    assert main(["targets", "--json"]) == 0
    rows = _json.loads(capsys.readouterr().out)
    assert {r["name"] for r in rows} >= {"supervisor_basic", "swarm_peer"}
    assert all("importable" in r and "topology" in r for r in rows)


def test_gate_never_passes_vacuously():
    r = gate.evaluate(_shaped([]), ["crash"])
    assert not r.passed and r.violations[0].rule == "no_perturbations"
    # no rules → nothing to fail, as before
    assert gate.evaluate(_shaped([]), []).passed


def test_gate_fails_when_the_baseline_itself_crashed():
    shaped = _shaped([_p("a")])
    shaped["baseline"]["crashed"] = True
    shaped["baseline"]["error_type"] = "KeyError"
    shaped["baseline"]["error"] = "'messages'"
    r = gate.evaluate(shaped, ["finding"])
    assert [v.rule for v in r.violations] == ["baseline_crashed"]
    assert "KeyError" in r.violations[0].summary
