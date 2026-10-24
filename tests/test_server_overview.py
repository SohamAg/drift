"""Server-level test for /api/overview, the portfolio-rollup dashboard.

Exercises the real endpoint chain (adapter-demo -> save-run -> overview)
rather than hand-crafting result JSON — proves the rollup correctly reads
what the app itself actually writes, not just a guessed shape. No LLM
needed: the bundled ticket_triage graph + judge="off" is deterministic.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import drift.server as server_module
from drift.server import create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Redirect results/ to a tmp dir so this test never reads or writes
    # into the repo's real results/ tree.
    monkeypatch.setattr(server_module, "RESULTS_DIR", tmp_path / "results")
    return TestClient(create_app())


def _run_adapter(client: TestClient, max_perturbations: int = 3) -> dict:
    resp = client.post(
        "/api/adapter-demo",
        json={
            "graph_name": "ticket_triage",
            "intensity": "light",
            "seed": 7,
            "max_perturbations": max_perturbations,
            "judge": "off",
            "divergence_mode": "exact",
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _save(client: TestClient, data: dict, group: str, name: str) -> None:
    resp = client.post(
        "/api/save-run",
        json={"data": data, "group": group, "name": name},
    )
    assert resp.status_code == 200, resp.text


def test_overview_empty_when_no_results(client: TestClient) -> None:
    data = client.get("/api/overview").json()
    assert data["totals"]["runs"] == 0
    assert data["totals"]["systems"] == 0
    assert data["systems"] == []
    assert data["findings"] == []


def test_overview_rolls_up_a_saved_adapter_run(client: TestClient) -> None:
    run = _run_adapter(client)
    _save(client, run, group="my_system", name="run1.json")

    data = client.get("/api/overview").json()
    assert data["totals"]["systems"] == 1
    # report.load_run treats one saved adapter-demo response as a single
    # "question" — n_runs = 1 baseline + len(perturbations).
    assert data["totals"]["runs"] == 1 + len(run["perturbations"])
    assert len(data["systems"]) == 1
    sysrow = data["systems"][0]
    assert sysrow["group"] == "my_system"
    assert sysrow["n_files"] == 1


def test_overview_sums_multiple_files_in_the_same_group(client: TestClient) -> None:
    run1 = _run_adapter(client)
    run2 = _run_adapter(client)
    _save(client, run1, group="same_system", name="a.json")
    _save(client, run2, group="same_system", name="b.json")

    data = client.get("/api/overview").json()
    assert data["totals"]["systems"] == 1
    sysrow = data["systems"][0]
    assert sysrow["n_files"] == 2
    assert sysrow["n_runs"] == data["totals"]["runs"]


def test_overview_separates_distinct_groups(client: TestClient) -> None:
    run1 = _run_adapter(client)
    run2 = _run_adapter(client)
    _save(client, run1, group="system_a", name="a.json")
    _save(client, run2, group="system_b", name="b.json")

    data = client.get("/api/overview").json()
    assert data["totals"]["systems"] == 2
    groups = {s["group"] for s in data["systems"]}
    assert groups == {"system_a", "system_b"}


def test_overview_skips_non_adapter_shaped_json(
    client: TestClient, tmp_path: Path
) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "not_a_run.json").write_text('{"hello": "world"}', encoding="utf-8")

    data = client.get("/api/overview").json()
    # Doesn't crash, doesn't count the file — report.load_run's
    # NotADriftRun is swallowed, not surfaced as an error to the caller.
    assert data["totals"]["runs"] == 0
    assert data["systems"] == []


def test_overview_finding_source_path_is_openable_in_results_viewer(
    client: TestClient,
) -> None:
    run = _run_adapter(client, max_perturbations=10)
    _save(client, run, group="findings_check", name="r.json")

    overview = client.get("/api/overview").json()
    for f in overview["findings"]:
        resp = client.get(f"/api/results/{f['source_path']}")
        assert resp.status_code == 200
