"""ungrounded_number against a curated corpus of real saved runs (ADR-0011).

tests/fixtures/ungrounded_corpus.json is built by
scripts/build_ungrounded_corpus.py from results/ (git-ignored). Every
positive was read by hand and is a real fabrication; every negative has a
final answer with specific numbers that the guards must clear. A guard
change that adds or loses a fire fails here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from drift.failures.library import ungrounded_number
from drift.failures.library.base import from_adapter_trace

CORPUS = json.loads((Path(__file__).parent / "fixtures" / "ungrounded_corpus.json").read_text(encoding="utf-8"))["runs"]


def _id(r):
    return f"{r['source'].rsplit('/', 1)[-1]}::{r['run']}"


@pytest.mark.parametrize("run", CORPUS, ids=_id)
def test_corpus_run_matches_its_hand_label(run):
    ctx = from_adapter_trace(run["trace"], initial_state=run["initial_state"])
    fired = bool(ungrounded_number.detect(ctx))
    assert fired == run["expect_fire"]


def test_corpus_covers_both_fabrications_and_guarded_negatives():
    positives = [r for r in CORPUS if r["expect_fire"]]
    assert len(positives) == 4
    assert {r["run"] for r in positives} >= {
        "AutoChaos.skip_node[research_expert]",
        "AutoChaos.handoff_misroute[transfer_to_research_expert]",
        "baseline",
    }
    assert len(CORPUS) - len(positives) >= 5
