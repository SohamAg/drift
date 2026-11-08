"""docs/detectors/README.md is generated from the library and must list every detector."""
from __future__ import annotations

from pathlib import Path

from drift.failures.library import ALL_DETECTORS

DOCS = Path(__file__).resolve().parents[1] / "docs" / "detectors"


def test_every_detector_has_a_doc_page_and_an_index_row():
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    for mod in ALL_DETECTORS:
        assert (DOCS / f"{mod.NAME}.md").exists(), f"missing docs/detectors/{mod.NAME}.md"
        assert f"[`{mod.NAME}`]({mod.NAME}.md)" in index, f"{mod.NAME} missing from docs/detectors/README.md — run scripts/gen_detector_index.py"
    assert f"**{len(ALL_DETECTORS)} detectors.**" in index


def test_index_has_no_stale_rows():
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    import re

    listed = set(re.findall(r"\[`([a-z_]+)`\]\(\1\.md\)", index))
    assert listed == {m.NAME for m in ALL_DETECTORS}
