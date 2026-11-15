"""Suite-wide fixtures."""
import pytest


@pytest.fixture(autouse=True)
def _no_github_step_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    # drift's own CI runs this suite inside GitHub Actions, where
    # GITHUB_STEP_SUMMARY is set: CLI tests would append their gate
    # summaries to the real job page. Tests that want it set it themselves.
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
