"""Tests for drift.cli's `serve` subcommand — specifically the
auto-open-dashboard-in-browser behavior added so running `drift serve`
lands the user straight on the dashboard instead of requiring them to
copy the printed URL manually.

threading.Timer and webbrowser.open are both faked: the real Timer
would make the test wait a second for no reason, and the real
webbrowser.open would try to launch an actual browser in CI.
"""
from __future__ import annotations

import argparse
import threading

import webbrowser

import drift.cli as cli_module
import drift.server as server_module


class _ImmediateTimer:
    """Stand-in for threading.Timer that fires synchronously on .start()."""

    def __init__(self, interval: float, function) -> None:
        self._function = function

    def start(self) -> None:
        self._function()


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(host="127.0.0.1", port=8765, reload=False, no_browser=False)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_serve_opens_browser_to_the_dashboard_url_by_default(monkeypatch) -> None:
    monkeypatch.setattr(server_module, "serve", lambda **kw: None)
    monkeypatch.setattr(threading, "Timer", _ImmediateTimer)
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))

    cli_module.cmd_serve(_make_args())

    assert opened == ["http://127.0.0.1:8765/"]


def test_serve_substitutes_localhost_when_host_is_a_wildcard_bind(monkeypatch) -> None:
    """0.0.0.0 is a valid bind address but not something a browser can open —
    the dashboard link should point at 127.0.0.1 instead."""
    monkeypatch.setattr(server_module, "serve", lambda **kw: None)
    monkeypatch.setattr(threading, "Timer", _ImmediateTimer)
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))

    cli_module.cmd_serve(_make_args(host="0.0.0.0", port=9000))

    assert opened == ["http://127.0.0.1:9000/"]


def test_serve_skips_opening_the_browser_when_no_browser_is_passed(monkeypatch) -> None:
    monkeypatch.setattr(server_module, "serve", lambda **kw: None)
    called = []
    monkeypatch.setattr(threading, "Timer", lambda *a, **k: called.append(True))

    cli_module.cmd_serve(_make_args(no_browser=True))

    assert called == []


def test_serve_still_starts_the_server_with_the_given_host_and_port(monkeypatch) -> None:
    monkeypatch.setattr(threading, "Timer", _ImmediateTimer)
    monkeypatch.setattr(webbrowser, "open", lambda url: None)
    captured = {}
    monkeypatch.setattr(
        server_module, "serve",
        lambda **kw: captured.update(kw),
    )

    cli_module.cmd_serve(_make_args(host="127.0.0.1", port=8765, reload=True))

    assert captured == {"host": "127.0.0.1", "port": 8765, "reload": True}
