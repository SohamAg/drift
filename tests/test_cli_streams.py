"""Console output encoding: verdict lines must survive pipes on Windows."""
import io

from drift.cli import _configure_streams


class _Pipe(io.TextIOWrapper):
    def __init__(self, encoding, tty=False):
        super().__init__(io.BytesIO(), encoding=encoding, errors="strict")
        self._tty = tty

    def isatty(self):
        return self._tty


def _bytes(stream):
    stream.flush()
    return stream.buffer.getvalue()


def test_cp1252_pipe_switches_to_utf8():
    s = _Pipe("cp1252")
    _configure_streams([s], env={})
    s.write("FAIL — 2 × crash")
    assert _bytes(s).decode("utf-8") == "FAIL — 2 × crash"


def test_explicit_pythonioencoding_is_respected():
    s = _Pipe("cp1252")
    _configure_streams([s], env={"PYTHONIOENCODING": "cp1252"})
    s.write("a — b")
    assert _bytes(s) == "a — b".encode("cp1252")


def test_terminal_keeps_encoding_but_never_crashes():
    s = _Pipe("ascii", tty=True)
    _configure_streams([s], env={})
    s.write("a — b")  # would raise UnicodeEncodeError without errors=replace
    assert _bytes(s) == b"a ? b"


def test_stream_without_reconfigure_is_ignored():
    class Bare:
        def write(self, s):  # pragma: no cover - not called
            pass
    _configure_streams([Bare()], env={})
