"""Phase-C observation regression tests, independent of build supervision."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from docs.tools import lifecycle_trace as trace


def records(text):
    prefix = "[docs lifecycle] "
    return [
        json.loads(line[len(prefix) :])
        for line in text.splitlines()
        if line.startswith(prefix)
    ]


def test_opt_in_exception_and_private_content(monkeypatch, tmp_path):
    path = tmp_path / "events"
    monkeypatch.setenv("SCPY_DOCS_TRACE_FILE", str(path))
    monkeypatch.setenv("SCPY_DOCS_TRACE", "0")
    monkeypatch.setattr(trace, "_targets", lambda: pytest.fail("disabled import"))
    with trace.span("off"), trace.observe():
        pass
    assert not path.exists()
    monkeypatch.setenv("SCPY_DOCS_TRACE", "1")
    with pytest.raises(ValueError), trace.span("controlled"):
        raise ValueError("private cell contents")
    start, end = records(path.read_text())
    assert start["span"] == end["span"]
    assert start["status"] == "START" and end["status"] == "END"
    assert end["outcome"] == "error" and end["duration"] >= 0
    assert end["utc"].endswith("Z") and end["pid"] == os.getpid()
    assert "private cell contents" not in path.read_text()


def test_monitoring_exception_and_release(monkeypatch, tmp_path):
    if not hasattr(sys, "monitoring"):
        pytest.skip("CPython monitoring requires 3.12+")

    def failing():
        raise TimeoutError("not logged")

    path = tmp_path / "events"
    monkeypatch.setenv("SCPY_DOCS_TRACE", "1")
    monkeypatch.setenv("SCPY_DOCS_TRACE_FILE", str(path))
    monkeypatch.setattr(trace, "_targets", lambda: {failing.__code__: "test.timeout"})
    before = [sys.monitoring.get_tool(i) for i in range(6)]
    with trace.observe(), pytest.raises(TimeoutError):
        failing()
    assert [sys.monitoring.get_tool(i) for i in range(6)] == before
    event = next(
        row for row in records(path.read_text()) if row.get("outcome") == "timeout"
    )
    assert event["exception_type"] == "TimeoutError"


def test_file_sink_with_closed_stderr(monkeypatch, tmp_path):
    path = tmp_path / "events"
    with (tmp_path / "closed").open("w") as stream:
        pass
    monkeypatch.setattr(sys, "__stderr__", stream)
    monkeypatch.setenv("SCPY_DOCS_TRACE", "1")
    monkeypatch.setenv("SCPY_DOCS_TRACE_FILE", str(path))
    trace.emit("sink", "START")
    assert records(path.read_text())[0]["phase"] == "sink"


@pytest.mark.skipif(sys.platform != "linux", reason="fork/shared pipe regression")
def test_atomic_multiprocess_lines(tmp_path):
    path = tmp_path / "events"
    code = """
import os
from docs.tools.lifecycle_trace import emit
children=[]
for worker in range(4):
    pid=os.fork()
    if not pid:
        for index in range(200): emit('parallel','START',worker=worker,index=index)
        os._exit(0)
    children.append(pid)
for pid in children: assert os.waitpid(pid,0)[1]==0
"""
    result = subprocess.run(
        [sys.executable, "-u", "-c", code],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "SCPY_DOCS_TRACE": "1", "SCPY_DOCS_TRACE_FILE": str(path)},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0
    for text in (result.stderr, path.read_text()):
        rows = records(text)
        assert len(rows) == len(text.splitlines()) == 800
        assert {(row["worker"], row["index"]) for row in rows} == {
            (worker, index) for worker in range(4) for index in range(200)
        }
