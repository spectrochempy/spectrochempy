"""Bounded Linux subprocess tests of the standalone RTD capture experiment."""
# ruff: noqa: S603

import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / ".github" / "workflows" / "scripts" / "rtd_process_capture_probe.py"
SPEC = importlib.util.spec_from_file_location("rtd_process_capture_probe", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux subreaper/pidfd experiment"
)


@pytest.fixture(scope="module")
def historical_supervisor():
    """Fetch only the probe's historical dependency, only when it is missing."""
    git = shutil.which("git")
    assert git is not None, "git is required for the historical supervisor tests"
    object_name = f"{probe.REVISION}:docs/tools/supervise_build.py"
    present = subprocess.run(
        [git, "cat-file", "-e", object_name],
        cwd=ROOT,
        capture_output=True,
        timeout=5,
        check=False,
    )
    if present.returncode:
        subprocess.run(
            [git, "fetch", "--no-tags", "origin", probe.REVISION],
            cwd=ROOT,
            capture_output=True,
            timeout=60,
            check=True,
        )
        subprocess.run(
            [git, "cat-file", "-e", object_name],
            cwd=ROOT,
            capture_output=True,
            timeout=5,
            check=True,
        )


@pytest.mark.usefixtures("historical_supervisor")
def test_paired_capture_and_historical_cleanup(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    result = subprocess.run(
        [sys.executable, "-I", "-u", str(SCRIPT), "run", "0.5"],
        cwd=ROOT,
        env={
            **os.environ,
            "TMPDIR": str(scratch),
            "READTHEDOCS_OUTPUT": str(tmp_path / "output"),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[0]["event"] == "probe_start"
    assert events[-1]["event"] == "probe_end"
    assert events[-1]["elapsed"] < 25
    scenarios = {
        (e["capture"], e["mode"]): e for e in events if e["event"] == "scenario"
    }
    assert set(scenarios) == {
        (capture, mode) for capture in ("direct", "historical") for mode in ("A", "B")
    }
    inherited = scenarios["direct", "A"]
    redirected = scenarios["direct", "B"]
    assert (
        inherited["main_exit_observed"]
        < inherited["child_end_marker"]
        <= inherited["eof"]
    )
    assert redirected["eof"] < redirected["child_end_marker"]
    assert inherited["child_identity"]["stdout"].startswith("pipe:")
    assert not redirected["child_identity"]["stdout"].startswith("pipe:")
    for row in scenarios.values():
        assert row["returncode"] == 0
        assert row["survivors"] == []
        child = row["child_identity"]
        assert child["pid"] == child["sid"] == child["pgid"]
        assert child["ppid"] == row["main_identity"]["pid"]
        assert not Path(f"/proc/{child['pid']}").exists()
        assert not Path(f"/proc/{row['main_identity']['pid']}").exists()
        if row["capture"] == "direct":
            assert (
                row["child_reaped"]["returncode"] == row["child_declared_status"] == 0
            )
        else:
            assert row["child_reaped"] is None
            assert row["child_declared_status"] == 143
            assert "parent exited; cleaning descendants" in row["supervisor_output"]
            assert "EXIT exit_code=0" in row["supervisor_output"]
    assert not list(scratch.iterdir())
    assert (tmp_path / "output" / "html" / "index.html").exists()


@pytest.mark.usefixtures("historical_supervisor")
def test_external_interruption_cleans_only_probe_processes(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    sentinel = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    process = None
    identities = []
    try:
        process = subprocess.Popen(
            [sys.executable, "-I", "-u", str(SCRIPT), "run"],
            cwd=ROOT,
            env={**os.environ, "TMPDIR": str(scratch)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            logs = list(scratch.glob("*/events.jsonl"))
            if logs:
                identities = [
                    json.loads(line) for line in logs[0].read_text().splitlines()
                ]
                if any(e["event"] == "child_start" for e in identities):
                    break
            time.sleep(0.01)
        assert any(e["event"] == "child_start" for e in identities)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode != 0
        assert "bounded probe deadline or external interruption" in stderr
        assert sentinel.poll() is None
        assert not list(scratch.iterdir())
        assert all(not Path(f"/proc/{e['pid']}").exists() for e in identities)
    finally:
        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        if process:
            process.stdout.close()
            process.stderr.close()
        sentinel.terminate()
        sentinel.wait(timeout=5)


def test_probe_configuration_has_no_documentation_install():
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load((ROOT / ".readthedocs.yaml").read_text())
    assert set(config) == {"version", "build"}
    build = config["build"]
    assert set(build) == {"os", "tools", "commands"}
    assert build["commands"] == [
        "git fetch --depth 1 origin ef092a238d3a968bf21f9129e2e3934c3d1a07ce",
        "python -u .github/workflows/scripts/rtd_process_capture_probe.py run",
    ]
    assert not SCRIPT.is_relative_to(ROOT / "docs")


@pytest.mark.parametrize("failure", ["constructor", "registration"])
def test_selector_failure_still_closes_and_cleans(tmp_path, monkeypatch, failure):
    process = SimpleNamespace(stdout=io.BytesIO())
    cleanup_calls = []
    selector = SimpleNamespace(closed=False)

    def register(*args):
        raise OSError("injected registration failure")

    def close():
        selector.closed = True

    def construct():
        if failure == "constructor":
            raise OSError("injected selector failure")
        return selector

    selector.register = register
    selector.close = close
    monkeypatch.setattr(probe.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(probe.selectors, "DefaultSelector", construct)
    monkeypatch.setattr(probe, "cleanup", lambda *args: cleanup_calls.append(args))
    monkeypatch.setattr(probe.tempfile, "tempdir", str(tmp_path))
    with pytest.raises(OSError, match="injected"):
        probe.scenario(None, "A", False, 0.5, time.monotonic() + 5, [])
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0][1] is process
    assert process.stdout.closed
    assert selector.closed == (failure == "registration")
    assert not list(tmp_path.iterdir())
