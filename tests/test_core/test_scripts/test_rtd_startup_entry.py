"""Independent shell milestone and future CLI exception-boundary regressions."""
# ruff: noqa: S603

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

if sys.platform != "linux":
    pytest.skip("Linux controller entry", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / ".github/workflows/scripts/docs_controller_entry.py"
SPEC = importlib.util.spec_from_file_location("docs_controller_entry", SCRIPT)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def failure_record(output):
    return json.loads(
        next(
            line.split("] ", 1)[1]
            for line in output.splitlines()
            if line.startswith("[docs controller FAILURE] ")
        )
    )


def test_shell_marker_completes_without_running_python(tmp_path):
    fake_python = tmp_path / "python"
    ran = tmp_path / "python-ran"
    fake_python.write_text(f"#!/bin/sh\ntouch '{ran}'\nexit 99\n")
    fake_python.chmod(0o700)
    result = subprocess.run(
        ["/bin/sh", str(ROOT / ".github/workflows/scripts/rtd_startup_preflight.sh")],
        cwd=ROOT,
        env={**os.environ, "PATH": str(tmp_path) + ":/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    assert result.returncode == 0
    assert (
        "[startup shell] id=scpy-rtd-installed-startup stage=before-python"
        in result.stdout
    )
    assert "following Python command not yet observed" in result.stdout
    assert not ran.exists()


@pytest.mark.parametrize("failure", ["import", "main"])
def test_entry_reports_exception_before_loop_without_claiming_cleanup(
    monkeypatch, capfd, failure
):
    def fail():
        raise OSError("private exception content")

    if failure == "import":
        monkeypatch.setattr(entry, "load_supervisor", fail)
    else:
        monkeypatch.setattr(
            entry, "load_supervisor", lambda: SimpleNamespace(main=fail)
        )
    os.write(1, b"previous output without newline")
    assert entry.run_main() == 1
    output = capfd.readouterr().out
    row = failure_record(output)
    assert row["exit_code"] == 1 and row["exception_type"] == "OSError"
    assert not row["adapter_owns_children"]
    assert row["cleanup_status"] == "not inferred from escaped exception"
    assert "private exception content" not in output


def test_entry_preserves_preexisting_child_protection(monkeypatch, capfd):
    sentinel = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True
    )
    try:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(SCRIPT),
                "--timeout",
                ".2",
                "--",
                sys.executable,
                "-c",
                "raise RuntimeError('must not execute')",
            ],
        )
        assert entry.run_main() == 1
        row = failure_record(capfd.readouterr().out)
        assert row["phase"] == "calling supervisor main"
        assert row["exception_type"] == "RuntimeError"
        assert not row["adapter_owns_children"]
        assert sentinel.poll() is None
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=5)


def test_entry_cli_requires_explicit_child_and_preserves_status():
    no_child = subprocess.run(
        [sys.executable, "-I", "-S", str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    assert no_child.returncode == 1
    assert failure_record(no_child.stdout)["phase"] == "explicit child command required"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(SCRIPT),
            "--timeout",
            ".2",
            "--grace",
            ".1",
            "--kill-grace",
            ".2",
            "--",
            sys.executable,
            "-c",
            "import time;time.sleep(5)",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 124
    final = json.loads(
        next(
            line.split("] ", 1)[1]
            for line in result.stdout.splitlines()
            if line.startswith("[docs recovery FINAL] ")
        )
    )
    assert final["cleanup_complete"] and final["live_pids"] == []
