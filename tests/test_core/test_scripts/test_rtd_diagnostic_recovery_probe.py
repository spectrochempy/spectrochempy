"""Validate only the phase-E.2 fixture and experimental RTD configuration."""
# ruff: noqa: S603

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / ".github/workflows/scripts/rtd_diagnostic_recovery_probe.py"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux ownership facilities")
def test_probe_preserves_failure_and_diagnostics_without_site_packages(tmp_path):
    sentinel = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    try:
        started = time.monotonic()
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-u", str(SCRIPT)],
            cwd=tmp_path,
            env={**os.environ, "TMPDIR": str(scratch)},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 124, result.stdout + result.stderr
        assert time.monotonic() - started < 8
        final = json.loads(
            next(
                line.split("] ", 1)[1]
                for line in result.stdout.splitlines()
                if line.startswith("[docs recovery FINAL] ")
            )
        )
        end = json.loads(
            next(
                line.split("] END ", 1)[1]
                for line in result.stdout.splitlines()
                if line.startswith("[recovery probe] END ")
            )
        )
        assert final["exit_code"] == 124 and final["reason"] == "deadline"
        assert final["cleanup_complete"] and final["live_pids"] == []
        assert final["forced_kill"] and final["trace_complete"]
        assert final["active_count"] == 2 and final["events"] >= 3
        assert not final["log_delivery_failed"]
        assert end["supervisor_code"] == 124 and end["remaining_count"] == 0
        assert end["remaining_pids"] == [] and end["identity_check"]
        assert end["id"] == "scpy-rtd-diagnostic-recovery-e2"
        assert all(not Path(f"/proc/{row['pid']}").exists() for row in end["actors"])
        active = [
            line
            for line in result.stdout.splitlines()
            if line.startswith("[docs recovery active] ")
        ]
        assert len(active) == 2
        assert all('"status": "START"' in line for line in active)
        assert any("controlled-main" in line for line in active)
        assert any("controlled-descendant" in line for line in active)
        assert result.stdout.index("[docs recovery FINAL]") < result.stdout.index(
            "[recovery probe] END"
        )
        assert not list(scratch.iterdir())
        assert sentinel.poll() is None
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=5)


def test_experimental_yaml_runs_only_stdlib_timeout_probe():
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load((ROOT / ".readthedocs.yaml").read_text())
    assert config == {
        "version": 2,
        "build": {
            "os": "ubuntu-24.04",
            "tools": {"python": "3.13"},
            "commands": [
                "python -I -S -u .github/workflows/scripts/rtd_diagnostic_recovery_probe.py"
            ],
        },
    }
    assert not SCRIPT.is_relative_to(ROOT / "docs")
