"""Exercise the RTD supervisor without a documentation build."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("raise SystemExit(0)", 0),
        ("raise SystemExit(7)", 7),
        ("import time; time.sleep(30)", 124),
        (
            "import signal, time; "
            "signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(30)",
            137,
        ),
    ],
)
def test_supervisor_status_and_log_replay(body, expected):
    child = "print('child sentinel', flush=True); " + body
    script = (
        "import sys\n"
        "from docs.tools.supervise_build import supervise\n"
        f"sys.exit(supervise([sys.executable, '-u', '-c', {child!r}], "
        "timeout=1, grace=0.2))\n"
    )
    result = subprocess.run(
        [sys.executable, "-u", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    assert "[docs supervisor] START" in result.stdout
    assert "child sentinel" in result.stdout
    assert f"EXIT exit_code={expected}" in result.stdout
    if expected == 124:
        assert "KeyboardInterrupt" in result.stdout


def test_supervisor_help_runs_real_builder():
    result = subprocess.run(
        [sys.executable, "-u", "docs/tools/supervise_build.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage: make.py" in result.stdout
    assert "[docs runner] SystemExit code=0" in result.stdout
    assert "[docs supervisor] EXIT exit_code=0" in result.stdout
