"""Check RTD diagnostics in fresh interpreters without building documentation."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def run_script(script):
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


@pytest.mark.parametrize("code", [0, 2])
def test_runner_preserves_exit_code(code):
    result = run_script(
        "from docs.tools import trace_build as runner\n"
        "def build(*args, **kwargs):\n"
        f"    raise SystemExit({code})\n"
        "runner.runpy.run_path = build\n"
        "runner.main()\n"
    )
    assert result.returncode == code
    assert f"SystemExit code={code}" in result.stderr
    assert "[docs runner] BUILDER_RETURN elapsed=" in result.stderr


def test_runner_exception_survives_python_stderr_redirection():
    result = run_script(
        "import io, sys\n"
        "from docs.tools import trace_build as runner\n"
        "def build(*args, **kwargs):\n"
        "    sys.stderr = io.StringIO()\n"
        "    raise RuntimeError('diagnostic sentinel')\n"
        "runner.runpy.run_path = build\n"
        "runner.main()\n"
    )
    assert result.returncode != 0
    assert "Traceback (most recent call last)" in result.stderr
    assert "RuntimeError: diagnostic sentinel" in result.stderr
    assert "rss_self_peak_kib=" in result.stderr


def test_runner_watchdog_dumps_stacks():
    result = run_script(
        "import time\n"
        "from docs.tools import trace_build as runner\n"
        "original = runner.faulthandler.dump_traceback_later\n"
        "runner.faulthandler.dump_traceback_later = "
        "lambda seconds, **kwargs: original(0.05, **kwargs)\n"
        "runner.runpy.run_path = lambda *args, **kwargs: time.sleep(0.2)\n"
        "runner.main()\n"
    )
    assert result.returncode == 0
    assert "Timeout (0:00:00.050000)!" in result.stderr
    assert "trace_build.py" in result.stderr
    assert "[docs runner] BUILDER_RETURN elapsed=" in result.stderr


def test_runner_watchdog_remains_active_during_atexit():
    result = run_script(
        "import atexit, time\n"
        "from docs.tools import trace_build as runner\n"
        "def finalizer():\n"
        "    time.sleep(0.25)\n"
        "atexit.register(finalizer)\n"
        "original = runner.faulthandler.dump_traceback_later\n"
        "runner.faulthandler.dump_traceback_later = "
        "lambda seconds, **kwargs: original(0.05, **kwargs)\n"
        "runner.runpy.run_path = lambda *args, **kwargs: None\n"
        "runner.main()\n"
    )
    assert result.returncode == 0
    assert "Timeout (0:00:00.050000)!" in result.stderr
    assert "in finalizer" in result.stderr
    assert result.stderr.index("BUILDER_RETURN") < result.stderr.index("Timeout (")
