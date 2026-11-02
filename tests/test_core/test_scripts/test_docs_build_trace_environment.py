"""Check opt-in CPU and dependency diagnostics without building documentation."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("trace", ["0", "1"])
@pytest.mark.parametrize("affinity_available", [False, True])
def test_docs_environment_trace(trace, affinity_available):
    script = (
        "import sys\n"
        "from unittest.mock import patch\n"
        "from docs import make\n"
        "def affinity(pid):\n"
        + (
            "    return {3, 1}\n"
            if affinity_available
            else "    raise OSError('unavailable')\n"
        )
        + "with patch.object(make.os, 'cpu_count', return_value=8), "
        "patch.object(make.mp, 'cpu_count', return_value=8), "
        "patch.object(make.os, 'sched_getaffinity', affinity, create=True), "
        "patch.object(make.metadata, 'version', return_value='test-version'):\n"
        "    make._trace_docs_environment(make.BuildDocumentation._get_jobs('2'))\n"
        "assert 'spectrochempy' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "SCPY_DOCS_TRACE": trace, "GITHUB_ACTIONS": "false"},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if trace == "0":
        assert result.stdout == ""
        return
    assert result.stdout.startswith("[docs trace] environment=")
    info = json.loads(result.stdout.split("=", 1)[1])
    assert info["os.cpu_count"] == 8
    assert info["multiprocessing.cpu_count"] == 8
    assert info["affinity"] == ([1, 3] if affinity_available else None)
    assert info["sphinx_workers"] == 2
    assert info["versions"] == dict.fromkeys(
        ("ipython", "jupyterlab", "notebook"), "test-version"
    )
