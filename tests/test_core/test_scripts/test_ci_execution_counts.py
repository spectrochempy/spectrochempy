"""Exercise the CI execution counter with real pytest sessions."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests._ci_execution_counts import check_execution


@pytest.mark.parametrize(
    ("body", "exitstatus", "executed"),
    [
        ("def test_ok():\n    pass\n", 0, 1),
        ("def test_bad():\n    assert False\n", 1, 1),
        ("", 5, 0),
        ("@pytest.mark.skip\ndef test_skip():\n    pass\n", 0, 0),
        ("@pytest.mark.xfail\ndef test_xfail():\n    assert False\n", 0, 1),
        ("@pytest.mark.xfail(run=False)\ndef test_notrun():\n    pass\n", 0, 0),
    ],
)
@pytest.mark.parametrize("workers", [0, 2])
def test_counts_actual_calls(tmp_path, body, exitstatus, executed, workers):
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    target = tmp_path / "test_cases.py"
    target.write_text("import pytest\n" + body, encoding="utf-8")
    counts = tmp_path / "counts.json"
    env = os.environ.copy()
    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        env.pop(name, None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    env["SCPY_CI_TEST_COUNTS"] = str(counts)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--noconftest",
        "-p",
        "tests._ci_execution_counts",
        "-q",
        str(target),
    ]
    if workers:
        command.extend(["-p", "xdist.plugin", "-n", str(workers)])
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == exitstatus, result.stdout + result.stderr
    report = json.loads(counts.read_text(encoding="utf-8"))
    assert report["executed"] == executed
    assert report["exitstatus"] == exitstatus


def test_all_empty_invocations_fail(tmp_path):
    for name in ("primary-1", "primary-2", "serial"):
        (tmp_path / f"{name}.json").write_text('{"executed": 0}', encoding="utf-8")
    with pytest.raises(ValueError, match="did not execute"):
        check_execution(tmp_path)


@pytest.mark.parametrize("nonempty", ["primary-1", "serial"])
def test_empty_partition_with_executed_tests_is_valid(tmp_path, nonempty):
    for name in ("primary-1", "serial"):
        (tmp_path / f"{name}.json").write_text(
            json.dumps({"executed": 2 if name == nonempty else 0}), encoding="utf-8"
        )
    assert check_execution(tmp_path) == 2


def test_missing_reports_fail(tmp_path):
    with pytest.raises(ValueError, match="No pytest execution reports"):
        check_execution(tmp_path)


@pytest.mark.parametrize("count", [-1, True, "1"])
def test_invalid_counts_fail(tmp_path, count):
    (tmp_path / "primary.json").write_text(
        json.dumps({"executed": count}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Invalid executed-test count"):
        check_execution(tmp_path)
