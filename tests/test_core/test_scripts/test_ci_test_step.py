"""Run the workflow's complete test step against small synthetic suites."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests._ci_bash import resolve_bash


@pytest.mark.parametrize(
    ("body", "event", "expected"),
    [
        ("", "pull_request", 1),
        ("", "push", 1),
        ("def test_ok():\n    pass\n", "pull_request", 0),
        ("@pytest.mark.serial\ndef test_serial():\n    pass\n", "pull_request", 0),
        ("@pytest.mark.skip\ndef test_skip():\n    pass\n", "pull_request", 1),
        ("def test_failure():\n    assert False\n", "pull_request", 1),
        ("def test_ok():\n    pass\n", "push", 0),
    ],
)
def test_workflow_exit_status(tmp_path, body, event, expected):
    bash = resolve_bash()
    if bash is None:
        pytest.skip("CI needs native bash or Git Bash")
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load(
        (root / ".github/workflows/test_package.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["test-package"]["steps"]
    script = next(s["run"] for s in steps if s.get("name") == "Test package")
    for expression, value in {
        "matrix.test_targets || 'tests'": "test_cases.py",
        "matrix.docs_examples": "true",
        "matrix.core_only": "false",
        "matrix.coverage": "false",
        "github.event_name": event,
    }.items():
        script = script.replace("${{ " + expression + " }}", value)
    assert "${{" not in script
    script = script.replace("-n auto", "-n 2")
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    serial\n", encoding="utf-8"
    )
    (tmp_path / "test_cases.py").write_text("import pytest\n" + body, encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "__init__.py").touch()
    shutil.copyfile(
        root / "tests/_ci_execution_counts.py", tests / "_ci_execution_counts.py"
    )
    env = os.environ.copy()
    for name in ("SCPY_TEST_TARGETS", "SCPY_CI_REPORT_DIR", "PYTEST_ADDOPTS"):
        env.pop(name, None)
    env.update(
        {
            "PATH": str(Path(sys.executable).parent) + os.pathsep + env["PATH"],
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTEST_PLUGINS": "xdist.plugin",
            "PYTHONPATH": str(tmp_path),
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_ENV": str(tmp_path / "env"),
            "CI_TIMING_FILE": str(tmp_path / "timings"),
        }
    )
    result = subprocess.run(
        [bash, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == expected, result.stdout + result.stderr
