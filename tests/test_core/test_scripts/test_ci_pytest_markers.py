"""Verify the tests selected by the workflow's real pytest argument arrays."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.mark.skipif(shutil.which("bash") is None, reason="CI uses bash")
@pytest.mark.parametrize("event", ["pull_request", "push"])
@pytest.mark.parametrize("core_only", [True, False])
def test_ci_pytest_marker_partitions(tmp_path, core_only, event):
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load(
        (root / ".github/workflows/test_package.yml").read_text(encoding="utf-8"),
    )
    step = next(
        step
        for step in workflow["jobs"]["test-package"]["steps"]
        if step.get("name") == "Test package"
    )
    setup = step["run"].split("run_tests() {", 1)[0]
    for expression, value in {
        "matrix.test_targets || 'tests'": "test_markers.py",
        "matrix.docs_examples": "false",
        "matrix.core_only": str(core_only).lower(),
        "github.event_name": event,
    }.items():
        setup = setup.replace("${{ " + expression + " }}", value)
    assert "${{" not in setup
    emit = (
        shlex.quote(Path(sys.executable).as_posix())
        + " -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "
    )
    setup += emit + '"${pytest_xdist_args[@]}"\n'
    setup += emit + '"${serial_pytest_args[@]}"\n'
    env = os.environ.copy()
    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "SCPY_TEST_TARGETS"):
        env.pop(name, None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        ["bash", "-c", setup],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    argument_sets = [json.loads(line) for line in result.stdout.splitlines()[-2:]]
    cases = {}
    source = ["import pytest", ""]
    for serial in (False, True):
        for group in ((), ("network",), ("data",), ("network", "data")):
            markers = (*group, *(("serial",) if serial else ()))
            name = "test_" + "_".join(markers or ("plain",))
            cases[name] = set(markers)
            source.extend(f"@pytest.mark.{marker}" for marker in markers)
            source.extend((f"def {name}():", "    pass", ""))
    (tmp_path / "test_markers.py").write_text("\n".join(source), encoding="utf-8")
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    network\n    data\n    serial\n", encoding="utf-8"
    )
    selected_sets = []
    partitions = argument_sets[: 2 if event == "pull_request" else 1]
    for partition, arguments in enumerate(partitions):
        if "-n" in arguments:
            index = arguments.index("-n")
            assert arguments[index + 1] == "auto"
            del arguments[index : index + 2]
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--noconftest",
                "--collect-only",
                "-q",
                "test_markers.py",
                *arguments,
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        selected = {
            line.split("::")[-1]
            for line in result.stdout.splitlines()
            if line.startswith("test_markers.py::")
        }
        expected = {
            name
            for name, markers in cases.items()
            if (not core_only or not markers.intersection({"network", "data"}))
            and (event != "pull_request" or ("serial" in markers) == bool(partition))
        }
        assert selected == expected
        selected_sets.append(selected)
    if event == "pull_request":
        assert selected_sets[0].isdisjoint(selected_sets[1])
