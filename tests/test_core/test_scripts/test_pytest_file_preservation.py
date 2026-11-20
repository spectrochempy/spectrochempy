"""A pytest session must leave checkout files intact, including on failure."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("outcome", ["pass", "fail", "collect"])
def test_pytest_preserves_checkout_files(tmp_path, outcome):
    root = Path(__file__).resolve().parents[3]
    checkout = tmp_path / "checkout"
    tests = checkout / "tests"
    plotting = tests / "test_plotting"
    plotting.mkdir(parents=True)
    for package in (tests, plotting):
        (package / "__init__.py").touch()
    shutil.copyfile(root / "tests/conftest.py", tests / "conftest.py")
    shutil.copyfile(
        root / "tests/test_plotting/plotting_fixtures.py",
        plotting / "plotting_fixtures.py",
    )
    paths = (
        ".github/data/corn.json",
        "fixtures/source.scp",
        "fixtures/source.pscp",
        "fixtures/source.jdx",
        "diagnostics/build.log",
        "local-notes.json",
    )
    expected = {}
    for name in paths:
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        expected[path] = f"preserve {name}\n".encode()
        path.write_bytes(expected[path])
    (checkout / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tests / "test_smoke.py").write_text(
        "import tempfile\n"
        "from pathlib import Path\n\n"
        "def test_smoke():\n"
        "    with tempfile.TemporaryDirectory() as directory:\n"
        "        output = Path(directory) / 'generated.scp'\n"
        "        output.write_bytes(b'test output')\n"
        "        assert output.exists()\n"
        "    assert not output.exists()\n"
        f"    assert {outcome != 'fail'}\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["SCP_TEST_DATA_DOWNLOAD"] = "0"
    env["PYTHONPATH"] = os.pathsep.join((str(checkout), str(root / "src")))
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_smoke.py"]
    if outcome == "collect":
        command.append("--collect-only")
    result = subprocess.run(
        command,
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == (1 if outcome == "fail" else 0), (
        result.stdout + result.stderr
    )
    for path, content in expected.items():
        assert path.is_file(), f"pytest deleted {path.relative_to(checkout)}"
        assert path.read_bytes() == content
