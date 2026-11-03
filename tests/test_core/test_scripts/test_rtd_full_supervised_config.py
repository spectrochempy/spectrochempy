"""Keep the complete RTD workload intact when selecting the supervisor."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_full_rtd_configuration_preserves_installation_and_workload():
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load((ROOT / ".readthedocs.yaml").read_text())
    assert config["version"] == 2
    build = config["build"]
    assert build["os"] == "ubuntu-24.04"
    assert build["tools"] == {"python": "3.13"}
    assert build["apt_packages"] == ["pandoc"]
    assert "commands" not in build
    jobs = build["jobs"]
    assert jobs["post_checkout"] == [
        "git fetch --unshallow || true",
        "git fetch --all --tags || true",
    ]
    assert jobs["install"] == [
        "python .github/actions/verified-pip/verified_pip.py",
        'python -m pip install -e ".[docs]"',
        'python -m pip install "ipython==9.17.1" "jupyterlab==4.6.4" "notebook==7.6.3"',
    ]
    assert (
        "python -m spectrochempy.ci.install_plugins --list-names"
        in jobs["post_install"][0]
    )
    assert (
        'python -m pip install --no-deps -e "plugins/$name"' in jobs["post_install"][0]
    )
    assert jobs["post_install"][1:] == [
        'python -m pip install "spectrochempy-nmr>=0.2,<0.3" "spectrochempy-hypercomplex>=0.2,<0.3"',
        "python -m pip install osqp scipy numpy-quaternion tensorly",
    ]
    assert len(jobs["build"]["html"]) == 1
    assert shlex.split(jobs["build"]["html"][0]) == [
        "SCPY_DOCS_TRACE=1",
        "python",
        "-u",
        ".github/workflows/scripts/docs_build_diagnostics.py",
        "--timeout",
        "600",
        "--grace",
        "15",
        "--kill-grace",
        "5",
        "--",
        "python",
        "-u",
        "docs/make.py",
        "html",
        "--warning-is-error",
        "-j",
        "auto",
    ]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervision")
@pytest.mark.parametrize("failure", ["trace", "sink"])
def test_diagnostic_failure_never_becomes_success(tmp_path, failure):
    pidfile = tmp_path / "pid"
    code = f"import os;open({str(pidfile)!r},'w').write(str(os.getpid()))"
    if failure == "trace":
        code += ";open(os.environ['SCPY_DOCS_TRACE_FILE'],'w').write('partial-record')"
    command = [
        sys.executable,
        "-u",
        str(ROOT / ".github/workflows/scripts/docs_build_diagnostics.py"),
        "--timeout",
        "2",
        "--grace",
        ".1",
        "--kill-grace",
        ".2",
        "--",
        sys.executable,
        "-c",
        code,
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env={**os.environ, "SCPY_DOCS_TRACE": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        if failure == "sink":
            process.stdout.close()
            process.wait(timeout=5)
        else:
            output, errors = process.communicate(timeout=5)
            final = json.loads(
                next(
                    line.split("] ", 1)[1]
                    for line in output.splitlines()
                    if line.startswith("[docs recovery FINAL] ")
                )
            )
            assert final["exit_code"] == 74, errors
            assert not final["trace_complete"]
            assert final["cleanup_complete"] and final["live_pids"] == []
        assert process.returncode == 74
        assert not Path(f"/proc/{pidfile.read_text()}").exists()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=20)
        process.stdout.close()
        process.stderr.close()
