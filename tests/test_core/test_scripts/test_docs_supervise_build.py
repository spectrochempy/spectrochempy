"""Exercise the RTD supervisor without a documentation build."""

import json
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="RTD supervisor is Linux-only"
)


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


def _run(source, **kwargs):
    return subprocess.run(
        [sys.executable, "-u", "-c", source],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        **kwargs,
    )


def _stop_test_processes(pidfiles):
    for file in pidfiles:
        if not file.exists():
            continue
        pid = int(file.read_text())
        path = Path(f"/proc/{pid}/cmdline")
        try:
            if b"rtd-test-child" in path.read_bytes():
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except FileNotFoundError:
            pass


@pytest.mark.parametrize("detached", [False, True])
@pytest.mark.parametrize("parent_timeout", [False, True])
def test_supervisor_cleans_descendants_after_parent_exit(
    tmp_path, detached, parent_timeout
):
    pidfile = tmp_path / "child.pid"
    child = tmp_path / "rtd-test-child.py"
    child.write_text(
        "import os, signal, time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(pidfile)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        f"subprocess.Popen([sys.executable, '-u', {str(child)!r}], start_new_session={detached!r})\n"
        f"ready = Path({str(pidfile)!r})\n"
        "deadline = time.monotonic() + 3\n"
        "while not ready.exists() and time.monotonic() < deadline: time.sleep(0.01)\n"
        "assert ready.exists()\n" + ("time.sleep(60)\n" if parent_timeout else "")
    )
    source = (
        "import sys\nfrom docs.tools.supervise_build import supervise\n"
        f"sys.exit(supervise([sys.executable, '-u', {str(parent)!r}], "
        "timeout=1, grace=0.15, kill_grace=1))\n"
    )
    try:
        result = _run(source)
        assert result.returncode == (124 if parent_timeout else 0), (
            result.stdout + result.stderr
        )
        assert "forced cleanup SIGKILL" in result.stdout
        assert not Path(f"/proc/{int(pidfile.read_text())}").exists()
        assert "CLEANUP FAILED" not in result.stdout
    finally:
        _stop_test_processes([pidfile])


@pytest.mark.parametrize("detached_child", [False, True])
def test_supervisor_streams_before_external_sigterm_and_cleans_task(
    tmp_path, detached_child
):
    pidfile = tmp_path / "root.pid"
    extra_pidfile = tmp_path / "extra.pid"
    extra = tmp_path / "rtd-test-child-detached.py"
    extra.write_text(
        "import os, signal, time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(extra_pidfile)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    launch = (
        (
            "import subprocess, sys\n"
            f"subprocess.Popen([sys.executable, {str(extra)!r}], start_new_session=True)\n"
            f"ready = Path({str(extra_pidfile)!r})\n"
            "deadline = time.monotonic() + 3\n"
            "while not ready.exists() and time.monotonic() < deadline: time.sleep(0.01)\n"
            "assert ready.exists()\n"
        )
        if detached_child
        else ""
    )
    child = tmp_path / "rtd-test-child.py"
    child.write_text(
        "import os, time\nfrom pathlib import Path\n"
        f"Path({str(pidfile)!r}).write_text(str(os.getpid()))\n"
        + launch
        + "print('stream sentinel', flush=True)\ntime.sleep(60)\n"
    )
    source = (
        "import sys\nfrom docs.tools.supervise_build import supervise\n"
        f"sys.exit(supervise([sys.executable, '-u', {str(child)!r}], "
        "timeout=10, grace=0.15, kill_grace=1))\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", source],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    received = b""
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + 5
            while b"stream sentinel" not in received and time.monotonic() < deadline:
                if selector.select(timeout=0.1):
                    received += os.read(process.stdout.fileno(), 65536)
        assert b"stream sentinel" in received
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 143, (received + stdout, stderr)
        assert b"cgroup_final=" in stdout
        assert not Path(f"/proc/{int(pidfile.read_text())}").exists()
        if detached_child:
            assert not Path(f"/proc/{int(extra_pidfile.read_text())}").exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
        _stop_test_processes([pidfile, extra_pidfile])


def test_supervisor_reads_final_cgroup_after_child_failure(tmp_path):
    (tmp_path / "memory.events").write_text("oom 0\noom_kill 0\n")
    (tmp_path / "memory.peak").write_text("4096")
    child = (
        "from pathlib import Path; "
        f"Path({str(tmp_path / 'memory.events')!r}).write_text('oom 1\\noom_kill 1\\n'); "
        "raise SystemExit(7)"
    )
    source = (
        "import sys\nfrom pathlib import Path\n"
        "from docs.tools import supervise_build as module\n"
        f"module._cgroup_directory = lambda: Path({str(tmp_path)!r})\n"
        f"sys.exit(module.supervise([sys.executable, '-c', {child!r}], timeout=2))\n"
    )
    result = _run(source)
    assert result.returncode == 7, result.stdout + result.stderr
    delta = next(
        line.split("=", 1)[1]
        for line in result.stdout.splitlines()
        if "cgroup_event_delta=" in line
    )
    assert json.loads(delta) == {"oom": 1, "oom_kill": 1}
    assert '"memory.peak": "4096"' in result.stdout


@pytest.mark.parametrize("child_code", [0, 7])
def test_supervisor_broken_output_does_not_mask_build_error(child_code):
    source = (
        "import errno, sys, tempfile\nfrom docs.tools import supervise_build as module\n"
        "tempfile.gettempdir()\n"
        "def broken(*args): raise BrokenPipeError(errno.EPIPE, 'test sink')\n"
        "module.os.write = broken\n"
        f"sys.exit(module.supervise([sys.executable, '-c', 'raise SystemExit({child_code})'], timeout=2))\n"
    )
    result = _run(source)
    assert result.returncode == (74 if child_code == 0 else child_code), result.stderr


@pytest.mark.parametrize("child_code", [0, 7])
def test_supervisor_log_read_error_preserves_build_failure(child_code):
    source = (
        "import sys\nfrom docs.tools import supervise_build as module\n"
        "def broken(*args): raise OSError('test read error')\n"
        "module.os.pread = broken\n"
        f"sys.exit(module.supervise([sys.executable, '-c', 'raise SystemExit({child_code})'], timeout=2))\n"
    )
    result = _run(source)
    assert result.returncode == (74 if child_code == 0 else child_code), result.stderr
    assert "log read failed: test read error" in result.stdout


def test_supervisor_pread_does_not_move_child_write_offset():
    child = (
        "import time\n"
        "for i in range(100):\n"
        "    print(f'ordered sentinel {i}', flush=True)\n"
        "    time.sleep(0.002)\n"
    )
    source = (
        "import sys\nfrom docs.tools.supervise_build import supervise\n"
        f"sys.exit(supervise([sys.executable, '-u', '-c', {child!r}], timeout=3))\n"
    )
    result = _run(source)
    assert result.returncode == 0, result.stdout + result.stderr
    observed = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("ordered sentinel")
    ]
    assert observed == [f"ordered sentinel {i}" for i in range(100)]


def test_supervisor_does_not_wait_forever_on_blocked_log_sink():
    source = (
        "import sys\nfrom docs.tools.supervise_build import supervise\n"
        "sys.exit(supervise([sys.executable, '-c', \"print('x' * 3000000)\"], timeout=2, grace=0.2))\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", source],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert process.wait(timeout=8) == 74
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
