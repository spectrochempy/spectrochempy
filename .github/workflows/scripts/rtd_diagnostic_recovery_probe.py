"""Phase E.2: deliberately return 124 after bounded diagnostic recovery."""
# ruff: noqa: S603,T201

import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).resolve()
PROBE_ID = "scpy-rtd-diagnostic-recovery-e2"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT.parent))

import docs_build_diagnostics  # noqa: E402

from docs.tools.lifecycle_trace import emit  # noqa: E402


def _identity(path, role):
    row = {
        "role": role,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "pgid": os.getpgrp(),
        "sid": os.getsid(0),
    }
    temporary = path.with_name(role + ".tmp")
    temporary.write_text(json.dumps(row), encoding="utf-8")
    temporary.replace(path.with_name(role + ".json"))


def actor(role, path):
    if not stat.S_ISREG(os.fstat(1).st_mode) or not stat.S_ISREG(os.fstat(2).st_mode):
        raise RuntimeError("probe actor output must use the supervisor's regular file")
    if role == "child":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        emit(
            PROBE_ID + ".descendant",
            "START",
            span=PROBE_ID + ":child",
            document="controlled-descendant",
        )
        _identity(path, role)
    else:
        _identity(path, role)
        emit(
            PROBE_ID + ".blocked",
            "START",
            span=PROBE_ID + ":main",
            document="controlled-main",
        )
        subprocess.Popen(
            [sys.executable, "-I", "-S", "-u", str(SCRIPT), "child", str(path)],
            start_new_session=True,
        )
        ready_by = time.monotonic() + 1.5
        while not path.with_name("child.json").exists():
            if time.monotonic() >= ready_by:
                raise TimeoutError("controlled descendant readiness timeout")
            time.sleep(0.005)
        emit(
            PROBE_ID,
            "INFO",
            document="both actors ready; waiting for internal deadline",
        )
    time.sleep(30)


def run_probe():
    os.environ["SCPY_DOCS_TRACE"] = "1"
    print(
        f"[recovery probe] START id={PROBE_ID} timeout=3 grace=0.3 kill_grace=0.3",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="scpy-recovery-probe-") as directory:
        path = Path(directory) / "actors"
        code = docs_build_diagnostics.supervise(
            [sys.executable, "-I", "-S", "-u", str(SCRIPT), "main", str(path)],
            timeout=3,
            grace=0.3,
            kill_grace=0.3,
        )
        identities = [
            json.loads(file.read_text())
            for file in sorted(Path(directory).glob("*.json"))
        ]
        remaining = [
            row["pid"] for row in identities if Path(f"/proc/{row['pid']}").exists()
        ]
        shape_ok = (
            {row["role"] for row in identities} == {"main", "child"}
            and len(identities) == 2
            and all(row["pid"] == row["pgid"] == row["sid"] for row in identities)
        )
        print(
            "[recovery probe] END "
            + json.dumps(
                {
                    "id": PROBE_ID,
                    "supervisor_code": code,
                    "actors": identities,
                    "remaining_pids": remaining,
                    "remaining_count": len(remaining),
                    "identity_check": shape_ok,
                }
            ),
            flush=True,
        )
        if remaining or not shape_ok:
            return code if code not in (0, 124) else 70
        return code or 1


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit(run_probe())
    if len(sys.argv) != 3 or sys.argv[1] not in ("main", "child"):
        raise SystemExit("expected no arguments, or actor role and identity file")
    actor(sys.argv[1], Path(sys.argv[2]))
