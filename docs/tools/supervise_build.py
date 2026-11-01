# ruff: noqa: S603
"""Linux-only, standalone RTD supervisor with bounded process and log cleanup."""

import ctypes
import errno
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import suppress
from pathlib import Path

_LIBC = ctypes.CDLL(None, use_errno=True)


def _pidfd_open(pid):
    fd = _LIBC.pidfd_open(pid, 0)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


class _Output:
    def __init__(self):
        self.fd = os.dup(1)
        self.flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
        os.set_blocking(self.fd, False)
        self.pending = b""
        self.failed = False

    def send(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8", errors="replace")
        if len(self.pending) + len(data) > 2097152:
            self.failed = True
            return
        self.pending += data
        self.flush()

    def flush(self):
        while self.pending:
            try:
                written = os.write(self.fd, self.pending)
            except BlockingIOError:
                return
            except OSError:
                self.failed = True
                self.pending = b""
                return
            if not written:
                self.failed = True
                return
            self.pending = self.pending[written:]

    def finish(self):
        deadline = time.monotonic() + 0.2
        while self.pending and time.monotonic() < deadline:
            self.flush()
            if self.pending:
                time.sleep(0.01)
        self.failed |= bool(self.pending)

    def close(self):
        fcntl.fcntl(self.fd, fcntl.F_SETFL, self.flags)
        os.close(self.fd)


def _process_table():
    table = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            table[int(path.parent.name)] = (int(fields[1]), fields[0], fields[19])
        except (OSError, ValueError, IndexError):
            continue
    return table


def _owned_processes(process):
    table = _process_table()
    parents = {os.getpid()}
    owned = {}
    while children := {
        pid: info
        for pid, info in table.items()
        if info[0] in parents and pid not in owned
    }:
        owned.update(children)
        parents = set(children)
    for pid, (_, state, _) in owned.items():
        if state == "Z" and pid != process.pid:
            with suppress(ChildProcessError):
                os.waitpid(pid, os.WNOHANG)
    return {pid: info for pid, info in owned.items() if info[1] != "Z"}


def _signal_owned(owned, sig):
    for pid, identity in owned.items():
        try:
            fd = _pidfd_open(pid)
        except ProcessLookupError:
            continue
        try:
            current = _process_table().get(pid)
            if (
                current is not None
                and current[2] == identity[2]
                and _LIBC.pidfd_send_signal(fd, int(sig), None, 0) != 0
            ):
                error = ctypes.get_errno()
                if error != errno.ESRCH:
                    raise OSError(error, os.strerror(error))
        finally:
            os.close(fd)


def _subreaper(value=None):
    previous = ctypes.c_int()
    if _LIBC.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_GET_CHILD_SUBREAPER failed")
    if value is not None and _LIBC.prctl(36, int(value), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")
    return previous.value


def _cgroup_directory():
    try:
        membership = next(
            line[3:]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            if " - cgroup2 " not in line:
                continue
            fields = line.split()
            mount = Path(fields[4])
            if membership == "/":
                return mount
            with suppress(ValueError):
                return mount / Path(membership).relative_to(fields[3])
    except (OSError, StopIteration):
        pass
    return None


def _snapshot(directory):
    values = {
        "scope": "whole cgroup, not exclusive task memory",
        "path": str(directory),
    }
    if directory is not None:
        for name in (
            "cpu.max",
            "memory.max",
            "memory.current",
            "memory.peak",
            "memory.events",
        ):
            try:
                values[name] = (directory / name).read_text().strip()
            except OSError as error:
                values[name] = f"unavailable: {error}"
    return values


def _event_delta(before, after):
    def counters(snapshot):
        return {
            key: int(value)
            for line in snapshot.get("memory.events", "").splitlines()
            if len(parts := line.split()) == 2
            for key, value in [parts]
            if value.isdigit()
        }

    initial, final = counters(before), counters(after)
    return {key: final[key] - initial[key] for key in initial.keys() & final.keys()}


def supervise(command, timeout=900, grace=30, kill_grace=5):
    """
    Supervise one task tree; return only after cleanup or a reported cleanup failure.

    Run in a dedicated Linux interpreter without pre-existing children. SIGKILL
    and OOM of the supervisor cannot be handled. Output is streamed best-effort;
    blocked/broken sinks are bounded and reported by a nonzero exit status.
    """
    if any(
        not math.isfinite(value) or value <= 0 for value in (timeout, grace, kill_grace)
    ):
        raise ValueError("supervisor deadlines must be positive")
    if any(info[0] == os.getpid() for info in _process_table().values()):
        raise RuntimeError(
            "supervisor requires an interpreter without existing children"
        )
    previous = _subreaper(True)
    received = []
    old_handlers = {}
    output = _Output()
    process = None
    code = 1
    started = time.monotonic()

    def interrupted(sig, frame):
        if not received:
            received.append(sig)

    try:
        os.close(_pidfd_open(os.getpid()))
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, interrupted)
        directory = _cgroup_directory()
        initial = _snapshot(directory)
        output.send(f"[docs supervisor] START timeout={timeout}s grace={grace}s\n")
        output.send(f"[docs supervisor] cgroup_initial={json.dumps(initial)}\n")
        with tempfile.TemporaryFile() as log:
            offset = 0
            read_failed = False

            def stream():
                nonlocal offset, read_failed
                if read_failed:
                    return False
                try:
                    data = os.pread(log.fileno(), 65536, offset)
                except OSError as error:
                    read_failed = True
                    output.send(f"[docs supervisor] log read failed: {error}\n")
                    return False
                offset += len(data)
                output.send(data)
                return bool(data)

            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                reason = None
                stop_at = None
                kill_at = None
                while True:
                    now = time.monotonic()
                    root_code = process.poll()
                    owned = _owned_processes(process)
                    stream()
                    if reason is None:
                        if received:
                            reason, code = "external signal", 128 + received[0]
                            sig = received[0]
                        elif now - started >= timeout:
                            reason, code, sig = "deadline", 124, signal.SIGINT
                        elif root_code is not None:
                            code = root_code
                            if not owned:
                                break
                            reason, sig = (
                                "parent exited; cleaning descendants",
                                signal.SIGTERM,
                            )
                        if reason is not None:
                            output.send(
                                f"[docs supervisor] {reason}; signalling {sorted(owned)}\n"
                            )
                            _signal_owned(owned, sig)
                            stop_at = now + grace
                    if reason is not None:
                        if not owned and root_code is not None:
                            break
                        if kill_at is None and now >= stop_at:
                            output.send(
                                f"[docs supervisor] forced cleanup SIGKILL {sorted(owned)}\n"
                            )
                            if reason == "deadline" and root_code is None:
                                code = 137
                            kill_at = now + kill_grace
                        if kill_at is not None:
                            _signal_owned(owned, signal.SIGKILL)
                            if now >= kill_at:
                                break
                    time.sleep(0.02)
            except BaseException:
                output.send(traceback.format_exc())
                code = 1
            finally:
                if process is not None:
                    cleanup_at = time.monotonic() + kill_grace
                    while True:
                        process.poll()
                        remaining = _owned_processes(process)
                        if not remaining:
                            break
                        _signal_owned(remaining, signal.SIGKILL)
                        if time.monotonic() >= cleanup_at:
                            output.send(
                                f"[docs supervisor] CLEANUP FAILED live_pids={sorted(remaining)}\n"
                            )
                            code = code or 70
                            break
                        time.sleep(0.02)
                    process.poll()
                    _owned_processes(process)
                end = time.monotonic() + 0.5
                while time.monotonic() < end and stream():
                    pass
                if offset < os.fstat(log.fileno()).st_size:
                    read_failed = True
                    output.send(
                        "[docs supervisor] log drain budget exhausted; output incomplete\n"
                    )
        final = _snapshot(directory)
        output.send(f"[docs supervisor] cgroup_final={json.dumps(final)}\n")
        output.send(
            f"[docs supervisor] cgroup_event_delta={json.dumps(_event_delta(initial, final))}\n"
        )
        code = 128 - code if code < 0 else code
        if received:
            code = 128 + received[0]
        output.finish()
        if output.failed or read_failed:
            code = code or 74
        output.send(
            f"[docs supervisor] EXIT exit_code={code} elapsed={time.monotonic() - started:.3f}s log_delivery_failed={output.failed or read_failed}\n"
        )
        output.finish()
        return (code or 74) if output.failed else code
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        _subreaper(previous)
        output.close()


if __name__ == "__main__":
    runner = Path(__file__).with_name("trace_build.py")
    sys.exit(
        supervise(
            [sys.executable, "-u", str(runner), *sys.argv[1:]],
            timeout=float(os.environ.get("SCPY_DOCS_TIMEOUT", "900")),
        )
    )
