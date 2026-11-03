"""Bounded Linux documentation command; ownership primitives from ef092a238."""
# ruff: noqa: S603

import argparse
import ctypes
import errno
import fcntl
import json
import math
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections import deque
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
        try:
            self.flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
            os.set_blocking(self.fd, False)
        except BaseException:
            os.close(self.fd)
            raise
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
        try:
            fcntl.fcntl(self.fd, fcntl.F_SETFL, self.flags)
        finally:
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


def _cleanup(process, budget):
    deadline = time.monotonic() + budget
    errors = set()
    while True:
        process.poll()
        remaining = _owned_processes(process)
        if not remaining and process.returncode is not None:
            return [], sorted(errors)
        try:
            _signal_owned(remaining, signal.SIGKILL)
        except OSError as error:
            errors.add(type(error).__name__)
        if time.monotonic() >= deadline:
            live = set(_owned_processes(process))
            if process.poll() is None:
                live.add(process.pid)
            return sorted(live), sorted(errors)
        time.sleep(0.02)


def _trace_report(path):
    """Parse only bounded lifecycle metadata, never raw build output or cell text."""
    allowed = {
        "utc",
        "monotonic",
        "pid",
        "phase",
        "status",
        "span",
        "document",
        "cell_index",
        "kernel_pid",
        "duration",
        "outcome",
        "exception_type",
        "returncode",
    }
    active = {}
    recent = deque(maxlen=20)
    invalid = 0
    count = 0
    complete = True
    started = time.monotonic()
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("trace is not a regular file")
        while True:
            line = stream.readline(16385)
            if not line:
                break
            if stream.tell() > 33554432 or time.monotonic() - started > 2:
                complete = False
                break
            if not line.endswith(b"\n") or len(line) > 16384:
                invalid += 1
                complete = False
                break
            try:
                row = json.loads(line.removeprefix(b"[docs lifecycle] "))
                safe = {
                    key: (value[:512] if isinstance(value, str) else value)
                    for key, value in row.items()
                    if key in allowed
                    and isinstance(value, (str, int, float, bool, type(None)))
                }
            except (ValueError, AttributeError):
                invalid += 1
                continue
            count += 1
            recent.append(safe)
            token = safe.get("span")
            if token and safe.get("status") == "START":
                if len(active) >= 256:
                    complete = False
                else:
                    active[token] = safe
            elif token and safe.get("status") in ("END", "INCOMPLETE"):
                active.pop(token, None)
    return {
        "events": count,
        "complete": complete and not invalid,
        "invalid": invalid,
        "active": list(active.values()),
        "recent": list(recent),
    }


def supervise(command, timeout=600, grace=15, kill_grace=5, stream_limit=1048576):
    """Run once in a child-free Linux interpreter, then report before returning."""
    if (
        any(
            not math.isfinite(value) or value <= 0
            for value in (timeout, grace, kill_grace)
        )
        or stream_limit < 0
    ):
        raise ValueError(
            "positive finite deadlines and nonnegative stream limit required"
        )
    if any(info[0] == os.getpid() for info in _process_table().values()):
        raise RuntimeError(
            "supervisor requires an interpreter without existing children"
        )
    previous = None
    output = None
    handlers = {}
    received = []
    started = time.monotonic()
    process = None
    code = 1
    reason = "setup failure"
    live = []
    cleanup_errors = []
    forced = False
    trace_error = None
    truncated = None
    stream_incomplete = False
    cleanup_attempted = False
    diagnostics = {
        "events": 0,
        "complete": False,
        "invalid": 0,
        "active": [],
        "recent": [],
    }

    def interrupted(sig, frame):
        if not received:
            received.append(sig)

    try:
        output = _Output()
        previous = _subreaper(True)
        os.close(_pidfd_open(os.getpid()))
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, interrupted)
        with tempfile.TemporaryDirectory(prefix="scpy-docs-diagnostics-") as directory:
            trace = Path(directory) / "lifecycle.jsonl"
            trace.touch()
            env = {**os.environ, "SCPY_DOCS_TRACE_FILE": str(trace)}
            with tempfile.TemporaryFile() as log:
                offset = 0
                reason = None
                stop_at = None
                kill_at = None
                try:
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env=env,
                    )
                    output.send("[docs recovery] START\n")
                    while True:
                        now = time.monotonic()
                        root_code = process.poll()
                        owned = _owned_processes(process)
                        if offset < stream_limit:
                            chunk = os.pread(
                                log.fileno(), min(65536, stream_limit - offset), offset
                            )
                            offset += len(chunk)
                            output.send(chunk)
                        if reason is None:
                            if received:
                                reason, code, sig = (
                                    "external signal",
                                    128 + received[0],
                                    received[0],
                                )
                            elif now - started >= timeout:
                                reason, code, sig = "deadline", 124, signal.SIGTERM
                            elif root_code is not None:
                                code = 128 - root_code if root_code < 0 else root_code
                                if not owned:
                                    reason = "command exited"
                                    break
                                reason, sig = (
                                    "parent exited; cleaning descendants",
                                    signal.SIGTERM,
                                )
                            if reason is not None:
                                stop_at = now + grace
                                _signal_owned(owned, sig)
                        if reason is not None:
                            if not owned and root_code is not None:
                                break
                            if kill_at is None and now >= stop_at:
                                forced = True
                                kill_at = now + kill_grace
                            if kill_at is not None:
                                _signal_owned(owned, signal.SIGKILL)
                                if now >= kill_at:
                                    break
                        time.sleep(0.02)
                except BaseException as error:
                    code = 1
                    reason = f"supervisor error: {type(error).__name__}"
                finally:
                    if process is not None:
                        cleanup_attempted = True
                        try:
                            live, cleanup_errors = _cleanup(process, kill_grace)
                        except Exception as error:
                            cleanup_errors = [type(error).__name__]
                            with suppress(Exception):
                                live = sorted(_owned_processes(process))
                            if process.poll() is None and process.pid not in live:
                                live.append(process.pid)
                    try:
                        drain_until = time.monotonic() + 0.5
                        while offset < stream_limit and time.monotonic() < drain_until:
                            chunk = os.pread(
                                log.fileno(), min(stream_limit - offset, 65536), offset
                            )
                            if not chunk:
                                break
                            offset += len(chunk)
                            output.send(chunk)
                        size = os.fstat(log.fileno()).st_size
                        truncated = size > stream_limit
                        stream_incomplete = offset < min(size, stream_limit)
                        output.failed |= stream_incomplete
                    except OSError:
                        output.failed = True
            try:
                diagnostics = _trace_report(trace)
            except Exception as error:
                trace_error = type(error).__name__
    except BaseException as error:
        code = code or 1
        reason = f"setup or recovery error: {type(error).__name__}"
        trace_error = type(error).__name__
    finally:
        if process is not None and not cleanup_attempted:
            try:
                live, cleanup_errors = _cleanup(process, kill_grace)
            except Exception as error:
                cleanup_errors = [type(error).__name__]
                if process.poll() is None:
                    live = [process.pid]
        if live or cleanup_errors:
            code = 70
        elif received:
            code = 128 + received[0]
        elif trace_error or not diagnostics["complete"]:
            code = code or 74
        try:
            if output is not None:
                code = _finish_report(
                    output,
                    process,
                    code,
                    reason,
                    started,
                    live,
                    cleanup_errors,
                    forced,
                    truncated,
                    stream_incomplete,
                    trace_error,
                    diagnostics,
                )
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            if previous is not None:
                _subreaper(previous)
            if output is not None:
                output.close()
    return code


def _finish_report(
    output,
    process,
    code,
    reason,
    started,
    live,
    cleanup_errors,
    forced,
    truncated,
    stream_incomplete,
    trace_error,
    diagnostics,
):
    output.finish()
    if output.pending:
        output.pending = b""
    output.send("\n[docs recovery REPORT]\n")
    for label, rows in (
        ("active", diagnostics["active"][-32:]),
        ("recent", diagnostics["recent"]),
    ):
        for row in rows:
            output.send(f"[docs recovery {label}] {json.dumps(row)}\n")
    output.finish()
    if output.failed:
        code = code or 74
    report = {
        "exit_code": code,
        "reason": reason,
        "elapsed": time.monotonic() - started,
        "root_pid": process.pid if process else None,
        "root_returncode": process.poll() if process else None,
        "cleanup_complete": not live and not cleanup_errors,
        "live_pids": live,
        "cleanup_errors": cleanup_errors,
        "forced_kill": forced,
        "stream_truncated": truncated,
        "stream_incomplete": stream_incomplete,
        "trace_enabled": os.environ.get("SCPY_DOCS_TRACE") == "1",
        "trace_complete": diagnostics["complete"],
        "trace_error": trace_error,
        "events": diagnostics["events"],
        "active_count": len(diagnostics["active"]),
        "active_omitted": max(0, len(diagnostics["active"]) - 32),
        "log_delivery_failed": output.failed,
    }
    output.send(f"[docs recovery FINAL] {json.dumps(report)}\n")
    output.finish()
    return (code or 74) if output.failed else code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--grace", type=float, default=15)
    parser.add_argument("--kill-grace", type=float, default=5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        root = Path(__file__).resolve().parents[3]
        command = [
            sys.executable,
            "-u",
            str(root / "docs" / "make.py"),
            "html",
            "--warning-is-error",
            "-j",
            "auto",
        ]
    return supervise(command, args.timeout, args.grace, args.kill_grace)


if __name__ == "__main__":
    sys.exit(main())
