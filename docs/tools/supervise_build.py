# ruff: noqa: T201,S603
"""Bound the diagnostic build and replay output without inherited capture pipes."""

import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path


def _signal_group(process, sig):
    with suppress(ProcessLookupError):
        os.killpg(process.pid, sig)


def supervise(command, timeout=900, grace=30):
    """Capture to a regular file, interrupt on deadline, and preserve exit status."""
    started = time.monotonic()
    print(f"[docs supervisor] START timeout={timeout}s grace={grace}s", flush=True)
    with (
        tempfile.TemporaryFile() as log,
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as process,
    ):
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print("[docs supervisor] deadline reached; sending SIGINT", flush=True)
            _signal_group(process, signal.SIGINT)
            try:
                process.wait(timeout=grace)
                code = 124
            except subprocess.TimeoutExpired:
                print("[docs supervisor] grace expired; sending SIGKILL", flush=True)
                _signal_group(process, signal.SIGKILL)
                process.wait()
                code = 137
        finally:
            if process.poll() is None:
                _signal_group(process, signal.SIGKILL)
                process.wait()
            size = os.fstat(log.fileno()).st_size
            log.seek(0)
            while size:
                chunk = log.read(min(size, 65536))
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                size -= len(chunk)
            sys.stdout.flush()
    code = 128 - code if code < 0 else code
    print(
        f"[docs supervisor] EXIT exit_code={code} "
        f"elapsed={time.monotonic() - started:.3f}s",
        flush=True,
    )
    return code


if __name__ == "__main__":
    runner = Path(__file__).with_name("trace_build.py")
    sys.exit(supervise([sys.executable, "-u", str(runner), *sys.argv[1:]]))
