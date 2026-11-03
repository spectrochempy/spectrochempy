"""Candidate CLI exception boundary; no process ownership or cleanup is added."""

import fcntl
import importlib.util
import json
import os
import sys
import time
from contextlib import suppress
from pathlib import Path


def load_supervisor():
    path = Path(__file__).with_name("docs_build_diagnostics.py")
    spec = importlib.util.spec_from_file_location("entry_docs_supervisor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report_entry_failure(phase, error):
    """Best-effort bounded output after this Python entry boundary is reached."""
    payload = (
        "\n[docs controller FAILURE] "
        + json.dumps(
            {
                "exit_code": 1,
                "phase": phase,
                "exception_type": type(error).__name__,
                "pid": os.getpid(),
                "cleanup_status": "not inferred from escaped exception",
                "adapter_owns_children": False,
            }
        )
        + "\n"
    ).encode()
    fd = None
    flags = None
    try:
        fd = os.dup(1)
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        os.set_blocking(fd, False)
        deadline = time.monotonic() + 0.2
        while payload and time.monotonic() < deadline:
            try:
                written = os.write(fd, payload)
            except BlockingIOError:
                time.sleep(0.01)
                continue
            if not written:
                break
            payload = payload[written:]
    except OSError:
        pass
    finally:
        if fd is not None:
            with suppress(OSError):
                if flags is not None:
                    fcntl.fcntl(fd, fcntl.F_SETFL, flags)
            with suppress(OSError):
                os.close(fd)


def run_main():
    phase = "loading supervisor"
    try:
        module = load_supervisor()
        phase = "calling supervisor main"
        return module.main()
    except Exception as error:
        report_entry_failure(phase, error)
        return 1


def main():
    if "--" not in sys.argv or not sys.argv[sys.argv.index("--") + 1 :]:
        report_entry_failure("explicit child command required", ValueError())
        return 1
    return run_main()


if __name__ == "__main__":
    sys.exit(main())
