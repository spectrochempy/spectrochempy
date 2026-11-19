"""Uncaptured phase and stack diagnostics for documentation command-line tools."""

import faulthandler
import sys
import time
import traceback
from contextlib import contextmanager


@contextmanager
def _phase(name):
    started = time.monotonic()
    stream = sys.__stderr__
    print(f"[docs-build] START {name}", file=stream, flush=True)
    outcome = "OK"
    try:
        yield
    except BaseException as error:
        if not (isinstance(error, SystemExit) and error.code in (None, 0)):
            outcome = "FAILED"
            traceback.print_exc(file=stream)
        raise
    finally:
        print(
            f"[docs-build] {outcome} {name} ({time.monotonic() - started:.1f}s)",
            file=stream,
            flush=True,
        )


@contextmanager
def _diagnostics():
    faulthandler.dump_traceback_later(120, repeat=True, file=sys.__stderr__)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
