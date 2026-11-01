"""Run the documentation builder with independent stderr diagnostics."""

import faulthandler
import multiprocessing
import os
import runpy
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None

_DIAGNOSTIC_STREAMS = []


def main():
    """Keep diagnostics alive until interpreter shutdown, including atexit."""
    started = time.monotonic()
    diagnostic = os.fdopen(os.dup(2), "w", buffering=1)
    _DIAGNOSTIC_STREAMS.append(diagnostic)
    faulthandler.enable(file=diagnostic, all_threads=True)
    faulthandler.dump_traceback_later(120, repeat=True, file=diagnostic)
    try:
        print(
            f"[docs runner] START {time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
            f"pid={os.getpid()} python={sys.version} "
            f"cpu_count={multiprocessing.cpu_count()} args={sys.argv[1:]!r}",
            file=diagnostic,
            flush=True,
        )
        for package in (
            "ipython",
            "jupyterlab",
            "notebook",
            "sphinx",
            "numpy",
            "scipy",
        ):
            try:
                installed = version(package)
            except PackageNotFoundError:
                installed = "not installed"
            print(f"[docs runner] {package}={installed}", file=diagnostic, flush=True)
        docs = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(docs))
        sys.argv[0] = str(docs / "make.py")
        runpy.run_path(sys.argv[0], run_name="__main__")
    except SystemExit as error:
        print(
            f"[docs runner] SystemExit code={error.code!r}",
            file=diagnostic,
            flush=True,
        )
        raise
    except BaseException:
        traceback.print_exc(file=diagnostic)
        diagnostic.flush()
        raise
    finally:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None
        if rss is not None and sys.platform == "darwin":
            rss /= 1024
        print(
            f"[docs runner] BUILDER_RETURN elapsed={time.monotonic() - started:.3f}s "
            f"rss_self_peak_kib={rss if rss is not None else 'unavailable'} "
            "(runner only, excludes workers and kernels; shutdown not complete)",
            file=diagnostic,
            flush=True,
        )


if __name__ == "__main__":
    main()
