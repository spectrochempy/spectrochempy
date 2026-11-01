"""Run the documentation builder with independent stderr diagnostics."""

import faulthandler
import multiprocessing
import os
import resource
import runpy
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pathlib import Path


def main():
    """Trace startup, periodic stacks, uncaught exceptions, and peak RSS."""
    started = time.monotonic()
    with os.fdopen(os.dup(2), "w", buffering=1) as diagnostic:
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
                print(
                    f"[docs runner] {package}={installed}", file=diagnostic, flush=True
                )
            for filename in (
                "/proc/meminfo",
                "/proc/self/status",
                "/proc/self/cgroup",
                "/sys/fs/cgroup/cpu.max",
                "/sys/fs/cgroup/memory.max",
                "/sys/fs/cgroup/memory.current",
                "/sys/fs/cgroup/memory.events",
            ):
                path = Path(filename)
                try:
                    content = path.read_text()
                except OSError as error:
                    content = str(error)
                print(
                    f"[docs runner] {filename}\n{content}", file=diagnostic, flush=True
                )
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
            faulthandler.cancel_dump_traceback_later()
            print(
                f"[docs runner] EXIT elapsed={time.monotonic() - started:.3f}s "
                f"peak_rss_kib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}",
                file=diagnostic,
                flush=True,
            )
            faulthandler.disable()


if __name__ == "__main__":
    main()
