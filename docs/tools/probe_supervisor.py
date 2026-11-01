# ruff: noqa: T201,S603
"""Temporary RTD experiment: exercise supervision without importing the builder."""

import sys

from supervise_build import supervise


def main():
    """Expect an internal deadline, visible child output, and exit status 124."""
    print("[rtd probe] ENTER expected_exit=124", flush=True)
    child = (
        "import sys, time\n"
        "print('[rtd probe child] STDOUT before sleep', flush=True)\n"
        "print('[rtd probe child] STDERR before sleep', file=sys.stderr, flush=True)\n"
        "time.sleep(60)\n"
    )
    code = supervise(
        [sys.executable, "-u", "-c", child], timeout=5, grace=1, kill_grace=1
    )
    print(f"[rtd probe] RETURN supervisor_code={code}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
