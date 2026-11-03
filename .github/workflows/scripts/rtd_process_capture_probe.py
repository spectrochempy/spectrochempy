"""Short Linux-only capture experiment; no documentation or scientific imports."""
# ruff: noqa: S603,T201

import hashlib
import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import types
from datetime import UTC
from datetime import datetime
from pathlib import Path

REVISION = "ef092a238d3a968bf21f9129e2e3934c3d1a07ce"
DIGEST = "b911dd0454d6a5f55f6a57c54901b9a66213f0b3eca0fa39cf10eff6ba9bd192"
SCRIPT = Path(__file__).resolve()


def event(kind, **fields):
    return {
        "event": kind,
        "utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "monotonic": time.monotonic(),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "pgid": os.getpgrp(),
        "sid": os.getsid(0),
        **fields,
    }


def record(path, kind, **fields):
    data = (json.dumps(event(kind, **fields)) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.write(fd, data) != len(data):
            raise OSError("partial event record write")
    finally:
        os.close(fd)


def historical():
    """Load the hash-verified historical Git blob without modifying its code."""
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git is required to read the historical supervisor")
    source = subprocess.run(
        [git, "--no-pager", "show", f"{REVISION}:docs/tools/supervise_build.py"],
        cwd=SCRIPT.parents[3],
        capture_output=True,
        check=True,
        timeout=5,
    ).stdout
    if hashlib.sha256(source).hexdigest() != DIGEST:
        raise RuntimeError("historical supervisor digest mismatch")
    module = types.ModuleType("historical_rtd_supervisor")
    exec(  # noqa: S102
        compile(source, f"{REVISION}:docs/tools/supervise_build.py", "exec"),
        module.__dict__,
    )
    return module


def actor(mode, path, seconds):
    if mode == "child":

        def interrupted(sig, frame):
            record(path, "child_end_marker", declared_status=128 + sig, reason="signal")
            sys.exit(128 + sig)

        signal.signal(signal.SIGTERM, interrupted)
        record(
            path,
            "child_start",
            stdout=os.readlink("/proc/self/fd/1"),
            stderr=os.readlink("/proc/self/fd/2"),
        )
        time.sleep(seconds)
        record(path, "child_end_marker", declared_status=0, reason="normal")
        return 0
    record(path, "main_start")
    command = [sys.executable, "-u", str(SCRIPT), "child", str(path), str(seconds)]
    if mode == "B":
        with path.with_suffix(".output").open("wb") as output:
            process = subprocess.Popen(
                command, start_new_session=True, stdout=output, stderr=output
            )
    else:
        process = subprocess.Popen(command, start_new_session=True)
    ready_by = time.monotonic() + 2
    while '"child_start"' not in path.read_text():
        if time.monotonic() >= ready_by:
            raise TimeoutError("descendant did not become ready")
        time.sleep(0.005)
    record(path, "main_end_marker", child_pid=process.pid)
    return 0


def owned(module):
    table = module._process_table()
    parents = {os.getpid()}
    result = {}
    while children := {
        pid: info
        for pid, info in table.items()
        if info[0] in parents and pid not in result
    }:
        result.update(children)
        parents = set(children)
    return result


def reap(module, process, statuses):
    process.poll()
    for pid in owned(module):
        if pid == process.pid:
            continue
        try:
            waited, status = os.waitpid(pid, os.WNOHANG)
            if waited:
                statuses[pid] = {
                    "returncode": os.waitstatus_to_exitcode(status),
                    "observed": time.monotonic(),
                }
        except ChildProcessError:
            pass


def cleanup(module, process, statuses):
    for sig, budget in ((signal.SIGTERM, 0.5), (signal.SIGKILL, 1.5)):
        deadline = time.monotonic() + budget
        while True:
            reap(module, process, statuses)
            remaining = owned(module)
            if not remaining:
                return
            module._signal_owned(remaining, sig)
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)
    raise RuntimeError(f"cleanup failed for owned PIDs: {sorted(owned(module))}")


def scenario(module, mode, through_supervisor, seconds, deadline, interrupted):
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="scpy-rtd-capture-") as directory:
        path = Path(directory) / "events.jsonl"
        command = [sys.executable, "-u", str(SCRIPT), mode, str(path), str(seconds)]
        if through_supervisor:
            command[3] = "supervised-" + mode
        process = subprocess.Popen(
            command,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        statuses = {}
        eof = None
        exit_time = None
        main_observed = None
        captured = bytearray()
        selector = None
        try:
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                reap(module, process, statuses)
                now = time.monotonic()
                if process.returncode is not None and exit_time is None:
                    exit_time = now
                if main_observed is None and path.exists():
                    identities = [
                        json.loads(line) for line in path.read_text().splitlines()
                    ]
                    main = next(
                        (item for item in identities if item["event"] == "main_start"),
                        None,
                    )
                    if main:
                        info = module._process_table().get(main["pid"])
                        if info is None or info[1] == "Z":
                            main_observed = time.monotonic()
                for key, _ in selector.select(0.01):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if data:
                        if len(captured) + len(data) > 1048576:
                            raise RuntimeError("probe output limit exceeded")
                        captured.extend(data)
                    else:
                        eof = time.monotonic()
                        selector.unregister(key.fileobj)
                if eof is not None and exit_time is not None and not owned(module):
                    break
                if interrupted or now >= min(deadline, started + 12):
                    raise TimeoutError(
                        "bounded probe deadline or external interruption"
                    )
            records = [json.loads(line) for line in path.read_text().splitlines()]
            markers = {item["event"]: item for item in records}
            main = markers["main_start"]
            child = markers["child_start"]
            end = markers["child_end_marker"]
            if (
                process.returncode != 0
                or child["sid"] != child["pid"]
                or child["pgid"] != child["pid"]
            ):
                raise RuntimeError(
                    "unexpected process status or detached-session identity"
                )
            if not through_supervisor:
                main_observed = exit_time
            result = event(
                "scenario",
                mode=mode,
                capture="historical" if through_supervisor else "direct",
                started=started,
                elapsed=time.monotonic() - started,
                main_end_marker=markers["main_end_marker"]["monotonic"],
                main_exit_observed=main_observed,
                capture_root_exit_observed=exit_time,
                child_end_marker=end["monotonic"],
                child_declared_status=end["declared_status"],
                child_reaped=statuses.get(child["pid"]),
                eof=eof,
                returncode=process.returncode,
                main_identity=main,
                child_identity=child,
                survivors=[],
                supervisor_output=captured.decode(errors="replace")
                if through_supervisor
                else "",
            )
            if not through_supervisor:
                if (
                    end["declared_status"] != 0
                    or statuses.get(child["pid"], {}).get("returncode") != 0
                ):
                    raise RuntimeError("direct descendant did not terminate normally")
                if mode == "A" and not exit_time < eof:
                    raise RuntimeError("inherited-descriptor ordering differs")
                if mode == "B" and not eof < end["monotonic"]:
                    raise RuntimeError("redirected-descriptor ordering differs")
            return result
        finally:
            try:
                if selector is not None:
                    selector.close()
            finally:
                try:
                    process.stdout.close()
                finally:
                    cleanup(module, process, statuses)


def run(seconds=5):
    if not sys.platform.startswith("linux") or not 0.1 <= seconds <= 5:
        raise ValueError("Linux and descendant duration in [0.1, 5] seconds required")
    module = historical()
    if any(info[0] == os.getpid() for info in module._process_table().values()):
        raise RuntimeError("probe requires a dedicated interpreter without children")
    previous = module._subreaper(True)
    interrupted = []
    old = {}
    started = time.monotonic()
    results = []
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old[sig] = signal.signal(sig, lambda sig, frame: interrupted.append(sig))
        print(
            json.dumps(
                event(
                    "probe_start",
                    supervisor_revision=REVISION,
                    supervisor_sha256=DIGEST,
                )
            ),
            flush=True,
        )
        for through in (False, True):
            for mode in ("A", "B"):
                result = scenario(
                    module, mode, through, seconds, started + 45, interrupted
                )
                results.append(result)
                print(json.dumps(result), flush=True)
        if owned(module):
            raise RuntimeError("probe descendants remain")
        if output := os.environ.get("READTHEDOCS_OUTPUT"):
            html = Path(output) / "html"
            html.mkdir(parents=True, exist_ok=True)
            (html / "index.html").write_text(
                "<!doctype html><title>RTD infrastructure probe</title><p>No scientific documentation was built.</p>",
                encoding="utf-8",
            )
        print(
            json.dumps(
                event(
                    "probe_end",
                    elapsed=time.monotonic() - started,
                    returncode=0,
                    survivors=[],
                )
            ),
            flush=True,
        )
        return results
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
        module._subreaper(previous)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(float(sys.argv[2]) if len(sys.argv) > 2 else 5)
    elif sys.argv[1].startswith("supervised-"):
        mode = sys.argv[1].removeprefix("supervised-")
        sys.exit(
            historical().supervise(
                [sys.executable, "-u", str(SCRIPT), mode, *sys.argv[2:]],
                timeout=8,
                grace=1,
                kill_grace=1,
            )
        )
    else:
        sys.exit(actor(sys.argv[1], Path(sys.argv[2]), float(sys.argv[3])))
