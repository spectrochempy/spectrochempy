"""Measurement harness for the unchanged supervisor, never a documentation build."""
# ruff: noqa: PLC0415,S603,T201

import time

BODY_ENTERED = time.monotonic()

import os
import sys

SITE_ON_ENTRY = "site" in sys.modules
if evidence := os.environ.get("SCPY_STARTUP_EVIDENCE"):
    with open(evidence, "a", encoding="utf-8") as stream:
        stream.write(
            f'{{"stage":"body_enter","time":{BODY_ENTERED},"pid":{os.getpid()}}}\n'
        )

import argparse
import ctypes
import importlib.util
import json
import signal
import stat
import subprocess
import tempfile
from pathlib import Path

IMPORTS_FINISHED = time.monotonic()
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).resolve()
SUPERVISOR = SCRIPT.with_name("docs_build_diagnostics.py")


def load_supervisor():
    spec = importlib.util.spec_from_file_location(
        "measured_docs_supervisor", SUPERVISOR
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment():
    import site

    folders = [*site.getsitepackages(), site.getusersitepackages()]
    pth = []
    for folder in dict.fromkeys(folders):
        for path in sorted(Path(folder).glob("*.pth")):
            lines = path.read_text(errors="replace").splitlines()
            pth.append(
                {
                    "name": path.name,
                    "executable_import_lines": sum(
                        line.startswith(("import ", "import\t")) for line in lines
                    ),
                }
            )
    return {
        "python": sys.version.split()[0],
        "isolated": sys.flags.isolated,
        "no_site": sys.flags.no_site,
        "site_loaded_on_entry": SITE_ON_ENTRY,
        "user_site_enabled": site.ENABLE_USER_SITE,
        "sitecustomize_loaded": "sitecustomize" in sys.modules,
        "usercustomize_loaded": "usercustomize" in sys.modules,
        "distutils_startup_hook_loaded": "_distutils_hack" in sys.modules,
        "scientific_modules_loaded": {
            name: name in sys.modules for name in ("spectrochempy", "numpy", "IPython")
        },
        "pth_inventory_not_proof_of_execution": pth,
        "pythonpath_set": bool(os.environ.get("PYTHONPATH")),
        "pythonpath_entry_count": len(
            os.environ.get("PYTHONPATH", "").split(os.pathsep)
        )
        if os.environ.get("PYTHONPATH")
        else 0,
        "cwd": os.getcwd(),
        "stdout": os.readlink("/proc/self/fd/1"),
        "stderr": os.readlink("/proc/self/fd/2"),
    }


def actor(mode, directory):
    directory = Path(directory)
    if mode == "noevents":
        time.sleep(5)
    if mode == "leaf":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    regular = all(stat.S_ISREG(os.fstat(fd).st_mode) for fd in (1, 2))
    identity = {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "pgid": os.getpgrp(),
        "sid": os.getsid(0),
        "regular_output": regular,
    }
    temporary = directory / f"{mode}.tmp"
    temporary.write_text(json.dumps(identity))
    temporary.replace(directory / f"{mode}.json")
    sys.path.insert(0, str(ROOT))
    from docs.tools.lifecycle_trace import emit

    emit(
        "startup.actor." + mode,
        "START",
        span="actor:" + mode,
        document="controlled-" + mode,
    )
    if mode in ("detached", "orphan"):
        subprocess.Popen(
            [sys.executable, "-u", str(SCRIPT), "actor", "leaf", str(directory)],
            start_new_session=True,
        )
        ready = time.monotonic() + 1.5
        while not (directory / "leaf.json").exists():
            if time.monotonic() >= ready:
                raise TimeoutError("leaf readiness timeout")
            time.sleep(0.005)
    if mode in ("success", "orphan"):
        emit("startup.actor." + mode, "END", span="actor:" + mode)
    else:
        time.sleep(5)


def controller(args):
    stages = [{"stage": "stdlib_imports_finished", "time": IMPORTS_FINISHED}]
    seen = set()
    targets = {ctypes.CDLL.__init__.__code__: "ctypes_cdll"}
    module = None

    def persist(row):
        with Path(os.environ["SCPY_STARTUP_EVIDENCE"]).open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps({**row, "pid": os.getpid()}) + "\n")

    for row in stages:
        persist(row)

    def mark(stage):
        if stage not in seen:
            seen.add(stage)
            row = {"stage": stage, "time": time.monotonic()}
            stages.append(row)
            persist(row)

    def profile(frame, event, arg):
        if frame.f_code in targets and event in ("call", "return"):
            mark(targets[frame.f_code] + "." + event)
        if (
            module is not None
            and frame.f_code is module.supervise.__code__
            and event == "c_return"
            and arg is time.monotonic
        ):
            mark("deadline_clock_initialized_observed")
        if (
            module is not None
            and frame.f_code is module._process_table.__code__
            and event == "return"
            and isinstance(arg, dict)
            and "preflight_children" not in seen
        ):
            seen.add("preflight_children")
            persist(
                {
                    "stage": "preflight_children",
                    "time": time.monotonic(),
                    "pids": [
                        pid for pid, info in arg.items() if info[0] == os.getpid()
                    ],
                }
            )

    prior_profile = sys.getprofile()
    code = 1
    error = None
    mark("measurement_environment.call")
    before = environment()
    mark("measurement_environment.return")
    persist(
        {
            "stage": "environment_snapshot",
            "time": time.monotonic(),
            "environment": before,
        }
    )
    sys.setprofile(profile)
    try:
        mark("supervisor_import.call")
        module = load_supervisor()
        mark("supervisor_import.return")
        functions = {
            "main": module.main,
            "supervise": module.supervise,
            "subreaper": module._subreaper,
            "pidfd": module._pidfd_open,
            "process_table": module._process_table,
            "popen": subprocess.Popen.__init__,
            "cleanup": module._cleanup,
            "report": module._finish_report,
            "temporary_file": tempfile.TemporaryFile,
            "temporary_directory": tempfile.TemporaryDirectory.__init__,
            "output_setup": module._Output.__init__,
            "output_flush": module._Output.flush,
            "output_close": module._Output.close,
        }
        targets.update({fn.__code__: name for name, fn in functions.items()})

        def delay_once(owner, name, stage, predicate=lambda number: number == 1):
            original = getattr(owner, name)
            counter = 0

            def delayed(*values, **options):
                nonlocal counter
                counter += 1
                if predicate(counter):
                    mark(stage + ".call")
                    time.sleep(args.delay)
                    mark(stage + ".return")
                return original(*values, **options)

            setattr(owner, name, delayed)

        if args.scenario == "pre_scan":
            delay_once(module, "_process_table", "injected_pre_clock_scan")
        elif args.scenario == "loop_scan":
            delay_once(module, "_owned_processes", "injected_loop_scan")
        elif args.scenario == "launch":
            delay_once(subprocess, "Popen", "injected_launch")
        elif args.scenario == "launch_return":
            original_popen = subprocess.Popen

            def delayed_return(*values, **options):
                process = original_popen(*values, **options)
                mark("injected_launch_return.call")
                time.sleep(args.delay)
                mark("injected_launch_return.return")
                return process

            subprocess.Popen = delayed_return
        elif args.scenario == "temporary_file":
            delay_once(tempfile, "TemporaryFile", "injected_temporary_file")
        elif args.scenario == "output":
            delay_once(module._Output, "flush", "injected_output")

        directory = Path(args.directory)
        command = [
            sys.executable,
            "-u",
            str(SCRIPT),
            "actor",
            args.actor,
            str(directory),
        ]
        sys.argv = [
            str(SUPERVISOR),
            "--timeout",
            str(args.timeout),
            "--grace",
            ".1",
            "--kill-grace",
            ".2",
            "--",
            *command,
        ]
        code = module.main()
    except BaseException as exception:
        error = type(exception).__name__
    finally:
        sys.setprofile(prior_profile)
        evidence = Path(os.environ["SCPY_STARTUP_EVIDENCE"])
        with evidence.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "stage": "controller_exit",
                        "time": time.monotonic(),
                        "pid": os.getpid(),
                        "code": code,
                        "exception_type": error,
                        "environment": before,
                    }
                )
                + "\n"
            )
    return code


def measure(
    module,
    isolated,
    directory,
    timeout,
    scenario="baseline",
    delay=0.6,
    actor_mode="detached",
    extra_env=None,
):
    directory = Path(directory)
    directory.mkdir()
    evidence = directory / "timeline.jsonl"
    env = {
        **os.environ,
        **(extra_env or {}),
        "SCPY_STARTUP_EVIDENCE": str(evidence),
        "SCPY_DOCS_TRACE": "1",
    }
    command = [
        sys.executable,
        *(["-I", "-S"] if isolated else []),
        "-u",
        str(SCRIPT),
        "controller",
        "--directory",
        str(directory),
        "--timeout",
        str(timeout),
        "--scenario",
        scenario,
        "--delay",
        str(delay),
        "--actor",
        actor_mode,
    ]
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    guarded = False
    output = ""
    capture_complete = None
    try:
        try:
            output, _ = process.communicate(timeout=8)
            capture_complete = time.monotonic()
        except subprocess.TimeoutExpired:
            guarded = True
        finally:
            live, errors = module._cleanup(process, 1)
        if guarded:
            output, _ = process.communicate(timeout=2)
            capture_complete = time.monotonic()
    finally:
        process.stdout.close()
    rows = []
    partial_evidence = False
    if evidence.exists():
        for line in evidence.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                partial_evidence = True
    controller_rows = [row for row in rows if row["pid"] == process.pid]
    final = next(
        (
            json.loads(line.split("] ", 1)[1])
            for line in output.splitlines()
            if line.startswith("[docs recovery FINAL] ")
        ),
        None,
    )
    actors = [json.loads(path.read_text()) for path in directory.glob("*.json")]
    remaining = [row["pid"] for row in actors if Path(f"/proc/{row['pid']}").exists()]
    return {
        "isolated": isolated,
        "scenario": scenario,
        "actor_mode": actor_mode,
        "launched": started,
        "elapsed": time.monotonic() - started,
        "controller_pid": process.pid,
        "controller_code": process.returncode,
        "observer_guard_expired": guarded,
        "observer_live_pids": live,
        "observer_cleanup_errors": errors,
        "controller_timeline": controller_rows,
        "actor_body_entries": [
            row
            for row in rows
            if row["pid"] != process.pid and row["stage"] == "body_enter"
        ],
        "partial_evidence": partial_evidence,
        "final_report": final,
        "actors": actors,
        "remaining_actor_pids": remaining,
        "capture_complete_observed": capture_complete,
    }


def compare(timeout=3, output_path=None):
    if not 0.1 <= timeout <= 3:
        raise ValueError("comparison timeout must be between 0.1 and 3 seconds")
    module = load_supervisor()
    if any(row[0] == os.getpid() for row in module._process_table().values()):
        raise RuntimeError("measurement harness requires no pre-existing children")
    previous = module._subreaper(True)
    try:
        with tempfile.TemporaryDirectory(prefix="scpy-startup-comparison-") as scratch:
            results = [
                measure(module, isolated, Path(scratch) / str(isolated), timeout)
                for isolated in (False, True)
            ]
        report = {
            "probe_id": "scpy-rtd-installed-startup",
            "kind": "startup-comparison",
            "internal_timeout": timeout,
            "results": results,
        }
        if output_path:
            Path(output_path).write_text(json.dumps(report, indent=2))
        valid = all(
            row["controller_code"] == 124
            and row["final_report"]
            and row["final_report"]["reason"] == "deadline"
            and row["final_report"]["cleanup_complete"]
            and row["final_report"]["trace_complete"]
            and row["final_report"]["active_count"] == 2
            and not row["final_report"]["log_delivery_failed"]
            and not row["remaining_actor_pids"]
            and not row["observer_live_pids"]
            and not row["observer_cleanup_errors"]
            and not row["observer_guard_expired"]
            and not row["partial_evidence"]
            for row in results
        )
        if valid and (output := os.environ.get("READTHEDOCS_OUTPUT")):
            html = Path(output) / "html"
            html.mkdir(parents=True, exist_ok=True)
            (html / "index.html").write_text(
                "<!doctype html><title>Startup infrastructure probe</title><p>No documentation was built.</p>"
            )
        output = module._Output()
        try:
            output.send(json.dumps(report) + "\n")
            output.finish()
            return (0 if valid else 1) if not output.failed else 74
        finally:
            output.close()
    finally:
        module._subreaper(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    run = subparsers.add_parser("compare")
    run.add_argument("--timeout", type=float, default=3)
    run.add_argument("--output")
    timed = subparsers.add_parser("controller")
    timed.add_argument("--directory", required=True)
    timed.add_argument("--timeout", type=float, default=0.4)
    timed.add_argument("--delay", type=float, default=0.6)
    timed.add_argument(
        "--scenario",
        choices=(
            "baseline",
            "pre_scan",
            "loop_scan",
            "launch",
            "launch_return",
            "temporary_file",
            "output",
        ),
        default="baseline",
    )
    timed.add_argument(
        "--actor",
        choices=("success", "block", "noevents", "detached", "orphan"),
        default="detached",
    )
    child = subparsers.add_parser("actor")
    child.add_argument(
        "actor", choices=("success", "block", "noevents", "detached", "orphan", "leaf")
    )
    child.add_argument("directory")
    args = parser.parse_args()
    if args.mode == "actor":
        actor(args.actor, args.directory)
        return 0
    if args.mode == "controller":
        if "SCPY_STARTUP_EVIDENCE" not in os.environ or not 0 <= args.delay <= 2:
            parser.error(
                "controller needs an evidence path and a finite delay in [0, 2]"
            )
        return controller(args)
    return compare(args.timeout, args.output)


if __name__ == "__main__":
    sys.exit(main())
