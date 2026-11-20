#!/usr/bin/env python3
"""Print lightweight CI diagnostics for SpectroChemPy jobs."""

# ruff: noqa: PLC0415, T201

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
import platform
import re
import sys
from pathlib import Path

OPTIONAL_DEPENDENCIES = [
    "cantera",
    "matplotlib",
    "nmrglue",
    "numpy",
    "osqp",
    "pandas",
    "pint",
    "quadprog",
    "scipy",
    "sklearn",
    "traitlets",
]


def _discover_official_plugins() -> list[str]:
    """Return official plugin distribution names by checking [tool.spectrochempy]."""
    plugins_dir = Path("plugins")
    if not plugins_dir.is_dir():
        return []

    results: list[str] = []
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    for pyproject in sorted(plugins_dir.glob("spectrochempy-*/pyproject.toml")):
        try:
            data = tomllib.loads(pyproject.read_text())
            tool_sc = data.get("tool", {}).get("spectrochempy", {})
            if tool_sc.get("official-plugin") is True:
                results.append(pyproject.parent.name)
        except Exception as exc:
            print(f"Warning: could not read {pyproject}: {exc}", file=sys.stderr)
            continue
    return results


OFFICIAL_PLUGIN_DISTRIBUTIONS = _discover_official_plugins()

PYTEST_STATUSES = [
    "passed",
    "failed",
    "errors",
    "skipped",
    "xfailed",
    "xpassed",
    "deselected",
    "warnings",
]


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _module_status(name: str) -> str:
    if importlib.util.find_spec(name) is None:
        return "missing"
    version = _distribution_version(name)
    return f"installed ({version})" if version else "installed"


def _entry_points() -> list[str]:
    try:
        eps = metadata.entry_points(group="spectrochempy.plugins")
    except TypeError:
        eps = metadata.entry_points().get("spectrochempy.plugins", [])
    return [f"{ep.name} = {ep.value}" for ep in eps]


def _plugin_runtime_state() -> list[str]:
    try:
        from spectrochempy.plugins.manager import plugin_manager
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return [f"could not import plugin manager: {exc!r}"]

    lines: list[str] = []
    try:
        active = plugin_manager.get_active_plugins()
        failed = plugin_manager.get_failed_plugins()
        readers = sorted(plugin_manager.list_readers())
        accessors = sorted(plugin_manager.list_accessors())
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return [f"could not inspect plugin manager: {exc!r}"]

    lines.append("active plugins: " + (", ".join(active) if active else "none"))
    if failed:
        lines.append("failed plugins:")
        lines.extend(f"  - {name}: {error}" for name, error in failed.items())
    else:
        lines.append("failed plugins: none")
    lines.append("plugin readers: " + (", ".join(readers) if readers else "none"))
    lines.append("plugin accessors: " + (", ".join(accessors) if accessors else "none"))
    return lines


def print_environment() -> None:
    print("::group::CI environment diagnostics")
    print(f"Python: {sys.version.replace(os.linesep, ' ')}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor() or 'unknown'}")
    print(f"Working directory: {Path.cwd()}")

    print("\nInstalled official plugin distributions:")
    for dist in OFFICIAL_PLUGIN_DISTRIBUTIONS:
        version = _distribution_version(dist)
        state = f"installed ({version})" if version else "missing"
        print(f"  - {dist}: {state}")

    print("\nSpectroChemPy plugin entry points:")
    entry_points = _entry_points()
    if entry_points:
        for line in entry_points:
            print(f"  - {line}")
    else:
        print("  none")

    print("\nOptional dependencies:")
    for name in OPTIONAL_DEPENDENCIES:
        print(f"  - {name}: {_module_status(name)}")

    print("\nSpectroChemPy plugin manager:")
    for line in _plugin_runtime_state():
        print(f"  {line}")
    print("::endgroup::")


def parse_pytest_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8", errors="replace")
    summary_lines = [
        line.strip()
        for line in text.splitlines()
        if re.match(r"=+ .* in .* =+$", line.strip())
    ]
    if not summary_lines:
        return {}

    summary = summary_lines[-1].strip("= ")
    result: dict[str, str] = {"raw": summary}

    for status in PYTEST_STATUSES:
        match = re.search(rf"(\d+)\s+{status}\b", summary)
        if match:
            result[status] = match.group(1)

    duration = re.search(r"\bin\s+([0-9:.]+[a-zA-Z]*)", summary)
    if duration:
        result["duration"] = duration.group(1)

    return result


def _read_timings(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]


def read_execution_reports(directory: Path | None) -> list[dict]:
    if directory is None:
        return []
    reports = []
    for path in sorted(directory.glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            for key in ("collected", "executed", "exitstatus"):
                value = report[key]
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"Invalid {key}")
        except (ValueError, KeyError, OSError, TypeError) as error:
            print(f"Warning: could not read {path.name}: {error}", file=sys.stderr)
            continue
        reports.append({**report, "invocation": path.stem})
    return reports


def print_job_summary(
    pytest_log: Path | None, timings: Path | None, pytest_reports: Path | None = None
) -> None:
    print("::group::CI job summary")
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")

    summary = parse_pytest_summary(pytest_log) if pytest_log else {}
    reports = read_execution_reports(pytest_reports)
    if reports:
        print("Pytest invocations:")
        for report in reports:
            print(
                f"  {report['invocation']}: collected={report['collected']} "
                f"executed={report['executed']} exitstatus={report['exitstatus']}"
            )
        print(f"Total test executions: {sum(r['executed'] for r in reports)}")
    elif summary:
        print("Pytest summary:")
        for status in PYTEST_STATUSES:
            print(f"  {status}: {summary.get(status, '0')}")
        print(f"  duration: {summary.get('duration', 'unknown')}")
        print(f"  raw: {summary['raw']}")
    elif pytest_log:
        print(f"Pytest summary: not found in {pytest_log}")
    else:
        print("Pytest summary: no pytest log provided")

    timings_lines = _read_timings(timings)
    print("Timing breakdown:")
    if timings_lines:
        for line in timings_lines:
            print(f"  - {line}")
    else:
        print("  none recorded")

    print("Plugin availability:")
    for dist in OFFICIAL_PLUGIN_DISTRIBUTIONS:
        state = "yes" if _distribution_version(dist) else "no"
        print(f"  {dist}: {state}")
    print("::endgroup::")

    github_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not github_summary:
        return

    with Path(github_summary).open("a", encoding="utf-8") as stream:
        stream.write("## CI diagnostics\n\n")
        stream.write(f"- Python: `{platform.python_version()}`\n")
        stream.write(f"- Platform: `{platform.platform()}`\n")
        if reports:
            stream.write("\n### Pytest invocations\n\n")
            stream.write("| Invocation | Collected | Executed | Exit status |\n")
            stream.write("|---|---:|---:|---:|\n")
            for report in reports:
                stream.write(
                    f"| {report['invocation']} | {report['collected']} | "
                    f"{report['executed']} | {report['exitstatus']} |\n"
                )
            stream.write(
                f"\nTotal test executions: **{sum(r['executed'] for r in reports)}**\n\n"
            )
        elif summary:
            stream.write("- Pytest:\n")
            for status in PYTEST_STATUSES:
                stream.write(f"  - {status}: `{summary.get(status, '0')}`\n")
            stream.write(f"  - duration: `{summary.get('duration', 'unknown')}`\n")
        if timings_lines:
            stream.write("- Timings:\n")
            for line in timings_lines:
                stream.write(f"  - {line}\n")
        stream.write("- Official plugins:\n")
        for dist in OFFICIAL_PLUGIN_DISTRIBUTIONS:
            state = "yes" if _distribution_version(dist) else "no"
            stream.write(f"  - {dist}: `{state}`\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("environment")
    summary = subparsers.add_parser("summary")
    summary.add_argument("--pytest-log")
    summary.add_argument("--timings")
    summary.add_argument("--pytest-reports")
    args = parser.parse_args()

    if args.command == "environment":
        print_environment()
    elif args.command == "summary":
        pytest_log = Path(args.pytest_log) if args.pytest_log else None
        timings = Path(args.timings) if args.timings else None
        pytest_reports = Path(args.pytest_reports) if args.pytest_reports else None
        print_job_summary(pytest_log, timings, pytest_reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
