#!/usr/bin/env python3
# ruff: noqa: S603, T201
r"""
Validate locally-built release artifacts before publication.

Supports Python artifacts (wheel + sdist) and Conda packages (.conda / .tar.bz2).
Designed to be invoked locally and in CI, never contacting any publication service.

Usage::

    python validate_release_artifacts.py python \\
        --package spectrochempy \\
        --version 0.12.8 \\
        --dist dist/

    python validate_release_artifacts.py conda \\
        --package spectrochempy-nmr \\
        --version 0.1.4 \\
        --artifact output/spectrochempy-nmr-0.1.4-0_abc.conda

    python validate_release_artifacts.py all ...
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class Severity(StrEnum):
    SUCCESS = "success"
    WARNING = "warning"
    FAILURE = "failure"
    SKIPPED = "skipped"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class CheckResult:
    name: str
    severity: Severity
    message: str
    details: str = ""

    @property
    def ok(self) -> bool:
        return self.severity in (
            Severity.SUCCESS,
            Severity.WARNING,
            Severity.NOT_APPLICABLE,
            Severity.SKIPPED,
        )


@dataclass
class ValidationReport:
    package: str
    version: str
    artifact_type: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.ok for r in self.checks)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.checks if r.severity == Severity.FAILURE)

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.checks if r.severity == Severity.WARNING)

    def add(
        self, name: str, severity: Severity, message: str, details: str = ""
    ) -> None:
        self.checks.append(
            CheckResult(name=name, severity=severity, message=message, details=details)
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "package": self.package,
            "version": self.version,
            "artifact_type": self.artifact_type,
            "passed": self.passed,
            "failure_count": self.failure_count,
            "warning_count": self.warning_count,
            "checks": [asdict(c) for c in self.checks],
        }

    def to_markdown(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"## Artifact validation: {self.package} {self.version}"
            f" ({self.artifact_type}) -- {status}",
            "",
            "| Check | Severity | Message |",
            "|-------|----------|---------|",
        ]
        for c in self.checks:
            lines.append(f"| {c.name} | {c.severity.value} | {c.message} |")
        lines.append("")
        return "\n".join(lines)


def _print_report(report: ValidationReport) -> None:
    symbols = {
        Severity.SUCCESS: "\u2713",
        Severity.WARNING: "\u26a0",
        Severity.FAILURE: "\u2717",
        Severity.SKIPPED: "-",
        Severity.NOT_APPLICABLE: "-",
    }
    print(f"\n{'=' * 60}")
    print(
        f"  Validation: {report.package} {report.version}" f" ({report.artifact_type})"
    )
    print(f"{'=' * 60}")
    for c in report.checks:
        sym = symbols[c.severity]
        print(f"  {sym} [{c.severity.value}] {c.name}: {c.message}")
        if c.details:
            for line in c.details.splitlines():
                print(f"      {line}")
    print(
        f"\n  Result: {'PASSED' if report.passed else 'FAILED'}"
        f" ({report.failure_count} failures, {report.warning_count} warnings)"
    )
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Utility: archive path safety checks
# ---------------------------------------------------------------------------

_DANGEROUS_NAMES = frozenset(
    {
        ".git",
        ".gitignore",
        ".gitattributes",
        "__pycache__",
        ".env",
        ".env.local",
        ".env.production",
        "credentials",
        "credentials.json",
        "service-account.json",
        "keyfile.json",
        ".netrc",
        ".ssh",
    }
)

_DANGEROUS_PATH_PATTERNS = (
    ".env",
    "credentials",
    "service-account",
    "keyfile.json",
    ".netrc",
)


def _check_archive_paths(entries: list[str], report: ValidationReport) -> None:
    """Check archive member paths for dangerous entries."""
    for entry in entries:
        normalized = entry.replace("\\", "/")
        if normalized.startswith("../") or normalized.startswith("/"):
            report.add(
                f"path:{entry}",
                Severity.FAILURE,
                f"Dangerous absolute/relative path: {entry}",
            )
        parts = normalized.split("/")
        if any(p == ".." for p in parts):
            report.add(
                f"path:{entry}",
                Severity.FAILURE,
                f"Path traversal detected: {entry}",
            )
        basename = parts[-1] if parts else ""
        if basename in _DANGEROUS_NAMES:
            report.add(
                f"path:{entry}",
                Severity.WARNING,
                f"Potentially sensitive file in archive: {basename}",
            )
        for pat in _DANGEROUS_PATH_PATTERNS:
            if pat in normalized.lower() and basename not in _DANGEROUS_NAMES:
                report.add(
                    f"path:{entry}",
                    Severity.WARNING,
                    f"Potentially sensitive path pattern: {entry}",
                )


# ---------------------------------------------------------------------------
# Python artifact discovery
# ---------------------------------------------------------------------------

_WHEEL_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9_.]+)-(?P<version>\d[\d._a-zA-Z+]*)"
    r"(?:-(?P<build>\d+))?"
    r"-(?P<python>[A-Za-z0-9_.]+)-(?P<abi>[A-Za-z0-9_.]+)"
    r"-(?P<plat>[A-Za-z0-9_.]+)\.whl$"
)


@dataclass
class PythonArtifacts:
    wheel: Path | None = None
    sdist: Path | None = None
    extra_wheels: list[Path] = field(default_factory=list)
    extra_sdists: list[Path] = field(default_factory=list)


def _normalise_dist_name(name: str) -> str:
    """Normalise a distribution name for comparison (PEP 625)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def discover_python_artifacts(
    dist_dir: Path,
    package: str,
    version: str,
    report: ValidationReport,
) -> PythonArtifacts:
    """Discover wheel and sdist in dist_dir for the given package/version."""
    if not dist_dir.is_dir():
        report.add(
            "discover",
            Severity.FAILURE,
            f"Distribution directory not found: {dist_dir}",
        )
        return PythonArtifacts()

    expected_norm = _normalise_dist_name(package)
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))

    matched_wheels: list[Path] = []
    matched_sdists: list[Path] = []

    for whl in wheels:
        m = _WHEEL_RE.match(whl.name)
        if (
            m
            and _normalise_dist_name(m.group("name")) == expected_norm
            and m.group("version") == version
        ):
            matched_wheels.append(whl)
        elif m and _normalise_dist_name(m.group("name")) == expected_norm:
            report.add(
                f"wheel-version:{whl.name}",
                Severity.WARNING,
                f"Wheel found for {package} but version mismatch:"
                f" {m.group('version')} (expected {version})",
            )

    for sdist in sdists:
        sn = sdist.name
        norm_pkg = package.replace("-", "_")
        if (sn.startswith(package) or sn.startswith(norm_pkg)) and version in sn:
            matched_sdists.append(sdist)

    result = PythonArtifacts()

    if not matched_wheels:
        report.add(
            "discover:wheel",
            Severity.FAILURE,
            f"No wheel found for {package}=={version} in {dist_dir}",
        )
    elif len(matched_wheels) == 1:
        result.wheel = matched_wheels[0]
        report.add(
            "discover:wheel",
            Severity.SUCCESS,
            f"Wheel found: {matched_wheels[0].name}",
        )
    else:
        result.wheel = matched_wheels[0]
        result.extra_wheels = matched_wheels[1:]
        report.add(
            "discover:wheel",
            Severity.WARNING,
            f"Multiple wheels found for {package}; using first:"
            f" {matched_wheels[0].name}",
            details="\n".join(p.name for p in matched_wheels),
        )

    if not matched_sdists:
        report.add(
            "discover:sdist",
            Severity.FAILURE,
            f"No sdist found for {package}=={version} in {dist_dir}",
        )
    elif len(matched_sdists) == 1:
        result.sdist = matched_sdists[0]
        report.add(
            "discover:sdist",
            Severity.SUCCESS,
            f"Sdist found: {matched_sdists[0].name}",
        )
    else:
        result.sdist = matched_sdists[0]
        result.extra_sdists = matched_sdists[1:]
        report.add(
            "discover:sdist",
            Severity.WARNING,
            f"Multiple sdists found for {package}; using first:"
            f" {matched_sdists[0].name}",
            details="\n".join(p.name for p in matched_sdists),
        )

    return result


# ---------------------------------------------------------------------------
# Twine check
# ---------------------------------------------------------------------------


def check_twine(artifacts: PythonArtifacts, report: ValidationReport) -> None:
    """Run twine check --strict on the wheel and sdist."""
    twine_bin = shutil.which("twine")
    if twine_bin is None:
        report.add(
            "twine",
            Severity.WARNING,
            "twine not installed; skipping twine check",
        )
        return

    files: list[str] = []
    if artifacts.wheel:
        files.append(str(artifacts.wheel))
    if artifacts.sdist:
        files.append(str(artifacts.sdist))
    if not files:
        report.add("twine", Severity.SKIPPED, "No artifacts to check")
        return

    result = subprocess.run(
        [twine_bin, "check", "--strict", *files],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        report.add("twine", Severity.SUCCESS, "twine check --strict passed")
    else:
        report.add(
            "twine",
            Severity.FAILURE,
            "twine check --strict failed",
            details=result.stdout + result.stderr,
        )


# ---------------------------------------------------------------------------
# Metadata reading helpers
# ---------------------------------------------------------------------------


def _parse_rfc822(text: str) -> dict[str, str]:
    """Parse a simple RFC-822 style metadata file into a dict."""
    result: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line and not line.startswith(" "):
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip()
    return result


def _parse_requires_dist(text: str) -> list[str]:
    """Parse Requires-Dist entries from metadata text."""
    deps: list[str] = []
    for line in text.splitlines():
        if line.startswith("Requires-Dist:"):
            deps.append(line[len("Requires-Dist:") :].strip())
    return deps


def _read_wheel_metadata_text(whl: Path) -> str:
    try:
        with zipfile.ZipFile(whl) as zf:
            for name in zf.namelist():
                if name.endswith(".dist-info/METADATA"):
                    return zf.read(name).decode("utf-8")
    except Exception:  # noqa: S110 - unreadable wheel yields empty metadata
        pass
    return ""


def _read_sdist_metadata_text(sdist: Path) -> str:
    try:
        with tarfile.open(sdist, "r:gz") as tf:
            for member in tf.getmembers():
                if member.name.endswith("/PKG-INFO"):
                    fobj = tf.extractfile(member)
                    if fobj:
                        return fobj.read().decode("utf-8")
    except Exception:  # noqa: S110 - unreadable sdist yields empty metadata
        pass
    return ""


# ---------------------------------------------------------------------------
# Python metadata validation
# ---------------------------------------------------------------------------


def validate_python_metadata(
    artifacts: PythonArtifacts,
    package: str,
    version: str,
    report: ValidationReport,
) -> None:
    """Validate metadata from wheel and sdist."""
    wheel_text = _read_wheel_metadata_text(artifacts.wheel) if artifacts.wheel else ""
    sdist_text = _read_sdist_metadata_text(artifacts.sdist) if artifacts.sdist else ""
    wheel_meta = _parse_rfc822(wheel_text) if wheel_text else {}
    sdist_meta = _parse_rfc822(sdist_text) if sdist_text else {}

    meta_source = wheel_meta or sdist_meta
    if not meta_source:
        report.add(
            "metadata:available",
            Severity.FAILURE,
            "No metadata readable from any artifact",
        )
        return
    report.add(
        "metadata:available",
        Severity.SUCCESS,
        "Metadata readable from artifacts",
    )

    # Name
    name_field = meta_source.get("Name", "")
    if not name_field:
        report.add(
            "metadata:name",
            Severity.FAILURE,
            "Name field missing from metadata",
        )
    elif _normalise_dist_name(name_field) != _normalise_dist_name(package):
        report.add(
            "metadata:name",
            Severity.FAILURE,
            f"Name mismatch: metadata='{name_field}', expected='{package}'",
        )
    else:
        report.add("metadata:name", Severity.SUCCESS, f"Name correct: {name_field}")

    # Version
    version_field = meta_source.get("Version", "")
    if not version_field:
        report.add(
            "metadata:version",
            Severity.FAILURE,
            "Version field missing from metadata",
        )
    elif version_field != version:
        report.add(
            "metadata:version",
            Severity.FAILURE,
            f"Version mismatch: metadata='{version_field}', expected='{version}'",
        )
    else:
        report.add(
            "metadata:version", Severity.SUCCESS, f"Version correct: {version_field}"
        )

    # Requires-Python
    rp = meta_source.get("Requires-Python", "")
    if not rp:
        report.add(
            "metadata:requires-python",
            Severity.WARNING,
            "Requires-Python not declared",
        )
    else:
        report.add(
            "metadata:requires-python",
            Severity.SUCCESS,
            f"Requires-Python: {rp}",
        )

    # License
    lic = meta_source.get("License", "") or meta_source.get("License-Expression", "")
    if not lic:
        report.add(
            "metadata:license",
            Severity.WARNING,
            "License field not found in metadata",
        )
    else:
        report.add("metadata:license", Severity.SUCCESS, f"License: {lic}")

    # Dependencies
    wheel_deps = _parse_requires_dist(wheel_text)
    sdist_deps = _parse_requires_dist(sdist_text)
    deps = wheel_deps or sdist_deps
    if deps:
        report.add(
            "metadata:dependencies",
            Severity.SUCCESS,
            f"Dependencies found: {len(deps)} requirement(s)",
        )
    else:
        report.add(
            "metadata:dependencies",
            Severity.WARNING,
            "No Requires-Dist entries found",
        )

    # Consistency
    if wheel_meta and sdist_meta:
        wn = _normalise_dist_name(wheel_meta.get("Name", ""))
        sn = _normalise_dist_name(sdist_meta.get("Name", ""))
        wv = wheel_meta.get("Version", "")
        sv = sdist_meta.get("Version", "")
        if wn == sn and wv == sv:
            report.add(
                "metadata:consistency",
                Severity.SUCCESS,
                "Wheel and sdist metadata consistent (name, version)",
            )
        else:
            report.add(
                "metadata:consistency",
                Severity.FAILURE,
                f"Inconsistent: wheel={wn}/{wv}, sdist={sn}/{sv}",
            )

    # Required metadata files in wheel
    _check_wheel_metadata_files(artifacts, report)


def _check_wheel_metadata_files(
    artifacts: PythonArtifacts, report: ValidationReport
) -> None:
    if not artifacts.wheel:
        return
    try:
        with zipfile.ZipFile(artifacts.wheel) as zf:
            names = zf.namelist()
            has_meta = any(n.endswith(".dist-info/METADATA") for n in names)
            has_rec = any(n.endswith(".dist-info/RECORD") for n in names)
            has_whl = any(n.endswith(".dist-info/WHEEL") for n in names)
            if has_meta and has_rec and has_whl:
                report.add(
                    "metadata:files",
                    Severity.SUCCESS,
                    "Required metadata files present (METADATA, RECORD, WHEEL)",
                )
            else:
                missing = []
                if not has_meta:
                    missing.append("METADATA")
                if not has_rec:
                    missing.append("RECORD")
                if not has_whl:
                    missing.append("WHEEL")
                report.add(
                    "metadata:files",
                    Severity.WARNING,
                    f"Missing metadata files: {', '.join(missing)}",
                )
    except Exception as exc:
        report.add(
            "metadata:files",
            Severity.WARNING,
            f"Cannot inspect wheel contents: {exc}",
        )


# ---------------------------------------------------------------------------
# Python archive content validation
# ---------------------------------------------------------------------------

_UNDESIRABLE_PATHS = (
    ".git/",
    ".pytest_cache/",
    ".tox/",
    ".nox/",
    "__pycache__/",
    "venv/",
    ".venv/",
    "env/",
    ".env/",
    "node_modules/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".eggs/",
)


def validate_python_content(
    artifacts: PythonArtifacts, package: str, report: ValidationReport
) -> None:
    """Check wheel contents for dangerous or undesirable entries."""
    if not artifacts.wheel:
        report.add("content:wheel", Severity.SKIPPED, "No wheel to inspect")
        return

    try:
        with zipfile.ZipFile(artifacts.wheel) as zf:
            names = zf.namelist()

            _check_archive_paths(names, report)

            absolute = [n for n in names if n.startswith("/")]
            if absolute:
                report.add(
                    "content:absolute-paths",
                    Severity.FAILURE,
                    f"Absolute paths in wheel: {', '.join(absolute[:5])}",
                )
            else:
                report.add(
                    "content:absolute-paths",
                    Severity.SUCCESS,
                    "No absolute paths in wheel",
                )

            traversal = [n for n in names if ".." in n.split("/")]
            if traversal:
                report.add(
                    "content:traversal",
                    Severity.FAILURE,
                    f"Path traversal in wheel: {', '.join(traversal[:5])}",
                )
            else:
                report.add(
                    "content:traversal",
                    Severity.SUCCESS,
                    "No path traversal in wheel",
                )

            undesirable: list[str] = []
            for name in names:
                bn = name.split("/")[-1] if "/" in name else name
                for prefix in _UNDESIRABLE_PATHS:
                    if name.startswith(prefix):
                        undesirable.append(name)
                        break
                if bn in {
                    ".env",
                    ".env.local",
                    "credentials.json",
                    "service-account.json",
                    ".netrc",
                }:
                    undesirable.append(name)

            if undesirable:
                report.add(
                    "content:undesirable",
                    Severity.WARNING,
                    f"Potentially undesirable files ({len(undesirable)}):"
                    f" {', '.join(undesirable[:10])}",
                )
            else:
                report.add(
                    "content:undesirable",
                    Severity.SUCCESS,
                    "No obviously undesirable files in wheel",
                )

            if hasattr(zipfile.Path, "is_symlink"):
                symlinks = [n for n in names if zipfile.Path(zf, n).is_symlink()]
                if symlinks:
                    report.add(
                        "content:symlinks",
                        Severity.WARNING,
                        f"Symlinks in wheel: {', '.join(symlinks[:5])}",
                    )

            # Check main package presence
            pkg_norm = package.replace("-", "_")
            has_pkg = any(
                n.startswith(f"{pkg_norm}/") and n.endswith(".py") for n in names
            )
            if has_pkg:
                report.add(
                    "content:package",
                    Severity.SUCCESS,
                    f"Package '{pkg_norm}' found in wheel",
                )
            else:
                report.add(
                    "content:package",
                    Severity.WARNING,
                    f"Package '{pkg_norm}' not found as top-level directory in wheel",
                )

    except zipfile.BadZipFile:
        report.add(
            "content:wheel",
            Severity.FAILURE,
            f"Corrupt wheel archive: {artifacts.wheel.name}",
        )
    except Exception as exc:
        report.add(
            "content:wheel",
            Severity.FAILURE,
            f"Cannot inspect wheel: {exc}",
        )


# ---------------------------------------------------------------------------
# Python installation and smoke test
# ---------------------------------------------------------------------------


def install_and_smoketest(
    artifacts: PythonArtifacts,
    package: str,
    version: str,
    module_name: str | None,
    no_deps: bool,
    report: ValidationReport,
) -> None:
    """Install the wheel in an isolated venv and run a smoke test."""
    if not artifacts.wheel:
        report.add(
            "install:wheel", Severity.SKIPPED, "No wheel available for install test"
        )
        return

    if module_name is None:
        module_name = package.replace("-", "_")

    venv_dir = Path(tempfile.mkdtemp(prefix="validate_venv_"))
    try:
        venv_python = venv_dir / "bin" / "python"
        venv_cmd = [sys.executable, "-m", "venv"]
        if no_deps:
            # When dependencies are already satisfied in the base environment,
            # give the venv access to them so the smoke test runs offline.
            venv_cmd.append("--system-site-packages")
        venv_cmd.append(str(venv_dir))
        result = subprocess.run(
            venv_cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            report.add(
                "install:venv",
                Severity.FAILURE,
                f"Failed to create venv: {result.stderr}",
            )
            return

        report.add(
            "install:venv",
            Severity.SUCCESS,
            f"Venv created: {venv_dir}"
            + (" (with system site-packages)" if no_deps else ""),
        )

        cmd = [str(venv_python), "-m", "pip", "install", str(artifacts.wheel)]
        if no_deps:
            cmd.append("--no-deps")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            report.add(
                "install:install",
                Severity.FAILURE,
                "pip install failed",
                details=result.stdout[-2000:] + result.stderr[-2000:],
            )
            return

        report.add(
            "install:install",
            Severity.SUCCESS,
            "Wheel installed successfully" + (" (--no-deps)" if no_deps else ""),
        )

        # Import test
        import_cmd = f"import {module_name}; print(getattr({module_name}, '__version__', 'unknown'))"
        result = subprocess.run(
            [str(venv_python), "-c", import_cmd],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            report.add(
                "install:import",
                Severity.FAILURE,
                f"Failed to import {module_name}",
                details=result.stderr[-2000:],
            )
        else:
            imported_ver = result.stdout.strip()
            report.add(
                "install:import",
                Severity.SUCCESS,
                f"Successfully imported {module_name} (version: {imported_ver})",
            )

    finally:
        shutil.rmtree(venv_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Sdist rebuild test
# ---------------------------------------------------------------------------


def rebuild_from_sdist(
    artifacts: PythonArtifacts,
    package: str,
    version: str,
    report: ValidationReport,
) -> None:
    """Rebuild a wheel from sdist and compare metadata semantically."""
    if not artifacts.sdist:
        report.add(
            "rebuild:sdist", Severity.SKIPPED, "No sdist available for rebuild test"
        )
        return

    tmpdir = Path(tempfile.mkdtemp(prefix="validate_rebuild_"))
    try:
        venv_dir = tmpdir / "venv"
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            capture_output=True,
            text=True,
        )
        venv_python = venv_dir / "bin" / "python"
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "build"],
            capture_output=True,
            text=True,
        )

        build_out = tmpdir / "rebuild_dist"
        build_out.mkdir()
        result = subprocess.run(
            [
                str(venv_python),
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(build_out),
                str(artifacts.sdist),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            report.add(
                "rebuild:build",
                Severity.FAILURE,
                "Failed to rebuild wheel from sdist",
                details=result.stdout[-2000:] + result.stderr[-2000:],
            )
            return

        rebuilt_wheels = list(build_out.glob("*.whl"))
        if not rebuilt_wheels:
            report.add(
                "rebuild:build",
                Severity.FAILURE,
                "No wheel produced from sdist rebuild",
            )
            return

        report.add(
            "rebuild:build",
            Severity.SUCCESS,
            f"Wheel rebuilt from sdist: {rebuilt_wheels[0].name}",
        )

        # Compare metadata semantically
        orig_text = (
            _read_wheel_metadata_text(artifacts.wheel) if artifacts.wheel else ""
        )
        rebuild_text = _read_wheel_metadata_text(rebuilt_wheels[0])
        orig_meta = _parse_rfc822(orig_text) if orig_text else {}
        rebuild_meta = _parse_rfc822(rebuild_text) if rebuild_text else {}

        fields_to_compare = ["Name", "Version", "Requires-Python"]
        all_match = True
        for field_name in fields_to_compare:
            ov = orig_meta.get(field_name, "")
            rv = rebuild_meta.get(field_name, "")
            if ov and rv and ov != rv:
                report.add(
                    f"rebuild:{field_name.lower()}",
                    Severity.WARNING,
                    f"{field_name} differs: original='{ov}', rebuild='{rv}'",
                )
                all_match = False

        orig_deps = set(_parse_requires_dist(orig_text))
        rebuild_deps = set(_parse_requires_dist(rebuild_text))
        if orig_deps and rebuild_deps and orig_deps != rebuild_deps:
            report.add(
                "rebuild:dependencies",
                Severity.WARNING,
                "Dependencies differ between original and rebuilt wheel",
            )
            all_match = False

        if all_match:
            report.add(
                "rebuild:metadata",
                Severity.SUCCESS,
                "Rebuilt wheel metadata matches original semantically",
            )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Conda artifact validation
# ---------------------------------------------------------------------------


def discover_conda_artifact(
    artifact_path: Path,
    package: str,
    version: str,
    report: ValidationReport,
) -> Path | None:
    """Discover and validate a Conda package file."""
    if not artifact_path.is_file():
        report.add(
            "conda:discover",
            Severity.FAILURE,
            f"Conda artifact not found: {artifact_path}",
        )
        return None

    name = artifact_path.name
    valid_ext = name.endswith(".conda") or name.endswith(".tar.bz2")
    if not valid_ext:
        report.add(
            "conda:extension",
            Severity.FAILURE,
            f"Invalid extension for Conda package: {name}",
        )
        return None

    report.add(
        "conda:extension",
        Severity.SUCCESS,
        f"Valid Conda package extension: {name}",
    )

    if not name.startswith(package) or not name.startswith(package.replace("-", "_")):
        report.add(
            "conda:name",
            Severity.FAILURE,
            f"Package name mismatch in filename: {name} (expected {package})",
        )
        return None
    report.add("conda:name", Severity.SUCCESS, "Package name correct in filename")

    if f"-{version}-" not in name:
        report.add(
            "conda:version",
            Severity.FAILURE,
            f"Version not found in filename: {name} (expected {version})",
        )
        return None
    report.add("conda:version", Severity.SUCCESS, "Version correct in filename")

    return artifact_path


def validate_conda_metadata(
    artifact: Path,
    package: str,
    version: str,
    report: ValidationReport,
) -> dict[str, Any]:
    """Read and validate Conda package metadata."""
    info: dict[str, Any] = {}
    name = artifact.name

    if name.endswith(".conda"):
        try:
            with zipfile.ZipFile(artifact) as zf:
                for n in zf.namelist():
                    if n.endswith("info/repodata_record.json") or n.endswith(
                        "info/index.json"
                    ):
                        with contextlib.suppress(json.JSONDecodeError):
                            info = json.loads(zf.read(n))
        except zipfile.BadZipFile:
            report.add(
                "conda:read",
                Severity.FAILURE,
                f"Cannot read .conda file: {name}",
            )
            return info
    elif name.endswith(".tar.bz2"):
        try:
            with tarfile.open(artifact, "r:bz2") as tf:
                for member in tf.getmembers():
                    if member.name.endswith("info/index.json"):
                        fobj = tf.extractfile(member)
                        if fobj:
                            info = json.loads(fobj.read().decode("utf-8"))
        except Exception as exc:
            report.add(
                "conda:read",
                Severity.FAILURE,
                f"Cannot read .tar.bz2 file: {name}: {exc}",
            )
            return info

    if not info:
        report.add(
            "conda:metadata",
            Severity.WARNING,
            "No index.json or repodata_record.json found in package",
        )
        return info

    report.add(
        "conda:metadata",
        Severity.SUCCESS,
        "Conda package metadata readable",
    )

    # Name
    pkg_name = info.get("name", "")
    if pkg_name and pkg_name != package:
        report.add(
            "conda:metadata:name",
            Severity.FAILURE,
            f"Conda metadata name mismatch: '{pkg_name}', expected '{package}'",
        )
    elif pkg_name:
        report.add(
            "conda:metadata:name",
            Severity.SUCCESS,
            f"Conda name correct: {pkg_name}",
        )

    # Version
    pkg_ver = info.get("version", "")
    if pkg_ver and pkg_ver != version:
        report.add(
            "conda:metadata:version",
            Severity.FAILURE,
            f"Conda metadata version mismatch: '{pkg_ver}', expected '{version}'",
        )
    elif pkg_ver:
        report.add(
            "conda:metadata:version",
            Severity.SUCCESS,
            f"Conda version correct: {pkg_ver}",
        )

    # Build
    build_str = info.get("build", "")
    if build_str:
        report.add(
            "conda:metadata:build",
            Severity.SUCCESS,
            f"Build string: {build_str}",
        )

    # noarch
    subdir = info.get("subdir", "")
    if "noarch" in subdir or "noarch" in str(build_str):
        report.add(
            "conda:metadata:subdir",
            Severity.SUCCESS,
            f"Subdir: {subdir} (noarch package)",
        )
    elif subdir:
        report.add(
            "conda:metadata:subdir",
            Severity.SUCCESS,
            f"Subdir: {subdir}",
        )

    # Dependencies
    depends = info.get("depends", [])
    if depends:
        report.add(
            "conda:metadata:depends",
            Severity.SUCCESS,
            f"Dependencies: {len(depends)} declared",
        )
    else:
        report.add(
            "conda:metadata:depends",
            Severity.WARNING,
            "No dependencies declared in Conda metadata",
        )

    return info


def validate_conda_content(
    artifact: Path, package: str, report: ValidationReport
) -> None:
    """Check Conda package contents for dangerous or undesirable entries."""
    name = artifact.name

    try:
        if name.endswith(".conda"):
            with zipfile.ZipFile(artifact) as zf:
                names = zf.namelist()
                _check_archive_paths(names, report)

                has_python = any(n.endswith(".py") and "info/" not in n for n in names)
                if has_python:
                    report.add(
                        "conda:content:python",
                        Severity.SUCCESS,
                        "Python files found in package",
                    )
                else:
                    report.add(
                        "conda:content:python",
                        Severity.WARNING,
                        "No Python files found outside info/",
                    )

        elif name.endswith(".tar.bz2"):
            with tarfile.open(artifact, "r:bz2") as tf:
                names = [m.name for m in tf.getmembers()]
                _check_archive_paths(names, report)

                has_python = any(n.endswith(".py") and "info/" not in n for n in names)
                if has_python:
                    report.add(
                        "conda:content:python",
                        Severity.SUCCESS,
                        "Python files found in package",
                    )
                else:
                    report.add(
                        "conda:content:python",
                        Severity.WARNING,
                        "No Python files found outside info/",
                    )
    except Exception as exc:
        report.add(
            "conda:content",
            Severity.WARNING,
            f"Cannot inspect Conda package contents: {exc}",
        )


def install_and_smoketest_conda(
    artifact: Path,
    package: str,
    version: str,
    module_name: str | None,
    report: ValidationReport,
) -> None:
    """Install the Conda package in an isolated env and run a smoke test."""
    micromamba = (
        shutil.which("micromamba") or shutil.which("mamba") or shutil.which("conda")
    )
    if micromamba is None:
        report.add(
            "conda:install",
            Severity.WARNING,
            "No conda/mamba/micromamba found; skipping install test",
        )
        return

    if module_name is None:
        module_name = package.replace("-", "_")

    with tempfile.TemporaryDirectory(prefix="validate_conda_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        env_dir = tmpdir_path / "env"
        local_channel = tmpdir_path / "channel"

        # Create local channel structure
        subdir_dir = local_channel / "noarch"
        if artifact.name.endswith(".tar.bz2"):
            subdir_dir = local_channel / "noarch"
        else:
            subdir_dir = local_channel / "noarch"
        subdir_dir.mkdir(parents=True)
        shutil.copy2(artifact, subdir_dir / artifact.name)

        # Index the channel
        if "conda" in micromamba or "mamba" in micromamba:
            subprocess.run(
                [micromamba, "index", str(local_channel)],
                capture_output=True,
                text=True,
            )
        else:
            subprocess.run(
                [micromamba, "index", str(local_channel)],
                capture_output=True,
                text=True,
            )

        # Create env and install
        create_cmd = [
            micromamba,
            "create",
            "-p",
            str(env_dir),
            "-y",
            "-c",
            str(local_channel),
        ]
        # Add conda-forge for dependencies (read-only)
        create_cmd.extend(["-c", "conda-forge"])
        # Add spectrocat for core dependency
        create_cmd.extend(["-c", "spectrocat"])
        create_cmd.append(package)

        result = subprocess.run(create_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            report.add(
                "conda:install",
                Severity.WARNING,
                "micromamba install failed (expected if core not in channel)",
                details=result.stdout[-2000:] + result.stderr[-2000:],
            )
            return

        report.add(
            "conda:install",
            Severity.SUCCESS,
            "Conda package installed in isolated env",
        )

        # Import test
        py_bin = env_dir / "bin" / "python"
        if not py_bin.exists():
            py_bin = env_dir / "bin" / "python3"
        if py_bin.exists():
            result = subprocess.run(
                [
                    str(py_bin),
                    "-c",
                    f"import {module_name}; print(getattr({module_name}, '__version__', 'unknown'))",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                report.add(
                    "conda:import",
                    Severity.FAILURE,
                    f"Failed to import {module_name}",
                    details=result.stderr[-2000:],
                )
            else:
                imported_ver = result.stdout.strip()
                report.add(
                    "conda:import",
                    Severity.SUCCESS,
                    f"Successfully imported {module_name} (version: {imported_ver})",
                )


# ---------------------------------------------------------------------------
# Plugin configuration
# ---------------------------------------------------------------------------

OFFICIAL_PLUGINS: dict[str, dict[str, str]] = {
    "spectrochempy-carroucell": {"module": "spectrochempy_carroucell"},
    "spectrochempy-hypercomplex": {"module": "spectrochempy_hypercomplex"},
    "spectrochempy-iris": {"module": "spectrochempy_iris"},
    "spectrochempy-nmr": {"module": "spectrochempy_nmr"},
    "spectrochempy-perkinelmer": {"module": "spectrochempy_perkinelmer"},
    "spectrochempy-tensor": {"module": "spectrochempy_tensor"},
}


def get_module_name(package: str) -> str | None:
    """Return the importable module name for a known package, or None."""
    if package in OFFICIAL_PLUGINS:
        return OFFICIAL_PLUGINS[package]["module"]
    if package == "spectrochempy":
        return "spectrochempy"
    return None


# ---------------------------------------------------------------------------
# Combined 'all' command
# ---------------------------------------------------------------------------


def validate_all_python(args: argparse.Namespace, report: ValidationReport) -> None:
    """Run all Python validation checks."""
    dist_dir = Path(args.dist)
    package = args.package
    version = args.version
    module = getattr(args, "module", None) or get_module_name(package)
    no_deps = getattr(args, "no_deps", False)

    artifacts = discover_python_artifacts(dist_dir, package, version, report)
    check_twine(artifacts, report)
    validate_python_metadata(artifacts, package, version, report)
    validate_python_content(artifacts, package, report)
    install_and_smoketest(artifacts, package, version, module, no_deps, report)
    rebuild_from_sdist(artifacts, package, version, report)


def validate_all_conda(args: argparse.Namespace, report: ValidationReport) -> None:
    """Run all Conda validation checks."""
    artifact_path = Path(args.artifact)
    package = args.package
    version = args.version
    module = getattr(args, "module", None) or get_module_name(package)

    artifact = discover_conda_artifact(artifact_path, package, version, report)
    if artifact:
        validate_conda_metadata(artifact, package, version, report)
        validate_conda_content(artifact, package, report)
        install_and_smoketest_conda(artifact, package, version, module, report)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate locally-built release artifacts.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output structured JSON report",
    )
    common.add_argument(
        "--markdown",
        action="store_true",
        dest="as_markdown",
        help="Output Markdown report (for GITHUB_STEP_SUMMARY)",
    )
    common.add_argument(
        "--json-output",
        dest="json_output",
        help="Write JSON report to file",
    )
    common.add_argument(
        "--markdown-output",
        dest="markdown_output",
        help="Write Markdown report to file",
    )

    # python
    p_py = sub.add_parser(
        "python",
        help="Validate Python artifacts (wheel + sdist)",
        parents=[common],
    )
    p_py.add_argument("--package", required=True, help="Package name")
    p_py.add_argument("--version", required=True, help="Expected version")
    p_py.add_argument("--dist", required=True, help="Distribution directory")
    p_py.add_argument(
        "--module", help="Module to import (default: derived from package)"
    )
    p_py.add_argument(
        "--no-deps",
        action="store_true",
        help="Install with --no-deps (dependencies already satisfied)",
    )

    # conda
    p_conda = sub.add_parser(
        "conda",
        help="Validate a Conda package",
        parents=[common],
    )
    p_conda.add_argument("--package", required=True, help="Package name")
    p_conda.add_argument("--version", required=True, help="Expected version")
    p_conda.add_argument("--artifact", required=True, help="Path to .conda / .tar.bz2")
    p_conda.add_argument(
        "--module", help="Module to import (default: derived from package)"
    )

    # all
    p_all = sub.add_parser(
        "all",
        help="Validate both Python and Conda artifacts",
        parents=[common],
    )
    p_all.add_argument("--package", required=True, help="Package name")
    p_all.add_argument("--version", required=True, help="Expected version")
    p_all.add_argument("--dist", help="Distribution directory for Python artifacts")
    p_all.add_argument("--artifact", help="Path to .conda / .tar.bz2")
    p_all.add_argument("--module", help="Module to import")
    p_all.add_argument(
        "--no-deps",
        action="store_true",
        help="Install with --no-deps",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reports: list[ValidationReport] = []

    if args.command == "python":
        report = ValidationReport(
            package=args.package,
            version=args.version,
            artifact_type="python",
        )
        validate_all_python(args, report)
        reports.append(report)

    elif args.command == "conda":
        report = ValidationReport(
            package=args.package,
            version=args.version,
            artifact_type="conda",
        )
        validate_all_conda(args, report)
        reports.append(report)

    elif args.command == "all":
        report = ValidationReport(
            package=args.package,
            version=args.version,
            artifact_type="all",
        )
        if args.dist:
            validate_all_python(args, report)
        if args.artifact:
            validate_all_conda(args, report)
        reports.append(report)

    # Output
    for report in reports:
        if not args.as_json and not args.as_markdown:
            _print_report(report)

        if args.as_json:
            print(json.dumps(report.to_json(), indent=2))

        if args.as_markdown:
            print(report.to_markdown())

        if args.json_output:
            Path(args.json_output).write_text(
                json.dumps(report.to_json(), indent=2), encoding="utf-8"
            )

        if args.markdown_output:
            Path(args.markdown_output).write_text(
                report.to_markdown(), encoding="utf-8"
            )

    return 0 if all(r.passed for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
