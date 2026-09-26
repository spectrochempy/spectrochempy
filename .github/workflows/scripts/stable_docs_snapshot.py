#!/usr/bin/env python3
# ruff: noqa: S603, T201
"""Build, validate, and atomically promote a stable docs snapshot."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote
from urllib.parse import urlsplit

FINAL_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
CORE_TAG_RE = re.compile(r"^spectrochempy-v(?P<version>\d+\.\d+\.\d+)$")
PLUGIN_PACKAGES = (
    "spectrochempy-carroucell",
    "spectrochempy-hypercomplex",
    "spectrochempy-iris",
    "spectrochempy-nmr",
    "spectrochempy-perkinelmer",
    "spectrochempy-tensor",
)
REPRESENTATIVE_API_PAGES = (
    "reference/generated/spectrochempy.NDDataset.html",
    "reference/generated/spectrochempy.read_omnic.html",
    "reference/generated/spectrochempy.fft.html",
)


class _LinkCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                self.links.append(value)


def _require_file(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"Required documentation page is missing: {path}")
    return path.read_text(encoding="utf-8")


def _validate_local_html_links(snapshot: Path, page: Path, content: str) -> None:
    parser = _LinkCollector()
    parser.feed(content)
    snapshot = snapshot.resolve()
    for link in parser.links:
        parsed = urlsplit(link)
        if (
            parsed.scheme
            or parsed.netloc
            or not parsed.path
            or parsed.path.startswith("/")
        ):
            continue
        target = (page.parent / unquote(parsed.path)).resolve()
        if snapshot not in target.parents and target != snapshot:
            raise RuntimeError(
                f"Documentation link escapes the snapshot: {page}: {link}"
            )
        if parsed.path.endswith("/"):
            target /= "index.html"
        if target.suffix == ".html" and not target.is_file():
            raise RuntimeError(f"Broken documentation link: {page}: {link}")


def validate_snapshot(snapshot: Path, version: str) -> None:
    """Validate identity, release notes, API coverage, and representative links."""
    if FINAL_VERSION_RE.fullmatch(version) is None:
        raise ValueError(f"Expected a final X.Y.Z version, got {version!r}")

    snapshot = Path(snapshot)
    marker = _require_file(snapshot / ".spectrochempy-doc-version").strip()
    if marker != version:
        raise RuntimeError(f"Snapshot marker is {marker!r}, expected {version!r}")

    notes = _require_file(snapshot / "whatsnew" / "latest.html")
    if f"Revision {version}" not in notes:
        raise RuntimeError(f"Stable release notes do not identify revision {version}")
    if ".dev" in notes:
        raise RuntimeError("Development release notes leaked into the stable snapshot")

    reference = snapshot / "reference" / "index.html"
    reference_content = _require_file(reference)
    pages = [reference]
    for relative in REPRESENTATIVE_API_PAGES:
        reference_link = str(Path(relative).relative_to("reference"))
        if reference_link not in reference_content:
            raise RuntimeError(f"Reference index does not link to {reference_link}")
        pages.append(snapshot / relative)

    for page in pages:
        content = _require_file(page)
        _validate_local_html_links(snapshot, page, content)


def promote_snapshot(snapshot: Path, published_root: Path, version: str) -> Path:
    """Validate a candidate, then replace the published version with rollback."""
    snapshot = Path(snapshot)
    published_root = Path(published_root)
    validate_snapshot(snapshot, version)

    destination = published_root / version
    backup = published_root / f".{version}.backup"
    if backup.exists():
        raise RuntimeError(f"Refusing to overwrite stale snapshot backup: {backup}")

    had_destination = destination.exists()
    if had_destination:
        os.replace(destination, backup)
    try:
        os.replace(snapshot, destination)
    except Exception:
        if had_destination and backup.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)
    return destination


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def rebuild_snapshot(
    project_root: Path,
    published_root: Path,
    tag: str,
    *,
    promote: bool,
) -> Path:
    """Build a tag with resolver-selected compatible plugins in an isolated venv."""
    match = CORE_TAG_RE.fullmatch(tag)
    if match is None:
        raise ValueError(f"Expected a final core tag, got {tag!r}")
    version = match.group("version")

    project_root = Path(project_root).resolve()
    published_root = Path(published_root).resolve()
    published_root.mkdir(parents=True, exist_ok=True)
    repair_base = Path(
        tempfile.mkdtemp(
            prefix=f"spectrochempy-docs-{version}-",
            dir=os.environ.get("RUNNER_TEMP"),
        )
    )
    repair_venv = repair_base / "venv"
    repair_build = published_root / ".stable-repair"
    candidate = repair_build / "html" / version
    shutil.rmtree(repair_build, ignore_errors=True)

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to reconstruct stable documentation")

    try:
        _run(
            [uv, "venv", "--python", sys.executable, str(repair_venv)], cwd=project_root
        )
        repair_python = repair_venv / "bin" / "python"
        _run(
            [uv, "pip", "install", "--python", str(repair_python), "-e", ".[docs]"],
            cwd=project_root,
        )
        _run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(repair_python),
                f"spectrochempy=={version}",
                *PLUGIN_PACKAGES,
            ],
            cwd=project_root,
        )

        diagnostics = (
            "from importlib.metadata import version; "
            f"packages={('spectrochempy', *PLUGIN_PACKAGES)!r}; "
            "print('Stable documentation environment:'); "
            "[print(f'  {package}=={version(package)}') for package in packages]; "
            f"assert version('spectrochempy') == {version!r}"
        )
        _run([str(repair_python), "-c", diagnostics], cwd=project_root)

        build_env = os.environ.copy()
        build_env.pop("SETUPTOOLS_SCM_PRETEND_VERSION", None)
        build_env.update(
            {
                "VIRTUAL_ENV": str(repair_venv),
                "PATH": f"{repair_venv / 'bin'}{os.pathsep}{build_env['PATH']}",
                "SCPY_BUILDDIR": str(repair_build),
                "SCPY_DOCS_CONTEXT": "archive",
            }
        )
        _run(
            [
                str(repair_python),
                str(project_root / "docs" / "make.py"),
                "--no-exec",
                "--no-sync",
                "-j1",
                "html",
                "-T",
                tag,
            ],
            cwd=project_root,
            env=build_env,
        )
        validate_snapshot(candidate, version)
        if promote:
            destination = promote_snapshot(candidate, published_root, version)
            shutil.rmtree(repair_build, ignore_errors=True)
            return destination
        return candidate
    finally:
        shutil.rmtree(repair_base, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--snapshot", type=Path, required=True)
    validate.add_argument("--version", required=True)

    promote = subparsers.add_parser("promote")
    promote.add_argument("--snapshot", type=Path, required=True)
    promote.add_argument("--published-root", type=Path, required=True)
    promote.add_argument("--version", required=True)

    rebuild = subparsers.add_parser("rebuild")
    rebuild.add_argument("--project-root", type=Path, required=True)
    rebuild.add_argument("--published-root", type=Path, required=True)
    rebuild.add_argument("--tag", required=True)
    rebuild.add_argument("--promote", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate":
        validate_snapshot(args.snapshot, args.version)
        print(f"Validated stable documentation snapshot {args.version}")
    elif args.command == "promote":
        destination = promote_snapshot(args.snapshot, args.published_root, args.version)
        print(f"Promoted stable documentation snapshot to {destination}")
    else:
        result = rebuild_snapshot(
            args.project_root,
            args.published_root,
            args.tag,
            promote=args.promote,
        )
        print(f"Validated stable documentation reconstruction at {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
