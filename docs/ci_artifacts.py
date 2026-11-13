# ruff: noqa: S603
"""Produce and consume the source artifacts for the documentation CI prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat
from sphinx.application import Sphinx

from docs import preexecute_notebooks as notebooks
from docs.make import BuildDocumentation

SOURCES = notebooks.SOURCES
GALLERY = Path("gettingstarted/examples/gallery")
GALLERY_TIMES = Path("sg_execution_times.rst")
BACKREFERENCES = GALLERY / "backreferences/backreferences_all.json"
KINDS = ("tutorials", "gallery")


def _commit():
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git is required to identify the documentation source")
    return subprocess.check_output(  # noqa: S603
        [git, "rev-parse", "HEAD"], cwd=SOURCES.parent.parent, text=True
    ).strip()


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _files(root, kind):
    if kind == "gallery":
        return [
            *sorted(path for path in (root / GALLERY).rglob("*") if path.is_file()),
            root / GALLERY_TIMES,
        ]
    return sorted(
        path
        for path in root.rglob("*.ipynb")
        if not path.is_relative_to(root / GALLERY)
        and ".ipynb_checkpoints" not in path.parts
    )


def _produce_gallery(*, no_data):
    # Sphinx-Gallery generates its sources during Sphinx initialization.
    BuildDocumentation(nosync=True, nodata=no_data)._prepare_build()
    with tempfile.TemporaryDirectory(prefix="scp-gallery-ci-") as directory:
        work = Path(directory)
        app = Sphinx(
            str(SOURCES),
            str(SOURCES.parent),
            str(work / "html"),
            str(work / "doctrees"),
            "html",
            confoverrides={
                "abort_on_example_error": True,
                "autosummary_generate": False,
                "intersphinx_mapping": {},
            },
            warningiserror=True,
        )
        if app.statuscode:
            raise RuntimeError("Gallery generation failed")


def produce(kind, bundle, *, no_data=False):
    if any(path.exists() for path in _files(SOURCES, kind)):
        raise RuntimeError(f"{kind} producer requires a clean source tree")
    bundle.mkdir(parents=True)
    if kind == "tutorials":
        notebooks.prepare(nodata=no_data, fresh=True)
        manifest = json.loads(notebooks.MANIFEST.read_text(encoding="utf-8"))
        for name in manifest["notebooks"]:
            subprocess.run(  # noqa: S603
                [sys.executable, "-m", "docs.preexecute_notebooks", "execute", name],
                check=True,
            )
        notebooks.verify()
    else:
        _produce_gallery(no_data=no_data)
        path = SOURCES / BACKREFERENCES
        backreferences = json.loads(path.read_text(encoding="utf-8"))
        for examples in backreferences.values():
            for example in examples:
                target = Path(example[2]).relative_to(SOURCES).as_posix()
                example[1] = example[2] = target
        path.write_text(json.dumps(backreferences), encoding="utf-8")

    entries = {}
    for source in _files(SOURCES, kind):
        name = source.relative_to(SOURCES).as_posix()
        target = bundle / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        entries[name] = _digest(target)
    (bundle / "manifest.json").write_text(
        json.dumps({"commit": _commit(), "kind": kind, "files": entries}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _verified_files(bundle, kind, commit):
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["commit"], manifest["kind"]) != (commit, kind):
        raise RuntimeError(f"Unexpected commit or artifact kind in {bundle}")
    root = bundle / "sources"
    files = _files(root, kind)
    inventory = {path.relative_to(root).as_posix(): _digest(path) for path in files}
    if not inventory or inventory != manifest["files"]:
        raise RuntimeError(f"Missing or altered files in {bundle}")
    if kind == "tutorials" and any(
        nbformat.read(path, as_version=4).metadata.get("nbsphinx", {}).get("execute")
        != "never"
        for path in files
    ):
        raise RuntimeError(f"Notebook execution is not disabled in {bundle}")
    if kind == "gallery" and not any(path.match("plot*.rst") for path in files):
        raise RuntimeError(f"No generated examples in {bundle}")
    return root, files


def render(artifacts):
    commit = _commit()
    inputs = [_verified_files(artifacts / kind, kind, commit) for kind in KINDS]
    for kind in KINDS:
        if any(path.exists() for path in _files(SOURCES, kind)):
            raise RuntimeError("Final render requires a clean source tree")
    for root, files in inputs:
        for path in files:
            target = SOURCES / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    if (SOURCES / BACKREFERENCES).exists():
        path = SOURCES / BACKREFERENCES
        backreferences = json.loads(path.read_text(encoding="utf-8"))
        for examples in backreferences.values():
            for example in examples:
                target = (SOURCES / example[2]).resolve()
                if not target.is_relative_to(SOURCES / GALLERY):
                    raise RuntimeError(f"Invalid gallery backreference: {target}")
                example[1] = example[2] = str(target)
        path.write_text(json.dumps(backreferences), encoding="utf-8")
    render_html(gallery_from_artifact=True)


def render_html(*, gallery_from_artifact):
    subprocess.run(
        [
            sys.executable,
            "-u",
            "docs/make.py",
            "html",
            "--no-sync",
            "--no-data",
            "--warning-is-error",
            "-j",
            "auto",
        ],
        env={
            **os.environ,
            "SCPY_DOCS_PREEXECUTED": "1",
            "SCPY_DOCS_GALLERY_ARTIFACT": "1" if gallery_from_artifact else "0",
        },
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    producer = commands.add_parser("produce")
    producer.add_argument("kind", choices=KINDS)
    producer.add_argument("bundle", type=Path)
    producer.add_argument("--no-data", action="store_true")
    consumer = commands.add_parser("render")
    consumer.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    if args.command == "produce":
        produce(args.kind, args.bundle, no_data=args.no_data)
    else:
        render(args.artifacts)


if __name__ == "__main__":
    main()
