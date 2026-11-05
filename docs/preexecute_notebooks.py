"""Separate notebook execution from the diagnostic Read the Docs Sphinx build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from docs.make import BUILDDIR
from docs.make import DOCS
from docs.make import BuildDocumentation

MANIFEST = BUILDDIR / "preexecuted-notebooks.json"
SOURCES = DOCS / "sources"


def _report(message):
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


def _save(manifest):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _should_execute(notebook, default):
    mode = notebook.metadata.get("nbsphinx", {}).get("execute", default)
    if mode not in {"auto", "always", "never"}:
        raise ValueError(f"Invalid nbsphinx execution mode: {mode!r}")
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    return mode == "always" or (
        mode == "auto"
        and any(cell.source for cell in cells)
        and not any(
            cell.get("outputs") or cell.get("execution_count") for cell in cells
        )
    )


def prepare():
    from sphinx.config import eval_config_file
    from sphinx.util.matching import Matcher

    MANIFEST.unlink(missing_ok=True)
    builder = BuildDocumentation()
    builder._prepare_build()
    config = eval_config_file(str(DOCS / "conf.py"), None)
    settings = {
        name: config[f"nbsphinx_{name}"]
        for name in (
            "execute",
            "kernel_name",
            "execute_arguments",
            "allow_errors",
            "timeout",
        )
    }
    if (
        settings["execute"] != "auto"
        or not config["sphinx_gallery_conf"]["plot_gallery"]
    ):
        raise RuntimeError("Expected normal notebook and Gallery execution settings")
    excluded = Matcher(config["exclude_patterns"] + config["nbsphinx_exclude_patterns"])
    notebooks = {}
    for stem in sorted(builder._get_notebook_files()):
        path = stem.with_suffix(".ipynb")
        name = path.relative_to(SOURCES).as_posix()
        if excluded(name):
            continue
        notebook = nbformat.read(path, as_version=4)
        notebooks[name] = {
            "execute": _should_execute(notebook, settings["execute"]),
            "input_sha256": _digest(path),
        }
    if not notebooks:
        raise RuntimeError("No synchronized notebooks found")
    _save({"settings": settings, "notebooks": notebooks})
    _report(
        f"PREPARED {len(notebooks)} notebooks; "
        f"{sum(item['execute'] for item in notebooks.values())} require execution; "
        f"settings={settings}"
    )
    for name, item in notebooks.items():
        _report(f"NOTEBOOK {name} execute={item['execute']}")


def execute(name):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entry = manifest["notebooks"][name]
    path = SOURCES / name
    if "output_sha256" in entry or _digest(path) != entry["input_sha256"]:
        raise RuntimeError(f"Notebook changed or already processed: {name}")
    notebook = nbformat.read(path, as_version=4)
    settings = manifest["settings"]
    metadata = notebook.metadata.get("nbsphinx", {})
    start = time.monotonic()
    _report(f"START {name} execute={entry['execute']}")
    if entry["execute"]:
        source_root = str(DOCS.parent / "src")
        pythonpath = os.environ.get("PYTHONPATH", "").split(os.pathsep)
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [
                source_root,
                *(path for path in pythonpath if path and path != source_root),
            ]
        )
        processor = ExecutePreprocessor(
            kernel_name=settings["kernel_name"],
            extra_arguments=settings["execute_arguments"],
            allow_errors=metadata.get("allow_errors", settings["allow_errors"]),
            timeout=metadata.get("timeout", settings["timeout"]),
            on_cell_start=lambda cell, cell_index: _report(
                f"CELL {name} index={cell_index} type={cell.cell_type}"
            ),
        )
        with tempfile.TemporaryDirectory(prefix="scp-notebook-config-") as config_home:
            previous_config_home = os.environ.get("SCP_CONFIG_HOME")
            os.environ["SCP_CONFIG_HOME"] = config_home
            try:
                processor.preprocess(notebook, {"metadata": {"path": str(path.parent)}})
            finally:
                if previous_config_home is None:
                    os.environ.pop("SCP_CONFIG_HOME", None)
                else:
                    os.environ["SCP_CONFIG_HOME"] = previous_config_home
    notebook.metadata.setdefault("nbsphinx", {})["execute"] = "never"
    nbformat.write(notebook, path)
    entry["output_sha256"] = _digest(path)
    entry["duration"] = time.monotonic() - start
    entry["error_outputs"] = sum(
        output.output_type == "error"
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )
    _save(manifest)
    _report(
        f"END {name} duration={entry['duration']:.3f}s "
        f"error_outputs={entry['error_outputs']}"
    )


def verify():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    actual = {
        path.relative_to(SOURCES).as_posix()
        for path in SOURCES.rglob("*.ipynb")
        if "gallery" not in path.parts and ".ipynb_checkpoints" not in path.parts
    }
    if actual != set(manifest["notebooks"]):
        raise RuntimeError("Notebook inventory changed after preparation")
    for name, entry in manifest["notebooks"].items():
        path = SOURCES / name
        if entry.get("output_sha256") != _digest(path):
            raise RuntimeError(f"Notebook missing, incomplete or changed: {name}")
        if nbformat.read(path, as_version=4).metadata.nbsphinx.execute != "never":
            raise RuntimeError(f"Notebook could execute again: {name}")
    _report(f"VERIFIED {len(actual)} prepared notebooks; no re-execution")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "execute", "verify"))
    parser.add_argument("notebook", nargs="?")
    args = parser.parse_args()
    if args.operation == "execute":
        if not args.notebook:
            parser.error("execute requires a notebook path relative to docs/sources")
        execute(args.notebook)
    elif args.notebook:
        parser.error("only execute accepts a notebook path")
    elif args.operation == "prepare":
        prepare()
    else:
        verify()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
