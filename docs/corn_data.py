"""Identify documentation sections that require the optional public Corn input."""

import json
import os
import re
from pathlib import Path

GALLERY_STEMS = (
    "plot_cross_validation",
    "plot_pls",
    "plot_pls_pipeline",
    "plot_generic_read",
)
NOTICE = (
    "Corn data could not be downloaded. Cells or examples that require Corn "
    "were not executed in this preview. Their source remains available; "
    "other examples are executed normally."
)
NOTICE_RST = (
    ".. admonition:: Corn unavailable\n"
    "   :class: important scpy-corn-unavailable\n\n"
    f"   {NOTICE}\n\n"
)


def require_real_data():
    """Prevent synthetic smoke-test inputs from entering documentation builds."""
    marker = b"SPECTROCHEMPY SYNTHETIC CORN FIXTURE"
    synthetic = os.environ.get("SCPY_CORN_SYNTHETIC") == "1"
    raw = os.environ.get("SCPY_CORN_DATA")
    if raw and Path(raw).is_file():
        with Path(raw).open("rb") as stream:
            synthetic |= marker in stream.read(116)
    archive = os.environ.get("SCPY_CORN_ZIP")
    if archive and Path(archive).is_file():
        from zipfile import ZipFile

        with ZipFile(archive) as bundle:
            for name in bundle.namelist():
                if name.lower().endswith(".mat"):
                    with bundle.open(name) as stream:
                        synthetic |= marker in stream.read(116)
    if synthetic:
        raise RuntimeError(
            "Synthetic Corn is for tests only; documentation requires real Corn"
        )


def unavailable():
    missing = os.environ.get("SCPY_DOCS_CORN_UNAVAILABLE") == "1"
    if missing and (
        os.environ.get("SCPY_CORN_DATA") or os.environ.get("SCPY_CORN_ZIP")
    ):
        raise RuntimeError("A configured Corn fixture must not be silently skipped")
    return missing


def gallery_pattern(pattern):
    if not unavailable():
        return pattern
    stems = "|".join(re.escape(stem) for stem in GALLERY_STEMS)
    return rf"^(?!.*[/\\](?:{stems})\.py$).*{pattern}"


def skip_notebook_cells(notebook):
    if not unavailable():
        return []
    import nbformat  # noqa: PLC0415

    skipped = []
    for cell in notebook.cells:
        tags = list(cell.metadata.get("tags", []))
        if cell.cell_type == "code" and "requires-corn" in tags:
            skipped.append((cell, tags))
            cell.metadata.tags = list(dict.fromkeys([*tags, "skip-execution"]))
            cell.outputs = []
            cell.execution_count = None
    if skipped:
        first = next(
            i for i, cell in enumerate(notebook.cells) if cell is skipped[0][0]
        )
        notebook.cells.insert(
            first, nbformat.v4.new_markdown_cell(f"> **Corn unavailable:** {NOTICE}")
        )
    return skipped


def annotate_gallery(root):
    if not unavailable():
        return []
    omitted = []
    for stem in GALLERY_STEMS:
        matches = list(root.rglob(f"{stem}.rst"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one unexecuted Corn gallery page: {stem}")
        page = matches[0]
        page.write_text(
            NOTICE_RST + page.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        omitted.append(page.relative_to(root).as_posix())
    (root / "corn-unavailable.json").write_text(json.dumps(omitted), encoding="utf-8")
    return omitted


def source_notice(app, docname, source):
    if not unavailable() and os.environ.get("SCPY_DOCS_GALLERY_ARTIFACT") != "1":
        return
    gallery = Path(app.srcdir) / "gettingstarted/examples/gallery"
    marker = gallery / "corn-unavailable.json"
    page = (Path(app.srcdir) / docname).with_suffix(".rst")
    if not marker.is_file() or not page.is_relative_to(gallery):
        return
    omitted = json.loads(marker.read_text(encoding="utf-8"))
    if page.relative_to(gallery).as_posix() in omitted and NOTICE not in source[0]:
        source[0] = NOTICE_RST + source[0]


def check_gallery_notices(app, exception):
    if exception is not None or app.builder.format != "html":
        return
    if not unavailable() and os.environ.get("SCPY_DOCS_GALLERY_ARTIFACT") != "1":
        return
    gallery = Path("gettingstarted/examples/gallery")
    marker = Path(app.srcdir) / gallery / "corn-unavailable.json"
    if not marker.is_file():
        return
    output = (Path(app.outdir) / gallery).resolve()
    for name in json.loads(marker.read_text(encoding="utf-8")):
        page = (output / Path(name).with_suffix(".html")).resolve()
        if not page.is_relative_to(output):
            raise RuntimeError(f"Invalid Corn gallery page: {name}")
        if not page.is_file() or NOTICE not in " ".join(
            page.read_text(encoding="utf-8").split()
        ):
            raise RuntimeError(f"Missing Corn availability notice in {page}")
