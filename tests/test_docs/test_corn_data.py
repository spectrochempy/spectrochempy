"""Test explicit, targeted omissions when public Corn data cannot be fetched."""

import re
from pathlib import Path

import pytest

from docs import corn_data


@pytest.fixture
def missing_corn(monkeypatch):
    monkeypatch.setenv("SCPY_DOCS_CORN_UNAVAILABLE", "1")
    monkeypatch.delenv("SCPY_CORN_DATA", raising=False)
    monkeypatch.delenv("SCPY_CORN_ZIP", raising=False)


def test_gallery_only_excludes_corn_examples(missing_corn):
    pattern = corn_data.gallery_pattern("/plot")
    for stem in corn_data.GALLERY_STEMS:
        assert re.search(pattern, f"/work/examples/{stem}.py") is None
    assert re.search(pattern, "/work/examples/plot_raman_modulation.py")


def test_configured_fixture_cannot_be_bypassed(missing_corn, monkeypatch):
    monkeypatch.setenv("SCPY_CORN_DATA", "/missing/configured.mat")
    with pytest.raises(RuntimeError, match="must not be silently skipped"):
        corn_data.unavailable()


def test_normal_execution_pattern_is_unchanged(monkeypatch):
    monkeypatch.delenv("SCPY_DOCS_CORN_UNAVAILABLE", raising=False)
    assert corn_data.gallery_pattern("/plot") == "/plot"


def test_gallery_notices_identify_omitted_pages(missing_corn, tmp_path):
    for stem in corn_data.GALLERY_STEMS:
        (tmp_path / f"{stem}.rst").write_text(f"{stem}\n====\n")
    omitted = corn_data.annotate_gallery(tmp_path)
    assert len(omitted) == 4
    assert all(
        (tmp_path / name).read_text().startswith(".. admonition:: Corn unavailable")
        for name in omitted
    )


def test_tagged_cells_skip_without_disabling_other_cells(missing_corn):
    nbformat = pytest.importorskip("nbformat")
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("print('independent')"),
            nbformat.v4.new_code_cell(
                "raise RuntimeError('Corn')", metadata={"tags": ["requires-corn"]}
            ),
            nbformat.v4.new_code_cell("print('after Corn')"),
        ]
    )
    skipped = corn_data.skip_notebook_cells(notebook)
    assert len(skipped) == 1
    assert skipped[0][1] == ["requires-corn"]
    assert "skip-execution" in skipped[0][0].metadata.tags
    assert notebook.cells[1].cell_type == "markdown"
    assert "Corn unavailable" in notebook.cells[1].source
    assert "skip-execution" not in notebook.cells[-1].metadata.get("tags", [])


def test_corn_notebook_tags_cover_the_dependent_sections():
    jupytext = pytest.importorskip("jupytext")
    root = Path(__file__).resolve().parents[2] / "docs/sources/userguide"
    expected = {
        "analysis/pls.py": 12,
        "processing/baseline.py": 10,
        "importexport/import.py": 1,
    }
    for name, count in expected.items():
        notebook = jupytext.read(root / name)
        tagged = [
            cell
            for cell in notebook.cells
            if "requires-corn" in cell.metadata.get("tags", [])
        ]
        assert len(tagged) == count
        assert any("SCPY_CORN_" in cell.source for cell in tagged)
    baseline = jupytext.read(root / "processing/baseline.py")
    independent = next(cell for cell in baseline.cells if "basc.fit(X)" in cell.source)
    assert "requires-corn" not in independent.metadata.get("tags", [])


def test_gallery_notice_survives_regeneration_and_artifact_transfer(
    missing_corn, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    gallery = tmp_path / "gettingstarted/examples/gallery"
    gallery.mkdir(parents=True)
    for stem in corn_data.GALLERY_STEMS:
        (gallery / f"{stem}.rst").write_text("Source only\n")
    corn_data.annotate_gallery(gallery)
    monkeypatch.delenv("SCPY_DOCS_CORN_UNAVAILABLE")
    monkeypatch.setenv("SCPY_DOCS_GALLERY_ARTIFACT", "1")
    app = SimpleNamespace(srcdir=tmp_path)
    source = ["Regenerated source\n"]
    corn_data.source_notice(app, "gettingstarted/examples/gallery/plot_pls", source)
    assert source[0].startswith(".. admonition:: Corn unavailable")
    corn_data.source_notice(app, "gettingstarted/examples/gallery/plot_pls", source)
    assert source[0].count(corn_data.NOTICE) == 1
    monkeypatch.delenv("SCPY_DOCS_GALLERY_ARTIFACT")
    source = ["Native source\n"]
    corn_data.source_notice(app, "gettingstarted/examples/gallery/plot_pls", source)
    assert source == ["Native source\n"]


def test_transferred_notices_render_in_html(missing_corn, tmp_path, monkeypatch):
    import subprocess
    import sys
    from types import SimpleNamespace

    pytest.importorskip("sphinx")

    sources = tmp_path / "sources"
    gallery = sources / "gettingstarted/examples/gallery"
    gallery.mkdir(parents=True)
    for stem in corn_data.GALLERY_STEMS:
        (gallery / f"{stem}.rst").write_text(f"{stem}\n{'=' * len(stem)}\n")
    corn_data.annotate_gallery(gallery)
    (sources / "index.rst").write_text(
        "Gallery\n=======\n\n.. toctree::\n\n"
        + "".join(
            f"   gettingstarted/examples/gallery/{stem}\n"
            for stem in corn_data.GALLERY_STEMS
        )
    )
    root = Path(__file__).resolve().parents[2]
    (sources / "conf.py").write_text(
        f"import sys\nsys.path.insert(0, {str(root)!r})\n"
        "from docs import corn_data\n"
        "def setup(app):\n"
        "    app.connect('source-read', corn_data.source_notice)\n"
        "    app.connect('build-finished', corn_data.check_gallery_notices)\n"
    )
    monkeypatch.delenv("SCPY_DOCS_CORN_UNAVAILABLE")
    monkeypatch.setenv("SCPY_DOCS_GALLERY_ARTIFACT", "1")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "-b",
            "html",
            str(sources),
            str(tmp_path / "html"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    app = SimpleNamespace(
        srcdir=sources, outdir=tmp_path / "html", builder=SimpleNamespace(format="html")
    )
    for stem in corn_data.GALLERY_STEMS:
        page = tmp_path / "html/gettingstarted/examples/gallery" / f"{stem}.html"
        assert corn_data.NOTICE in page.read_text()
        assert "scpy-corn-unavailable" in page.read_text()
    page.write_text("<p>Notice lost during rendering</p>")
    with pytest.raises(RuntimeError, match="Missing Corn availability notice"):
        corn_data.check_gallery_notices(app, None)
    (gallery / "corn-unavailable.json").write_text('["../outside.rst"]')
    with pytest.raises(RuntimeError, match="Invalid Corn gallery page"):
        corn_data.check_gallery_notices(app, None)
    monkeypatch.delenv("SCPY_DOCS_GALLERY_ARTIFACT")
    corn_data.check_gallery_notices(app, None)
