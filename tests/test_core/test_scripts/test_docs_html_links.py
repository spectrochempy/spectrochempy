from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / "docs" / "tools" / "check_html_links.py"
spec = importlib.util.spec_from_file_location("check_html_links", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def test_nested_links_fragments_downloads_and_images(tmp_path):
    (tmp_path / "index.html").write_text(
        '<a href="guides/page.html#units%20and%20masks">Guide</a>'
    )
    (tmp_path / "guides").mkdir()
    (tmp_path / "guides" / "page.html").write_text(
        '<h1 id="units and masks">Units</h1>'
        '<a href="../index.html">Home</a>'
        '<a href="../example.ipynb?download=1">Notebook</a>'
        '<img src="../plot.png"><a href="#units%20and%20masks">Section</a>'
        '<a name="legacy"></a><a href="#legacy">Legacy anchor</a>'
    )
    (tmp_path / "example.ipynb").write_text("{}")
    (tmp_path / "plot.png").write_bytes(b"image")
    assert checker.check_html_links(tmp_path) == []


def test_missing_anchor_and_resources_are_reported(tmp_path):
    (tmp_path / "index.html").write_text(
        '<a href="page.html#lost-label">Guide</a><img src="missing.png">'
        '<script src="missing.js"></script><link href="missing.css">'
        '<a href="missing.py">Download</a>'
    )
    (tmp_path / "page.html").write_text('<h1 id="existing">Guide</h1>')
    assert set(checker.check_html_links(tmp_path)) == {
        ("index.html", "page.html#lost-label", "missing anchor"),
        ("index.html", "missing.png", "missing file"),
        ("index.html", "missing.js", "missing file"),
        ("index.html", "missing.css", "missing file"),
        ("index.html", "missing.py", "missing file"),
    }


def test_external_links_and_static_templates_are_not_local_pages(tmp_path):
    (tmp_path / "index.html").write_text(
        '<a href="https://example.org/page#anchor">External</a>'
        '<a href="//example.org/page">External</a>'
        '<a href="/stable/">Deployment root</a><a href="mailto:a@example.org">Mail</a>'
        '<a href="#">Top</a>'
    )
    (tmp_path / "_static").mkdir()
    (tmp_path / "_static" / "webpack-macros.html").write_text(
        "<script src=\"{{ pathto('theme.js', 1) }}\"></script>"
    )
    assert checker.check_html_links(tmp_path) == []


def test_empty_build_is_not_a_success(tmp_path):
    with pytest.raises(ValueError, match="No index.html"):
        checker.check_html_links(tmp_path)


@pytest.mark.parametrize(
    ("source", "anchors"),
    [
        (
            "index.rst.tmpl",
            {"home", "spectrochempy-advanced-spectroscopic-data-analysis"},
        ),
        ("userguide/index.rst", {"user-guide-and-tutorials"}),
        ("gettingstarted/examples/index.rst", {"example-s-gallery"}),
        ("devguide/issues.rst", {"contributing-bugs-report"}),
    ],
)
def test_document_heading_anchors_remain_rendered(source, anchors):
    docutils = pytest.importorskip("docutils.core")
    jinja2 = pytest.importorskip("jinja2")
    text = (SCRIPT.parents[1] / "sources" / source).read_text(encoding="utf-8")
    if source.endswith(".tmpl"):
        text = jinja2.Template(text).render(single=False, include_api=False)
    text = text.split(".. toctree::", 1)[0]
    parts = docutils.publish_parts(
        text, writer_name="html5", settings_overrides={"doctitle_xform": False}
    )
    page = checker.Page(parts["body"])
    assert anchors <= page.anchors
