# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for the generated list of papers citing SpectroChemPy."""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("pybtex")
pytest.importorskip("latexcodec")

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"
BIBFILE = DOCS / "sources" / "reference" / "bibliography.bib"

sys.path.insert(0, str(DOCS))
import citing_papers  # noqa: E402

BIB = textwrap.dedent(
    """
    @article{smith:2026,
      spectrochempy_citing = {true},
      author = {Smith, Ann and Doe, John},
      title = {First paper},
      journal = {Some Journal},
      volume = {3},
      pages = {10-20},
      year = {2026},
      doi = {10.1000/first},
    }
    @article{smith:2026b,
      spectrochempy_citing = {true},
      author = {Smith, Ann},
      title = {Second paper},
      journal = {Other Journal},
      year = {2026},
      doi = {10.1000/second},
    }
    @phdthesis{adams:2024,
      spectrochempy_citing = {true},
      author = {Adams, Bob},
      title = {A thesis},
      school = {Some University},
      year = {2024},
      url = {https://HAL_THESES/tel-01234567},
    }
    @article{other:2025,
      author = {Other, Carl},
      title = {Not citing},
      journal = {Some Journal},
      year = {2025},
    }
    """
).replace("HAL_THESES", citing_papers.HAL_THESES)


@pytest.fixture
def bibfile(tmp_path):
    path = tmp_path / "bibliography.bib"
    path.write_text(BIB, encoding="utf-8")
    return path


def test_list_format_and_order(bibfile):
    rst = citing_papers.citing_papers_rst(bibfile)
    assert "other:2025" not in rst
    assert rst.index("2026\n====") < rst.index("2024\n====")
    assert rst.index("smith:2026`") < rst.index("smith:2026b`")
    assert (
        "- :cite:t:`smith:2026`. First paper. *Some Journal* **3**, 10–20. "
        "`doi:10.1000/first <https://doi.org/10.1000/first>`__"
    ) in rst
    assert (
        "- :cite:t:`adams:2024`. A thesis. PhD thesis, Some University. "
        f"`HAL: tel-01234567 <https://{citing_papers.HAL_THESES}/tel-01234567>`__"
    ) in rst


def test_source_read_notes_bibliography_dependency(bibfile):
    noted = []

    class Env:
        def note_dependency(self, filename):
            noted.append(filename)

    class App:
        env = Env()

    source = ["Title\n\n.. citing-papers-list\n"]
    citing_papers.citing_papers_source_read(App(), source, bibfile)
    assert noted == [str(bibfile)]
    assert ":cite:t:`smith:2026`" in source[0]
    assert citing_papers.MARKER not in source[0]


def test_year_suffixes(bibfile):
    assert citing_papers.year_suffixes(bibfile) == {
        "smith:2026": "a",
        "smith:2026b": "b",
    }


def test_install_year_suffixes_is_repeatable(bibfile):
    pytest.importorskip("sphinxcontrib.bibtex")
    from pybtex.database import parse_file
    from sphinxcontrib.bibtex.style import template

    entries = parse_file(str(bibfile)).entries
    original = getattr(template.year, "_original_f", template.year.f)
    try:
        citing_papers.install_year_suffixes(bibfile)
        citing_papers.install_year_suffixes(bibfile)
        year = template.year.f
        assert str(year([], {"entry": entries["smith:2026"]})) == "2026a"
        assert str(year([], {"entry": entries["smith:2026b"]})) == "2026b"
        assert str(year([], {"entry": entries["other:2025"]})) == "2025"
    finally:
        template.year.f = original


def test_incremental_build_follows_bibliography(tmp_path, bibfile):
    pytest.importorskip("sphinx")
    pytest.importorskip("sphinxcontrib.bibtex")
    src = tmp_path / "src"
    (src / "reference").mkdir(parents=True)
    bib = src / "reference" / "bibliography.bib"
    bib.write_text(bibfile.read_text(encoding="utf-8"), encoding="utf-8")
    (src / "index.rst").write_text(
        "Root\n====\n\n.. toctree::\n\n   reference/papers\n   reference/bib\n",
        encoding="utf-8",
    )
    (src / "reference" / "papers.rst").write_text(
        "Papers\n======\n\n.. citing-papers-list\n", encoding="utf-8"
    )
    (src / "reference" / "bib.rst").write_text(
        "Bibliography\n============\n\n.. bibliography::\n   :all:\n",
        encoding="utf-8",
    )
    (src / "conf.py").write_text(
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(DOCS)!r})
            extensions = ["sphinxcontrib.bibtex"]
            bibtex_bibfiles = ["reference/bibliography.bib"]
            bibtex_reference_style = "author_year"

            def _papers(app, docname, source):
                if docname == "reference/papers":
                    from citing_papers import citing_papers_source_read
                    citing_papers_source_read(app, source, {str(bib)!r})

            def setup(app):
                app.connect("source-read", _papers)
            """
        ),
        encoding="utf-8",
    )

    def build():
        subprocess.run(
            [sys.executable, "-m", "sphinx", "-q", "-b", "html", "-d",
             str(tmp_path / "doctrees"), str(src), str(tmp_path / "html")],
            check=True,
        )  # fmt: skip
        html = (tmp_path / "html" / "reference" / "papers.html").read_text(
            encoding="utf-8"
        )
        return re.sub(r"\s+", " ", html)

    assert "Second paper" in build()
    bib.write_text(
        bib.read_text(encoding="utf-8").replace("Second paper", "Renamed paper"),
        encoding="utf-8",
    )
    html = build()
    assert "Renamed paper" in html
    assert "Second paper" not in html


def test_repository_bibliography():
    rst = citing_papers.citing_papers_rst(BIBFILE)
    keys = re.findall(r":cite:t:`([^`]+)`", rst)
    assert len(keys) == len(set(keys))
    # Entries that must stay in the list
    for key in ("naik:2023", "naik:2026", "Oviedo:2024"):
        assert key in keys
    assert "Assessment of Chemometric Analysis" in rst
