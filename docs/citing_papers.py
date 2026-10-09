"""
Generate the list of papers citing SpectroChemPy from the bibliography.

Entries of ``bibliography.bib`` carrying the field
``spectrochempy_citing = {true}`` are listed by year, newest year first and
alphabetically by first author within a year, as
"Authors [year]. Title. *Journal* **volume**, pages. link". Adding a citing
paper therefore only requires adding its BibTeX entry with that field.
"""

import codecs
import contextlib
import re
from pathlib import Path

# pybtex and latexcodec come with the documentation dependencies
# (sphinxcontrib-bibtex); they are imported when needed, so that importing
# docs/conf.py (e.g. in tests) does not require them.

CITING_FIELD = "spectrochempy_citing"
MARKER = ".. citing-papers-list"
HAL_THESES = "theses.hal.science"  # codespell:ignore


def _plain(text):
    """Convert BibTeX/LaTeX text to plain Unicode text."""
    import latexcodec  # noqa: F401, PLC0415  (registers the "ulatex" codec)

    text = text or ""
    with contextlib.suppress(ValueError, UnicodeError):
        text = codecs.decode(text, "ulatex")
    text = re.sub(r"[{}]", "", text).replace("--", "–").replace("\\&", "&")
    return " ".join(text.split())


def _pages(pages):
    pages = _plain(pages)
    if not re.search(r"\w", pages):
        return ""
    if re.fullmatch(r"\w+-\w+", pages):
        pages = pages.replace("-", "–")
    return pages


def _source(entry, fields):
    """Journal (or equivalent), volume, and pages of an entry."""
    if entry.type.lower() == "phdthesis":
        return f"PhD thesis, {_plain(fields.get('school', ''))}"
    name = ""
    for key in ("journal", "booktitle", "note", "publisher"):
        if fields.get(key):
            name = _plain(fields[key])
            break
    if not name and fields.get("archiveprefix", "").lower() == "arxiv":
        name = "arXiv"
    source = f"*{name}*" if name else ""
    volume = fields.get("volume", "")
    if volume and volume != fields.get("year"):
        source += f" **{_plain(volume)}**"
    pages = _pages(fields.get("pages", ""))
    if pages:
        source += f", {pages}"
    return source


def _link(entry, fields):
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", fields.get("doi", ""))
    url = fields.get("url", "")
    hal = fields.get("hal_id")
    if not hal and HAL_THESES in url:
        match = re.search(re.escape(HAL_THESES) + r"/(tel-\d+)", url)
        hal = match.group(1) if match else None
    if entry.type.lower() == "phdthesis" and hal:
        return f"`HAL: {hal} <https://{HAL_THESES}/{hal}>`__"
    if doi:
        return f"`doi:{doi} <https://doi.org/{doi}>`__"
    if url:
        return f"`URL <{url}>`__"
    return ""


def _entry_line(key, entry):
    fields = {name.lower(): value for name, value in entry.fields.items()}
    title = _plain(fields.get("title", "")).rstrip(".,")
    source = _source(entry, fields)
    text = ". ".join(part for part in (f":cite:t:`{key}`", title, source) if part)
    if not text.endswith((".*", "?", "!")):
        text += "."
    return f"- {text} {_link(entry, fields)}".rstrip()


def _first_author(entry):
    persons = entry.persons.get("author", [])
    return _plain(" ".join(persons[0].last_names)).lower() if persons else ""


def citing_papers_rst(bibfile):
    """Return the RST list of citing papers, grouped by year."""
    from pybtex.database import parse_file  # noqa: PLC0415

    database = parse_file(str(bibfile))
    by_year = {}
    for key, entry in database.entries.items():
        flag = {name.lower(): value for name, value in entry.fields.items()}.get(
            CITING_FIELD, ""
        )
        if flag.strip().lower() != "true":
            continue
        by_year.setdefault(str(entry.fields.get("year", "")), []).append((key, entry))
    lines = []
    for year in sorted(by_year, reverse=True):
        lines += [year, "=" * max(len(year), 4), ""]
        for key, entry in sorted(
            by_year[year], key=lambda item: _first_author(item[1])
        ):
            lines += [_entry_line(key, entry), ""]
    return "\n".join(lines)


def insert_citing_papers(source, bibfile):
    """Replace the marker comment of ``papers.rst`` with the generated list."""
    if MARKER not in source:
        return source
    return source.replace(MARKER, citing_papers_rst(Path(bibfile)))


def year_suffixes(bibfile):
    """
    Return a/b/... suffixes for entries sharing first author and year.

    Author-year citations of such entries would otherwise be identical,
    e.g. two "Rejman et al. [2026]"; suffixes are assigned in key order.
    """
    from pybtex.database import parse_file  # noqa: PLC0415

    database = parse_file(str(bibfile))
    groups = {}
    for key, entry in database.entries.items():
        year = str(entry.fields.get("year", ""))
        groups.setdefault((_first_author(entry), year), []).append(key)
    suffixes = {}
    for (author, year), keys in groups.items():
        if author and year and len(keys) > 1:
            for index, key in enumerate(sorted(keys, key=str.lower)):
                suffixes[key.lower()] = "abcdefghijklmnopqrstuvwxyz"[index]
    return suffixes


def install_year_suffixes(bibfile):
    """Add the year suffixes to the year shown in author-year citations."""
    from pybtex.richtext import Text  # noqa: PLC0415
    from sphinxcontrib.bibtex.style import template  # noqa: PLC0415

    suffixes = year_suffixes(bibfile)
    original = template.year.f

    def year_with_suffix(children, data):
        text = original(children, data)
        suffix = suffixes.get(data["entry"].key.lower())
        return Text(text, suffix) if suffix else text

    template.year.f = year_with_suffix
