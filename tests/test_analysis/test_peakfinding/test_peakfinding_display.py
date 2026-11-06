"""Unit-aware result displays for terminal, notebook and gallery use."""

from html.parser import HTMLParser

import pytest

from spectrochempy.analysis.peakfinding.peakfinding import PeakFindingResult
from spectrochempy.analysis.peakfinding.peakfinding import PeakTable
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import ur


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.headers = 0
        self.rows = 0
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag == "th":
            self.headers += 1
        if tag == "tr":
            self.rows += 1

    def handle_data(self, data):
        self.text.append(data)


@pytest.mark.parametrize("dims", [None, ("y", "x")])
def test_real_peak_results_display_without_changing_data(dims):
    if dims is None:
        dataset = NDDataset(
            [0, 1, 0, 2, 0], coordset=[Coord([0, 1, 2, 3, 4], units="cm^-1")]
        )
    else:
        dataset = NDDataset(
            [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
            coordset=[Coord([1, 2, 3], units="cm^-1"), Coord([4, 5, 6], units="cm^-1")],
            units="absorbance",
        )
    result = dataset.find_peaks(dims=dims, as_result=True)
    before = result.to_dict()
    history = result.history_entries
    for obj in (result, result.table, result.table.top(1, by="height")):
        text = repr(obj)
        html = obj._repr_html_()
        parser = _TableParser()
        parser.feed(html)
        assert parser.headers == len(result.table.columns)
        assert parser.rows == len(obj) + 1
        assert "cm⁻¹" in text
        assert "cm⁻¹" in "".join(parser.text)
    if dims:
        assert "dims=('y', 'x')" in repr(result)
        assert "dims=(&#x27;y&#x27;, &#x27;x&#x27;)" in result._repr_html_()
    assert result.to_dict() == before
    assert result.history_entries == history


def test_display_converts_compatible_units_and_preserves_precision():
    rows = [
        {"position": 1.23456789 * ur.cm, "height": 1.23456789},
        {"position": 20 * ur.mm, "height": None},
    ]
    table = PeakTable(None, rows=rows, columns=("position", "height"))
    text = repr(table)
    parser = _TableParser()
    parser.feed(table._repr_html_())
    assert "position (cm)" in text
    assert "1.23457" in text
    assert "2" in parser.text
    assert table.to_dict() == rows


@pytest.mark.parametrize("count", [0, 10, 12])
def test_display_has_stable_empty_schema_and_bounded_preview(count):
    rows = [{"index": i, "height": float(i)} for i in range(count)]
    table = PeakTable(None, rows=rows, columns=("index", "height"))
    for obj in (table, PeakFindingResult(table, analysis_dims=("y", "x"))):
        parser = _TableParser()
        parser.feed(obj._repr_html_())
        assert parser.headers == 2
        assert parser.rows == min(count, 10) + 1
        assert ("Showing 10 of 12 peaks." in repr(obj)) == (count > 10)
        assert ("Showing 10 of 12 peaks." in obj._repr_html_()) == (count > 10)
    assert len(table.to_dict()) == count


def test_display_escapes_html_and_retains_incompatible_units():
    table = PeakTable(
        None,
        rows=[
            {"<position>": 1 * ur.cm, "label": '<script>alert("x")</script>'},
            {"<position>": 2 * ur.s, "label": "a & b"},
        ],
        columns=("<position>", "label"),
    )
    html = table._repr_html_()
    assert "<script>" not in html
    assert "&lt;position&gt;" in html
    assert "&lt;script&gt;" in html
    assert "a &amp; b" in html
    assert str(1 * ur.cm) in html
    assert str(2 * ur.s) in html


def test_empty_result_display():
    result = PeakFindingResult(None)
    assert "n_peaks=0" in repr(result)
    assert "<table>" in result._repr_html_()
