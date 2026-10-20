from datetime import UTC
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.reporting import FigureReference
from spectrochempy.reporting import _code
from spectrochempy.reporting import _coordinate_extent
from spectrochempy.reporting import _escape
from spectrochempy.reporting import render_dataset_markdown


def _dataset():
    coord = scp.Coord([1, 2, 3], title="Wavenumber", units="1/cm")
    dataset = scp.NDDataset(
        [1.0, 2.0, 3.0],
        coordset=[coord],
        name="sample",
        title="Absorbance",
        units="absorbance",
    )
    dataset.origin = "instrument"
    dataset.meta.operator = "Ada"
    dataset.meta.path = Path("/secret/sample")
    dataset.annotate(
        "Imported sample",
        date=datetime(2026, 10, 16, tzinfo=UTC),
    )
    return dataset


def test_to_markdown_report_matches_golden_file_without_mutation():
    dataset = _dataset()
    before = {
        "history": dataset.history_entries,
        "meta": dataset.meta.to_dict().copy(),
        "data": dataset.data.copy(),
    }

    report = dataset.to_markdown_report(
        title="Report",
        metadata=("operator", "path", "missing"),
        figures=(
            FigureReference(
                "Spectrum",
                "figures/spectrum.png",
                "Measured signal.",
            ),
        ),
    )

    expected = (Path(__file__).parent / "fixtures" / "markdown_report.md").read_text()
    assert report == expected
    assert dataset.history_entries == before["history"]
    assert dataset.meta.to_dict() == before["meta"]
    assert (dataset.data == before["data"]).all()


@pytest.mark.parametrize(
    "target",
    [
        "/private/spectrum.png",
        "../spectrum.png",
        "~/spectrum.png",
        "C:\\private\\spectrum.png",
        "C:/private/spectrum.png",
    ],
)
def test_to_markdown_report_rejects_sensitive_figure_paths(target):
    dataset = _dataset()

    with pytest.raises(ValueError, match="safe relative POSIX path"):
        dataset.to_markdown_report(
            title="Report",
            figures=(FigureReference("Spectrum", target),),
        )


def test_to_markdown_report_rejects_malformed_figure_url():
    with pytest.raises(ValueError, match="valid URL"):
        _dataset().to_markdown_report(
            title="Report",
            figures=(FigureReference("Spectrum", "http://[::1"),),
        )


def test_to_markdown_report_rejects_implicit_figures():
    with pytest.raises(TypeError, match="FigureReference"):
        _dataset().to_markdown_report(title="Report", figures=(object(),))


@pytest.mark.parametrize(
    "target",
    [
        "figures/spectrum\n## forged.png",
        "figures/spectrum\r.png",
        "figures/spectrum\x00.png",
        "figures/spectrum\u200b.png",
    ],
)
def test_to_markdown_report_rejects_figure_target_control_characters(target):
    with pytest.raises(ValueError, match="control characters"):
        _dataset().to_markdown_report(
            title="Report",
            figures=(FigureReference("Spectrum", target),),
        )


@pytest.mark.parametrize(
    "target",
    [
        "figures/spectrum(1).png",
        "figures/spectrum<1>.png",
        'figures/spectrum"1".png',
        "figures/spectrum'1'.png",
        "figures/spectrum`1`.png",
    ],
)
def test_to_markdown_report_rejects_figure_target_markdown_syntax(target):
    with pytest.raises(ValueError, match="Markdown control characters"):
        _dataset().to_markdown_report(
            title="Report",
            figures=(FigureReference("Spectrum", target),),
        )


@pytest.mark.parametrize(
    "target",
    [
        "figures/spectrum%2.png",
        "figures/spectrum%zz.png",
        "figures/spectrum%0A.png",
        "figures/spectrum%00.png",
        "figures/spectrum%7F.png",
    ],
)
def test_to_markdown_report_rejects_unsafe_figure_target_percent_encoding(target):
    with pytest.raises(ValueError, match="percent-encoding"):
        _dataset().to_markdown_report(
            title="Report",
            figures=(FigureReference("Spectrum", target),),
        )


@pytest.mark.parametrize(
    "target",
    [
        "figures/my%20spectrum.png",
        "https://example.org/a%20b.png",
        "https://example.org/spectrum.png",
    ],
)
def test_to_markdown_report_accepts_safe_figure_targets(target):
    report = _dataset().to_markdown_report(
        title="Report",
        figures=(FigureReference("Spectrum", target),),
    )

    assert f"![Spectrum]({target})" in report


def test_to_markdown_report_escapes_structural_markdown():
    dataset = _dataset()
    dataset.annotate(
        "hi\n## forged `heading`",
        date=datetime(2026, 10, 17, tzinfo=UTC),
    )

    report = dataset.to_markdown_report(
        title="Report\n## forged",
        figures=(FigureReference("label\n## forged", "figures/spectrum.png"),),
    )

    assert "\n## forged" not in report
    assert "Report\\n\\#\\# forged" in report
    assert "hi\\n\\#\\# forged \\`heading\\`" in report
    assert "label\\n\\#\\# forged" in report


def test_to_markdown_report_redacts_path_objects():
    dataset = _dataset()
    dataset.meta.path = Path("/secret/sample")

    report = dataset.to_markdown_report(title="Report", metadata=("path",))

    assert "- `path`: [redacted: path]" in report
    assert "/secret/sample" not in report


def test_to_markdown_report_does_not_guess_sensitive_strings():
    dataset = _dataset()
    dataset.meta.note = "Stored at /home/user/data.csv"

    report = dataset.to_markdown_report(title="Report", metadata=("note",))

    assert "does not detect or certify" in report
    assert "/home/user/data\\.csv" in report


@pytest.mark.parametrize(
    "value",
    [
        "Measured at 25 C. Baseline corrected.",
        "TopSpin 4.0",
        "600.13 MHz",
        "Dr. Smith",
        "1.5",
        "version 1.12",
        "1.5.3",
        "mg/L",
        "N/A",
        "and/or",
        "24/7",
        "absorbance/a.u.",
        "10.1364/AO.46.000123",
        "A/B calibration",
    ],
)
def test_to_markdown_report_renders_supplied_text(value):
    dataset = _dataset()
    dataset.meta.note = value

    report = dataset.to_markdown_report(title="Report", metadata=("note",))

    metadata_section = report.split("#### Selected metadata")[1]
    assert "[redacted" not in metadata_section
    assert f"- `note`: {_escape(value)}" in metadata_section


def test_to_markdown_report_renders_dotted_origin():
    dataset = _dataset()
    dataset.origin = "Bruker BioSpin 4.0"

    report = dataset.to_markdown_report(title="Report")

    assert "- Origin: Bruker BioSpin 4\\.0" in report


def test_to_markdown_report_preserves_numpy_longdouble_precision():
    dataset = _dataset()
    value = np.longdouble(1) + np.finfo(np.longdouble).eps
    dataset.meta.extended = value

    report = dataset.to_markdown_report(title="Report", metadata=("extended",))

    assert str(value) in report


def test_to_markdown_report_bounds_metadata_collections():
    dataset = _dataset()
    dataset.meta.large = list(range(33))
    dataset.meta.deep = {"one": {"two": {"three": {"four": {"five": 5}}}}}

    report = dataset.to_markdown_report(title="Report", metadata=("large", "deep"))

    assert "[omitted: collection too large]" in report
    assert "[omitted: maximum depth]" in report


def test_to_markdown_report_preserves_history_order_and_redacts_nested_paths():
    dataset = _dataset()
    dataset.annotate(
        "Second entry",
        date=datetime(2026, 10, 15, tzinfo=UTC),
    )
    dataset.meta.nested = {"later": 2, "path": Path("/secret/nested")}

    report = dataset.to_markdown_report(title="Report", metadata=("nested",))

    assert report.index("Imported sample") < report.index("Second entry")
    assert "{later: 2, path: [redacted: path]}" in report


def test_to_markdown_report_marks_absent_origin():
    report = scp.NDDataset([1.0, 2.0, 3.0]).to_markdown_report(title="Report")

    assert "- Origin: —" in report
    assert "- Origin: \n" not in report


def test_to_markdown_report_distinguishes_missing_metadata_from_none():
    dataset = _dataset()
    dataset.meta.present = None

    report = dataset.to_markdown_report(
        title="Report",
        metadata=("present", "missing"),
    )

    assert "- `present`: —" in report
    assert "- `missing`: [absent]" in report


@pytest.mark.parametrize(
    "target",
    [
        "figures/my spectrum.png",
        "figures/my\tspectrum.png",
        "https://example.org/fig a.png",
    ],
)
def test_to_markdown_report_rejects_figure_targets_with_whitespace(target):
    with pytest.raises(ValueError, match="whitespace"):
        _dataset().to_markdown_report(
            title="Report",
            figures=(FigureReference("Spectrum", target),),
        )


def test_to_markdown_report_accepts_https_figure_target():
    report = _dataset().to_markdown_report(
        title="Report",
        figures=(FigureReference("Spectrum", "https://example.org/spectrum.png"),),
    )

    assert "![Spectrum](https://example.org/spectrum.png)" in report


def test_to_markdown_report_escapes_paths_in_titles_and_keys():
    coord = scp.Coord([1.0, 2.0, 3.0], title="/home/user/data.csv")
    dataset = scp.NDDataset(
        [1.0, 2.0, 3.0],
        coordset=[coord],
        title="/home/user/title.csv",
    )
    dataset.meta.nested = {"/secret/key": 1}

    report = dataset.to_markdown_report(
        title="Report",
        metadata=("nested",),
    )

    assert "/home/user/title\\.csv" in report
    assert "/home/user/data\\.csv" in report
    assert "/secret/key: 1" in report


def test_to_markdown_report_uses_minimum_and_maximum_for_non_monotone_coordinate():
    coord = scp.Coord([5.0, 1.0, 3.0], title="Non-monotone")
    dataset = scp.NDDataset([1.0, 2.0, 3.0], coordset=[coord])

    report = dataset.to_markdown_report(title="Report")

    assert "min 1.0; max 5.0" in report


def test_coordinate_extent_ignores_non_finite_values():
    coord = scp.Coord([5.0, np.nan, 3.0], title="With NaN")

    assert _coordinate_extent(coord) == "min 3.0; max 5.0"


def test_coordinate_extent_reports_first_and_last_for_non_numeric_data():
    # Coord only accepts numeric data, so exercise the non-numeric fallback
    # with the minimal interface used by the helper.
    class CategoricalCoord:
        data = np.array(["alpha", "beta", "gamma"])

    assert _coordinate_extent(CategoricalCoord()) == "first alpha; last gamma"


def test_to_markdown_report_renders_multiple_coordinates():
    row = scp.Coord([1.0, 2.0], title="Row")
    column = scp.Coord([1.0, 2.0, 3.0], title="Column")
    dataset = scp.NDDataset(np.zeros((2, 3)), coordset=[row, column])

    report = dataset.to_markdown_report(title="Report")

    assert "- Shape: `(2, 3)`" in report
    assert "- Dimensions: `y, x`" in report
    assert "- `y` — Row; size: 2; unit: —; min 1.0; max 2.0" in report
    assert "- `x` — Column; size: 3; unit: —; min 1.0; max 3.0" in report


def test_to_markdown_report_partial_coordset_does_not_log_error(caplog):
    coord = scp.Coord(np.array([1.0, 2.0, 3.0]), name="x")
    dataset = scp.NDDataset(np.zeros((2, 3)), coordset=[coord])

    with caplog.at_level("ERROR"):
        report = dataset.to_markdown_report(title="Report")

    assert "- `y`: —" in report
    assert "could not find this dimension name" not in caplog.text


def test_to_markdown_report_bounds_total_render_budget():
    def build(depth):
        if depth == 0:
            return "x" * 50
        return {f"k{index}": build(depth - 1) for index in range(32)}

    dataset = _dataset()
    dataset.meta.huge = build(4)

    report = dataset.to_markdown_report(title="Report", metadata=("huge",))

    assert len(report) < 50_000
    assert "[omitted: render budget exhausted]" in report


@pytest.mark.parametrize(
    "value, expected",
    [
        ("normal", "`normal`"),
        ("`x", "`` `x ``"),
        ("x`", "`` x` ``"),
        ("`", "`` ` ``"),
        ("x`y", "``x`y``"),
        ("``", "``` `` ```"),
    ],
)
def test_code_spans_handle_backtick_edges(value, expected):
    assert _code(value) == expected


def test_render_dataset_markdown_rejects_non_dataset():
    with pytest.raises(TypeError, match="NDDataset"):
        render_dataset_markdown(object(), title="Report")


def test_render_dataset_markdown_rejects_blank_title():
    with pytest.raises(ValueError, match="title"):
        render_dataset_markdown(_dataset(), title="   ")


def test_render_dataset_markdown_rejects_string_metadata():
    with pytest.raises(TypeError, match="sequence"):
        render_dataset_markdown(_dataset(), title="Report", metadata="operator")
