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
    dataset.meta.path = "/secret/sample"
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

    expected = (Path(__file__).parent / "fixtures" / "markdown_report.md").read_text(
        encoding="utf-8"
    )
    assert report == expected
    assert report == dataset.to_markdown_report(
        title="Report",
        metadata=("operator", "path", "missing"),
        figures=(
            FigureReference("Spectrum", "figures/spectrum.png", "Measured signal."),
        ),
    )
    assert dataset.history_entries == before["history"]
    assert dataset.meta.to_dict() == before["meta"]
    assert (dataset.data == before["data"]).all()


@pytest.mark.parametrize(
    "target",
    [
        "javascript:alert(1)",
        "data:image/png;base64,abc",
        "/spectrum.png",
        "//example.org/a.png",
        "https://",
        "figures/a\n## forged.png",
    ],
)
def test_to_markdown_report_rejects_invalid_figure_targets(target):
    with pytest.raises(ValueError):
        _dataset().to_markdown_report(
            title="Report", figures=(FigureReference("Spectrum", target),)
        )


@pytest.mark.parametrize(
    "target, encoded",
    [
        ("../figures/my spectrum(1).png", "../figures/my%20spectrum%281%29.png"),
        ("./figures/échantillon.png", "./figures/%C3%A9chantillon.png"),
        ("~/spectrum%20one.png", "~/spectrum%20one.png"),
        ("figures/50%.png", "figures/50%.png"),
        ("figures/spectrum#1?.png", "figures/spectrum%231%3F.png"),
        ("https://[::1]/spectrum(1).png", "https://[::1]/spectrum%281%29.png"),
        (
            "https://example.org/a b.png?x=1&y=2",
            "https://example.org/a%20b.png?x=1&y=2",
        ),
        ("http://example.org/spectrum.png", "http://example.org/spectrum.png"),
        ('figures/a[1]"<>.png', "figures/a%5B1%5D%22%3C%3E.png"),
    ],
)
def test_to_markdown_report_encodes_figure_targets(target, encoded):
    report = _dataset().to_markdown_report(
        title="Report", figures=(FigureReference("Spectrum", target),)
    )
    assert f"![Spectrum]({encoded})" in report


def test_to_markdown_report_rejects_implicit_figures():
    with pytest.raises(TypeError, match="FigureReference"):
        _dataset().to_markdown_report(title="Report", figures=(object(),))


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


@pytest.mark.parametrize(
    "caption", ["~~~", "    indented caption", "\tindented caption"]
)
def test_to_markdown_report_captions_cannot_open_markdown_blocks(caption):
    report = _dataset().to_markdown_report(
        title="Report",
        figures=(
            FigureReference("First", "figures/first.png", caption),
            FigureReference("Second", "figures/second.png", "Second caption"),
        ),
    )

    assert f"Caption: {_escape(caption)}" in report
    assert "### Second\n\n![Second](figures/second.png)" in report


@pytest.mark.parametrize(
    "path", [Path("/secret/sample"), Path("measurements") / "sample.csv"]
)
def test_to_markdown_report_renders_path_objects(path):
    dataset = _dataset()
    dataset.meta.path = path

    report = dataset.to_markdown_report(title="Report", metadata=("path",))

    expected = str(path).replace("\\", "\\\\")
    assert f"- `path`: {expected}" in report
    assert "[redacted" not in report


def test_to_markdown_report_does_not_guess_sensitive_strings():
    dataset = _dataset()
    dataset.meta.note = "Stored at /home/user/data.csv"

    report = dataset.to_markdown_report(title="Report", metadata=("note",))

    assert "/home/user/data.csv" in report


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

    assert "- Origin: Bruker BioSpin 4.0" in report


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

    assert "[truncated: 1 more items]" in report
    assert "[truncated: maximum depth]" in report


def test_to_markdown_report_preserves_history_order_and_nested_paths():
    dataset = _dataset()
    dataset.annotate(
        "Second entry",
        date=datetime(2026, 10, 15, tzinfo=UTC),
    )
    dataset.meta.nested = {"later": 2, "path": Path("/secret/nested")}

    report = dataset.to_markdown_report(title="Report", metadata=("nested",))

    assert report.index("Imported sample") < report.index("Second entry")
    expected = str(Path("/secret/nested")).replace("\\", "\\\\")
    assert f"{{later: 2, path: {expected}}}" in report


def test_to_markdown_report_renders_nested_history_parameters_and_sources():
    dataset = _dataset()
    dataset.replace_history(
        [
            {
                "date": datetime(2026, 10, 17, tzinfo=UTC),
                "operation": "custom_processing",
                "parameters": {
                    "model": "example",
                    "settings": {
                        "alpha": 1,
                        "danger\n## key": "safe",
                        "items": ["first", {"note": "second\n## forged"}],
                    },
                    "sources": [
                        {
                            "role": "right",
                            "name": "fitted_baseline",
                            "shape": [3],
                        },
                        {
                            "role": "left",
                            "name": "sample *signal*",
                            "shape": [3],
                        },
                    ],
                },
                "message": "Applied custom processing",
            }
        ]
    )

    report = dataset.to_markdown_report(title="Report")

    assert "##### 1. Custom processing" in report
    assert "- Method: `example`" in report
    assert "- Parameters:\n  - Settings:" in report
    assert "    - Danger\\n\\#\\# key: safe" in report
    assert "        - Note: second\\n\\#\\# forged" in report
    assert "- Sources:\n  - Item 1:" in report
    assert "    - Name: fitted_baseline" in report
    assert "    - Name: sample \\*signal\\*" in report
    assert report.index("    - Role: right") < report.index("    - Role: left")
    assert "Parameters: {" not in report
    assert "\n## forged" not in report


def test_to_markdown_report_bounds_nested_history_parameters():
    dataset = _dataset()
    dataset.replace_history(
        [
            {
                "date": datetime(2026, 10, 17, tzinfo=UTC),
                "operation": "bounded",
                "parameters": {
                    "large": list(range(33)),
                    "deep": {"one": {"two": {"three": {"four": 4}}}},
                },
                "message": "Bounded parameters",
            }
        ]
    )

    report = dataset.to_markdown_report(title="Report")

    assert "[truncated: 1 more items]" in report
    assert "[truncated: maximum depth]" in report


def test_to_markdown_report_preserves_every_operation_in_long_history():
    dataset = _dataset()
    dataset.replace_history(
        [
            {
                "date": datetime(2026, 10, 17, tzinfo=UTC),
                "operation": f"step_{index}",
                "parameters": {},
                "message": f"Step {index}",
            }
            for index in range(300)
        ]
    )

    report = dataset.to_markdown_report(title="Report")

    assert report.count("##### ") == 300
    positions = [report.index(f"  - Operation: `step_{index}`") for index in range(300)]
    assert positions == sorted(positions)
    assert "Step 299" in report


def test_to_markdown_report_identifies_baseline_and_smoothing_methods():
    coord = scp.Coord(
        np.linspace(4000.0, 1000.0, 80),
        title="Wavenumber",
        units="1/cm",
    )
    dataset = scp.NDDataset(
        np.linspace(1.0, 0.0, 80) + np.exp(-0.5 * ((np.arange(80) - 40) / 6) ** 2),
        coordset=[coord],
        name="sample",
        title="Spectrum",
        units="absorbance",
    )
    before = dataset.history_entries

    processed = scp.savgol(dataset.basc(), size=7, order=2)
    report = processed.to_markdown_report(title="Report")

    assert "##### 1. Baseline correction" in report
    assert "- Method: `polynomial`" in report
    assert "##### 2. Savgol" in report
    assert "  - Size: 7" in report
    assert "  - Order: 2" in report
    assert "  - Operation: `baseline_correction`" in report
    assert "  - Operation: `savgol`" in report
    assert "Created using method" not in report
    assert [entry["operation"] for entry in processed.history_entries] == [
        "baseline_correction",
        "savgol",
    ]
    assert dataset.history_entries == before


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

    assert "/home/user/title.csv" in report
    assert "/home/user/data.csv" in report
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


def test_to_markdown_report_truncates_values_without_losing_later_operations():
    dataset = _dataset()
    dataset.meta.note = "signal. Savitzky-Golay " + "x" * 5000
    report = dataset.to_markdown_report(title="Report", metadata=("note",))
    assert "signal. Savitzky-Golay " in report
    assert "[truncated: text too long]" in report
    assert "Imported sample" in report
    assert report.endswith(
        "It is not an executable workflow or a complete provenance record.\n"
    )
    assert "## Scope and limitations" not in report


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


def test_to_markdown_report_sorts_mappings_independently_of_insertion_order():
    dataset = _dataset()
    dataset.meta.settings = {"z": 2, "a": 1}
    first = dataset.to_markdown_report(title="Report", metadata=("settings",))
    dataset.meta.settings = {"a": 1, "z": 2}
    assert first == dataset.to_markdown_report(title="Report", metadata=("settings",))


def test_to_markdown_report_bounds_history_values_and_keeps_following_operation():
    dataset = _dataset()
    dataset.replace_history(
        [
            {
                "date": datetime(2026, 10, 17, tzinfo=UTC),
                "operation": "large",
                "parameters": {
                    "sources": [{"name": str(i)} for i in range(40)],
                    "settings": {str(i): i for i in range(40)},
                },
                "message": "x" * 5000,
            },
            {
                "date": datetime(2026, 10, 17, tzinfo=UTC),
                "operation": "last",
                "parameters": {},
                "message": "Finished",
            },
        ]
    )
    report = dataset.to_markdown_report(title="Report")
    assert report.count("[truncated: 8 more items]") == 2
    assert "[truncated: text too long]" in report
    assert "##### 2. Last" in report
    assert "  - Operation: `last`" in report
    assert "  - Message: Finished" in report


def test_to_markdown_report_keeps_image_labels_and_captions_literal():
    report = _dataset().to_markdown_report(
        title="signal. Savitzky-Golay",
        figures=(
            FigureReference(
                "a](/evil)![b", "spectrum.png", "<script>alert(1)</script>"
            ),
        ),
    )
    assert report.startswith("# signal. Savitzky-Golay\n")
    assert r"![a\](/evil)!\[b](spectrum.png)" in report
    assert "<script>" not in report


def test_to_markdown_report_escapes_windows_path_text():
    dataset = _dataset()
    dataset.meta.path = r"C:\measurements\sample.csv"

    report = dataset.to_markdown_report(title="Report", metadata=("path",))

    assert r"- `path`: C:\\measurements\\sample.csv" in report
