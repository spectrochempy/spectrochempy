from datetime import UTC
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.reporting import FigureReference
from spectrochempy.reporting import _code
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
        title="Rapport d’essai",
        metadata=("operator", "path", "missing"),
        figures=(
            FigureReference(
                "Spectre",
                "figures/spectrum.png",
                "Signal mesuré.",
            ),
        ),
    )

    expected = (Path(__file__).parent / "fixtures" / "markdown_report.md").read_text()
    assert report == expected
    assert dataset.history_entries == before["history"]
    assert dataset.meta.to_dict() == before["meta"]
    assert (dataset.data == before["data"]).all()


@pytest.mark.parametrize(
    "target", ["/private/spectrum.png", "../spectrum.png", "~/spectrum.png"]
)
def test_to_markdown_report_rejects_sensitive_figure_paths(target):
    dataset = _dataset()

    with pytest.raises(ValueError, match="safe relative POSIX path"):
        dataset.to_markdown_report(
            title="Rapport",
            figures=(FigureReference("Spectre", target),),
        )


def test_to_markdown_report_rejects_implicit_figures():
    with pytest.raises(TypeError, match="FigureReference"):
        _dataset().to_markdown_report(title="Rapport", figures=(object(),))


def test_to_markdown_report_rejects_figure_markdown_injection():
    with pytest.raises(ValueError, match="Markdown control characters"):
        _dataset().to_markdown_report(
            title="Rapport",
            figures=(FigureReference("Spectre", "figures/spectrum\n## forged.png"),),
        )


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
    "path",
    [
        "/private/data",
        "~/private/data",
        "../private/data",
        "report.csv",
        "data.fid",
        "figures/spectrum.png",
        "example.com",
        "private report.csv",
        "figures/my spectrum.png",
        "notes / private.csv",
        "report.spectrochempy",
        "report.123456789",
        "C:\\private\\data",
    ],
)
def test_to_markdown_report_masks_absolute_and_relative_metadata_paths(path):
    dataset = _dataset()
    dataset.meta.path = path

    report = dataset.to_markdown_report(title="Rapport", metadata=("path",))

    assert "[masqué : chemin ou nom de fichier]" in report
    assert path not in report


def test_to_markdown_report_preserves_scientific_text_with_a_slash():
    dataset = _dataset()
    dataset.meta.calibration = "A/B calibration"

    report = dataset.to_markdown_report(title="Rapport", metadata=("calibration",))

    assert "A/B calibration" in report


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
    ],
)
def test_to_markdown_report_renders_punctuated_scientific_text(value):
    dataset = _dataset()
    dataset.meta.note = value

    report = dataset.to_markdown_report(title="Rapport", metadata=("note",))

    metadata_section = report.split("#### Metadata sélectionnée")[1]
    assert "[masqué" not in metadata_section
    assert f"- `note`: {_escape(value)}" in metadata_section


def test_to_markdown_report_renders_dotted_origin():
    dataset = _dataset()
    dataset.origin = "Bruker BioSpin 4.0"

    report = dataset.to_markdown_report(title="Rapport")

    assert "- Origine: Bruker BioSpin 4\\.0" in report


def test_to_markdown_report_preserves_numpy_longdouble_precision():
    dataset = _dataset()
    value = np.longdouble(1) + np.finfo(np.longdouble).eps
    dataset.meta.extended = value

    report = dataset.to_markdown_report(title="Rapport", metadata=("extended",))

    assert str(value) in report


def test_to_markdown_report_bounds_metadata_collections():
    dataset = _dataset()
    dataset.meta.large = list(range(33))
    dataset.meta.deep = {"one": {"two": {"three": {"four": {"five": 5}}}}}

    report = dataset.to_markdown_report(title="Rapport", metadata=("large", "deep"))

    assert "[omis : collection trop grande]" in report
    assert "[omis : profondeur maximale]" in report


def test_to_markdown_report_preserves_history_order_and_masks_paths():
    dataset = _dataset()
    dataset.annotate(
        "Second entry",
        date=datetime(2026, 10, 15, tzinfo=UTC),
    )
    dataset.meta.nested = {"later": 2, "path": Path("/secret/nested")}

    report = dataset.to_markdown_report(title="Rapport", metadata=("nested",))

    assert report.index("Imported sample") < report.index("Second entry")
    assert "{later: 2, path: [masqué : chemin ou nom de fichier]}" in report


def test_to_markdown_report_marks_absent_origin():
    report = scp.NDDataset([1.0, 2.0, 3.0]).to_markdown_report(title="Rapport")

    assert "- Origine: —" in report
    assert "- Origine: \n" not in report


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
            title="Rapport",
            figures=(FigureReference("Spectre", target),),
        )


def test_to_markdown_report_accepts_https_figure_target():
    report = _dataset().to_markdown_report(
        title="Rapport",
        figures=(FigureReference("Spectre", "https://example.org/spectrum.png"),),
    )

    assert "![Spectre](https://example.org/spectrum.png)" in report


def test_to_markdown_report_masks_paths_in_titles_and_keys():
    coord = scp.Coord([1.0, 2.0, 3.0], title="/home/user/data.csv")
    dataset = scp.NDDataset(
        [1.0, 2.0, 3.0],
        coordset=[coord],
        title="/home/user/title.csv",
    )
    dataset.meta.nested = {"/secret/key": 1}

    report = dataset.to_markdown_report(
        title="Rapport",
        metadata=("nested",),
    )

    assert "/home/user" not in report
    assert "/secret/key" not in report
    assert "- Titre: [masqué : chemin ou nom de fichier]" in report
    assert "[masqué : chemin ou nom de fichier]: 1" in report


def test_to_markdown_report_partial_coordset_does_not_log_error(caplog):
    coord = scp.Coord(np.array([1.0, 2.0, 3.0]), name="x")
    dataset = scp.NDDataset(np.zeros((2, 3)), coordset=[coord])

    with caplog.at_level("ERROR"):
        report = dataset.to_markdown_report(title="Rapport")

    assert "- `y`: —" in report
    assert "could not find this dimension name" not in caplog.text


def test_to_markdown_report_bounds_total_render_budget():
    def build(depth):
        if depth == 0:
            return "x" * 50
        return {f"k{index}": build(depth - 1) for index in range(32)}

    dataset = _dataset()
    dataset.meta.huge = build(4)

    report = dataset.to_markdown_report(title="Rapport", metadata=("huge",))

    assert len(report) < 50_000
    assert "[omis : budget de rendu épuisé]" in report


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
        render_dataset_markdown(object(), title="Rapport")


def test_render_dataset_markdown_rejects_blank_title():
    with pytest.raises(ValueError, match="title"):
        render_dataset_markdown(_dataset(), title="   ")


def test_render_dataset_markdown_rejects_string_metadata():
    with pytest.raises(TypeError, match="sequence"):
        render_dataset_markdown(_dataset(), title="Rapport", metadata="operator")
