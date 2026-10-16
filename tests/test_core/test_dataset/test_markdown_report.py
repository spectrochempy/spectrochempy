from datetime import UTC
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.reporting import FigureReference


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
    ],
)
def test_to_markdown_report_renders_punctuated_scientific_text(value):
    dataset = _dataset()
    dataset.meta.note = value

    report = dataset.to_markdown_report(title="Rapport", metadata=("note",))

    metadata_section = report.split("#### Metadata sélectionnée")[1]
    assert "[masqué" not in metadata_section
    assert value in report.replace("\\", "")


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
