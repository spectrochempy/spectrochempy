# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

# ruff: noqa: S101

"""Regression tests for public NMR reader metadata overrides."""

import numpy as np
import pytest
from spectrochempy_nmr.readers.read_agilent import read_agilent
from spectrochempy_nmr.readers.read_jeol import read_jeol
from spectrochempy_nmr.readers.read_simpson import read_simpson
from spectrochempy_nmr.readers.read_tecmag import read_tecmag
from spectrochempy_nmr.readers.read_topspin import read_topspin

import spectrochempy as scp


def _reader_cases(tmp_path):
    simpson_path = tmp_path / "metadata-options.spe"
    simpson_path.write_text(
        "SIMP\nNP=4\nSW=10000\nDATA\n1.0 0.0\n0.5 0.25\n0.0 0.5\n-0.25 0.0\nEND\n"
    )

    data_dir = scp.preferences.datadir
    extra_dir = data_dir.parent / "testdata-extra" / "testdata" / "nmrdata"
    return [
        (
            read_jeol,
            extra_dir / "jeol" / "1H.jdf",
            "jeol",
        ),
        (
            read_tecmag,
            extra_dir / "tecmag" / "LiCl_ref1.tnt",
            "tecmag",
        ),
        (read_simpson, simpson_path, "simpson"),
        (
            read_agilent,
            extra_dir / "agilent" / "agilent_1d" / "fid",
            "agilent",
        ),
        (
            read_topspin,
            data_dir
            / "nmrdata"
            / "bruker"
            / "tests"
            / "nmr"
            / "topspin_1d"
            / "1"
            / "fid",
            "topspin",
        ),
    ]


def _assert_scientific_payload_unchanged(actual, control):
    assert isinstance(actual, scp.NDDataset)
    np.testing.assert_array_equal(actual.data, control.data)
    np.testing.assert_array_equal(actual.mask, control.mask)
    assert actual.shape == control.shape
    assert actual.dims == control.dims
    assert actual.coordset == control.coordset
    assert actual.units == control.units
    assert actual.title == control.title
    assert actual.meta == control.meta
    assert actual.name == control.name
    assert actual.filename == control.filename
    assert actual.acquisition_date == control.acquisition_date
    assert (
        actual.history_entries[-1]["operation"]
        == control.history_entries[-1]["operation"]
    )
    assert (
        actual.history_entries[-1]["message"] == control.history_entries[-1]["message"]
    )


@pytest.mark.parametrize("case_index", range(5))
def test_public_reader_metadata_options(tmp_path, case_index):
    reader, path, default_origin = _reader_cases(tmp_path)[case_index]
    if not path.exists():
        pytest.skip(f"NMR test data not available: {path}")

    control = reader(path)
    origin_only = reader(path, origin="custom-origin")
    description_only = reader(path, description="custom description")
    combined = reader(
        path,
        origin="combined-origin",
        description="combined description",
    )

    assert control.origin == default_origin
    assert control.description == ""
    assert origin_only.origin == "custom-origin"
    assert origin_only.description == ""
    assert description_only.origin == default_origin
    assert description_only.description == "custom description"
    assert combined.origin == "combined-origin"
    assert combined.description == "combined description"

    for overridden in (origin_only, description_only, combined):
        _assert_scientific_payload_unchanged(overridden, control)


def test_reader_metadata_options_distinguish_empty_string_and_none(tmp_path):
    path = _reader_cases(tmp_path)[2][1]

    control = read_simpson(path)
    none_values = read_simpson(path, origin=None, description=None)
    empty_values = read_simpson(path, origin="", description="")

    assert none_values.origin == control.origin == "simpson"
    assert none_values.description == control.description == ""
    assert empty_values.origin == ""
    assert empty_values.description == ""
    _assert_scientific_payload_unchanged(none_values, control)
    _assert_scientific_payload_unchanged(empty_values, control)
