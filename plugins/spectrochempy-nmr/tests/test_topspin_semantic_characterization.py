# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: S101
"""Semantic characterization baseline for the TopSpin reader."""

from __future__ import annotations

import pytest

import spectrochempy as scp
from tests.test_core.test_readers._reader_semantic_helpers import (
    assert_coordinate_semantics,
)
from tests.test_core.test_readers._reader_semantic_helpers import (
    assert_dataset_identity,
)
from tests.test_core.test_readers._reader_semantic_helpers import (
    assert_dataset_provenance,
)
from tests.test_core.test_readers._reader_semantic_helpers import (
    assert_meta_keys_present,
)

pytest.importorskip("spectrochempy_nmr", reason="requires the NMR plugin")

DATADIR = scp.preferences.datadir
NMRDATA = DATADIR / "nmrdata"
NMRDIR = NMRDATA / "bruker" / "tests" / "nmr"


def _require_path(path):
    if not path.exists():
        pytest.skip(f"NMR test data not available: {path}")
    return path


def _read_topspin_or_skip(*args, **kwargs):
    try:
        result = scp.read_topspin(*args, **kwargs)
    except FileNotFoundError as exc:
        pytest.skip(f"NMR test data incomplete: {exc}")
    if result is None:
        pytest.skip("NMR test data could not be read in this environment")
    return result


@pytest.fixture
def topspin_dataset_1d():
    return _read_topspin_or_skip(_require_path(NMRDIR / "topspin_1d" / "1" / "fid"))


@pytest.fixture
def topspin_dataset_2d():
    return _read_topspin_or_skip(_require_path(NMRDIR / "topspin_2d" / "1" / "ser"))


@pytest.mark.data
def test_topspin_1d_currently_sets_origin_filename_typed_acquisition_date_and_meta(
    topspin_dataset_1d,
):
    dataset = topspin_dataset_1d

    assert_dataset_identity(dataset, title="intensity", units="count")
    assert_dataset_provenance(
        dataset,
        origin="topspin",
        filename_name="fid",
        acquisition_date_present=True,
    )
    assert "expno:1" in dataset.name
    assert dataset.history == []

    x = assert_coordinate_semantics(dataset, "x")
    assert x.meta["acquisition_frequency"] is not None
    assert_meta_keys_present(dataset, "date", "datatype", "pathname", "expno")
    assert dataset.meta.readonly


@pytest.mark.data
def test_topspin_2d_currently_uses_runtime_coordinates_and_no_import_history(
    topspin_dataset_2d,
):
    dataset = topspin_dataset_2d

    assert_dataset_identity(dataset, title="intensity", units="count")
    assert_dataset_provenance(
        dataset,
        origin="topspin",
        filename_name="ser",
        acquisition_date_present=True,
    )
    assert dataset.history == []
    assert_coordinate_semantics(dataset, "x")
    y = assert_coordinate_semantics(dataset, "y")
    assert y.title in {"time", None}
    assert y.size == dataset.shape[0]
