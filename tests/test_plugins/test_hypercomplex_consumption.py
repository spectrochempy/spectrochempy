# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""Core-side contract for consuming the standalone hypercomplex wheel."""

from __future__ import annotations

import importlib.metadata as metadata
import importlib.util

import numpy as np
import pytest

import spectrochempy as scp

pytestmark = pytest.mark.plugin


def _components():
    return tuple(
        np.arange(8.0).reshape(4, 2) + offset for offset in (1.0, 11.0, 21.0, 31.0)
    )


def test_hypercomplex_entry_point_accessor_and_components():
    """The installed wheel registers the accessor and preserves components."""
    from spectrochempy_hypercomplex import as_quaternion

    entry_points = {
        entry_point.name: entry_point.value
        for entry_point in metadata.entry_points(group="spectrochempy.plugins")
    }
    assert entry_points["hypercomplex"] == (
        "spectrochempy_hypercomplex:HyperComplexPlugin"
    )

    components = _components()
    dataset = scp.NDDataset(as_quaternion(*components))

    assert dataset.hyper.is_quaternion
    for name, expected in zip(("RR", "RI", "IR", "II"), components, strict=True):
        assert np.array_equal(getattr(dataset.hyper, name), expected)


def test_hypercomplex_native_roundtrip_from_consumer(tmp_path):
    """The core consumer preserves the installed plugin's quaternion data."""
    from spectrochempy_hypercomplex import as_quaternion

    components = _components()
    dataset = scp.NDDataset(as_quaternion(*components))
    filename = dataset.save_as(tmp_path / "consumer-quaternion", confirm=False)
    rebuilt = scp.NDDataset.load(filename)

    assert rebuilt._data.dtype.name == "quaternion"
    assert rebuilt.hyper.is_quaternion
    for name, expected in zip(("RR", "RI", "IR", "II"), components, strict=True):
        assert np.array_equal(getattr(rebuilt.hyper, name), expected)


def test_nmr_and_hypercomplex_entry_points_are_available_together():
    """Both installed distributions expose their official entry points."""
    if importlib.util.find_spec("spectrochempy_nmr") is None:
        pytest.skip("requires the optional spectrochempy-nmr plugin")

    entry_points = {
        entry_point.name
        for entry_point in metadata.entry_points(group="spectrochempy.plugins")
    }

    assert {"nmr", "hypercomplex"} <= entry_points
