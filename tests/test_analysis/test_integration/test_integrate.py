# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""Tests integration"""

import numpy as np
from numpy.testing import assert_allclose

import spectrochempy as scp


def test_integrate():
    y = scp.Coord(np.arange(4.0), title="sample", units="s")
    x = scp.Coord(np.linspace(0.0, 4.0, 9), title="wavelength", units="cm^-1")
    xx, yy = np.meshgrid(x.data, y.data)
    dataset = scp.NDDataset(
        xx**2 + 2.0 * yy,
        coordset=[y, x],
        units="absorbance",
        title="synthetic integration surface",
    )

    # default dim='x'
    area_trap = dataset.trapz()
    area_simp = dataset.simps()

    assert area_trap.shape == (4,)
    assert area_trap.dims == ["y"]
    assert area_trap.y.title == "sample"
    assert area_trap.title == "area"

    expected_trap_x = -(21.5 + 8.0 * y.data)
    expected_simp_x = -(64.0 / 3.0 + 8.0 * y.data)
    assert_allclose(area_trap.data, expected_trap_x)
    assert_allclose(area_simp.data, expected_simp_x)
    assert_allclose(area_simp.data, -(64.0 / 3.0 + 8.0 * area_simp.y.data))

    area_trap_x = dataset.trapz(dim="x")
    assert_allclose(area_trap_x.data, area_trap.data)

    area_trap_y = dataset.trapz(dim="y")
    assert area_trap_y.shape == (9,)
    assert area_trap_y.dims == ["x"]
    assert area_trap_y.x.title == "wavelength"

    expected_trap_y = 3.0 * x.data**2 + 9.0
    assert_allclose(area_trap_y.data, expected_trap_y)
    assert area_trap_y.units == dataset.units * dataset.y.units
