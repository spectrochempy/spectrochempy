# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
import pytest

from spectrochempy.core.dataset.baseobjects.ndarray import NDArray
from spectrochempy.core.script import Script
from spectrochempy.core.units import ur
from spectrochempy.utils import testing


def test_compare_ndarrays(IR_dataset_1D):
    nda1 = NDArray(IR_dataset_1D)
    nda2 = nda1.copy()

    assert not nda1._implements("NDDataset")
    testing.assert_ndarray_equal(nda1, nda2)

    # equality does not depend on title
    nda3 = nda1.copy()
    nda3.title = "xxx"
    testing.assert_ndarray_equal(nda1, nda3)

    # should have same units
    nda3.ito("km", force=True)
    with testing.raises(AssertionError):
        testing.assert_ndarray_equal(nda1, nda3)

    # almost equal ndarrays
    nda4 = nda1.copy()
    nda4.data += 1.0e-6
    with testing.raises(AssertionError):
        testing.assert_ndarray_equal(nda1, nda4)
    testing.assert_ndarray_almost_equal(nda1, nda4)
    with testing.raises(AssertionError):
        testing.assert_ndarray_equal(nda1, nda4, decimal=7)

    # data only
    nda5 = nda1.copy()
    nda5.ito("km", force=True)
    with testing.raises(AssertionError):
        testing.assert_ndarray_equal(nda1, nda5)
    testing.assert_ndarray_equal(nda1, nda5, data_only=True)


def test_compare_coords(IR_dataset_2D):
    x1 = IR_dataset_2D.x
    x2 = x1.copy()

    assert x1._implements("Coord")
    testing.assert_coord_equal(x1, x2)

    y1 = IR_dataset_2D.y
    y2 = y1.copy()

    assert y2._implements("Coord")
    testing.assert_coord_equal(y1, y2)

    # equality do depend on title for coordinates
    y3 = y1.copy()
    y3.title = "xxx"
    with testing.raises(AssertionError):
        testing.assert_coord_equal(y1, y3)

    # should have same units
    y2.ito("km", force=True)
    with testing.raises(AssertionError):
        testing.assert_coord_equal(y1, y2)

    x2.ito("km", force=True)
    with testing.raises(AssertionError):
        testing.assert_coord_equal(x1, x2)

    # almost equal coords
    x4 = x1.copy()
    x4.sigdigits = 6
    x4 += 1.0e-6
    testing.assert_coord_almost_equal(x1, x4, decimal=3)


def test_compare_dataset(IR_dataset_1D):
    # dataset comparison

    nd1 = IR_dataset_1D.copy()
    nd2 = nd1.copy()

    testing.assert_dataset_equal(nd1, nd2)

    nd3 = nd1.copy()
    nd3.title = "ddd"

    with testing.raises(AssertionError):
        testing.assert_dataset_equal(nd1, nd3)

    nd4 = nd1.copy()
    nd4.data += 0.001

    with testing.raises(AssertionError):
        testing.assert_dataset_equal(nd1, nd4)

    testing.assert_dataset_almost_equal(nd1, nd4, decimal=3)

    with testing.raises(AssertionError):
        testing.assert_dataset_almost_equal(nd1, nd4, decimal=4)


def test_compare_project(simple_project):
    # project comparison

    proj1 = simple_project.copy()
    proj1.name = "PROJ1"
    proj2 = proj1.copy()
    proj2.name = "PROJ2"

    testing.assert_project_equal(proj1, proj2)

    proj3 = proj2.copy()
    proj3.add_script(Script(content="print()", name="just_a_try"))

    with testing.raises(AssertionError):
        testing.assert_project_equal(proj1, proj3)


def test_compare_units():
    testing.assert_units_equal(ur.km, ur.m)
    with pytest.raises(AssertionError):
        testing.assert_units_equal(ur.km, ur.m, strict=True)

    with pytest.raises(AssertionError):
        testing.assert_units_equal(ur.absorbance, ur.transmittance)
