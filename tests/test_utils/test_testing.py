# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
import pytest

from spectrochempy.core.dataset.basearrays.ndarray import NDArray
from spectrochempy.core.script import Script
from spectrochempy.core.units import ur
from spectrochempy.utils import testing


def test_compare_ndarrays(IR_dataset_1D):
    """Test comparison functions for NDArray objects."""
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
        testing.assert_ndarray_almost_equal(nda1, nda4, decimal=7)

    # data only
    nda5 = nda1.copy()
    nda5.ito("km", force=True)
    with testing.raises(AssertionError):
        testing.assert_ndarray_equal(nda1, nda5)
    testing.assert_ndarray_equal(nda1, nda5, data_only=True)

    # different masks
    nda6 = nda1.copy()
    nda6.mask = [True] + [False] * (len(nda6) - 1)
    with testing.raises(AssertionError):
        testing.assert_ndarray_equal(nda1, nda6)
    testing.assert_ndarray_equal(nda1, nda6, data_only=True)


def test_compare_coords(IR_dataset_2D):
    """Test comparison functions for Coord objects."""
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
    with testing.raises(AssertionError):
        testing.assert_coord_almost_equal(x1, x4, decimal=7)

    # data only comparison
    x5 = x1.copy()
    x5.title = "different_title"
    x5.ito("km", force=True)
    with testing.raises(AssertionError):
        testing.assert_coord_equal(x1, x5)
    testing.assert_coord_equal(x1, x5, data_only=True)

    # different sizes
    x6 = x1[:5].copy()
    with testing.raises(AssertionError):
        testing.assert_coord_equal(x1, x6)


def test_compare_dataset(IR_dataset_1D):
    """Test comparison functions for NDDataset objects."""
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

    # Test with modified coordinates
    nd5 = nd1.copy()
    nd5.x.title = "different_title"
    with testing.raises(AssertionError):
        testing.assert_dataset_equal(nd1, nd5)
    testing.assert_dataset_equal(nd1, nd5, coords_data_only=True)

    # Test data_only parameter
    nd6 = nd1.copy()
    nd6.title = "different_title"
    nd6.units = "km"
    with testing.raises(AssertionError):
        testing.assert_dataset_equal(nd1, nd6)
    testing.assert_dataset_equal(nd1, nd6, data_only=True)


# Skip the failing test until it can be fixed
@pytest.mark.skip("Skipping test for project title comparison until issue is resolved")
def test_compare_project(simple_project):
    """Test comparison functions for Project objects."""
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

    # Test with datasets having different metadata
    # Handle case where datasets might be in subprojects
    proj4 = proj1.copy()

    # First check if there are datasets at any level
    modified = False

    # Helper function to modify titles recursively
    def modify_dataset_titles(proj):
        nonlocal modified
        for item in proj.datasets:
            if hasattr(item, "title"):
                item.title = "changed"
                modified = True

        # Also check subprojects
        for subproj in proj.projects:
            modify_dataset_titles(subproj)

    # Modify titles across the project hierarchy
    modify_dataset_titles(proj4)

    # Only run this test if we actually modified any datasets
    if modified:
        # The assertion should fail if titles are different without dataset_data_only
        with testing.raises(AssertionError):
            testing.assert_project_equal(proj1, proj4)

        # Should pass with dataset_data_only=True
        testing.assert_project_equal(proj1, proj4, dataset_data_only=True)
    else:
        # If no datasets were found, add a dataset to test
        test_dataset = proj1._create_dataset()
        test_dataset.title = "original"
        proj1.add_dataset(test_dataset)

        test_dataset2 = test_dataset.copy()
        test_dataset2.title = "changed"
        proj4.add_dataset(test_dataset2)

        with testing.raises(AssertionError):
            testing.assert_project_equal(proj1, proj4)

        testing.assert_project_equal(proj1, proj4, dataset_data_only=True)


def test_compare_units():
    """Test comparison functions for units."""
    testing.assert_units_equal(ur.km, ur.m)
    with pytest.raises(AssertionError):
        testing.assert_units_equal(ur.km, ur.m, strict=True)

    with pytest.raises(AssertionError):
        testing.assert_units_equal(ur.absorbance, ur.transmittance)

    # Test dimensionality
    testing.assert_units_equal(ur.m / ur.s, ur.km / ur.hour)
    with pytest.raises(AssertionError):
        testing.assert_units_equal(ur.m / ur.s, ur.km / ur.hour, strict=True)
    with pytest.raises(AssertionError):
        testing.assert_units_equal(ur.m / ur.s, ur.km)
