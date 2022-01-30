#  -*- coding: utf-8 -*-

#  =====================================================================================================================
#    Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#    CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory
#  =====================================================================================================================

import pytest

from spectrochempy.core.dataset.ndarray import NDArray
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.project.project import Project
from spectrochempy.core.scripts.script import Script
from spectrochempy.utils import testing


@pytest.yield_fixture(scope="function")
def simple_project():

    proj = Project(
        # subprojects
        Project(name="P350", label=r"$\mathrm{M_P}\,(623\,K)$"),
        Project(name="A350", label=r"$\mathrm{M_A}\,(623\,K)$"),
        Project(name="B350", label=r"$\mathrm{M_B}\,(623\,K)$"),
        # attributes
        name="project_1",
        label="main project",
    )

    assert proj.projects_names == ["P350", "A350", "B350"]

    ir = NDDataset([1.1, 2.2, 3.3], coordset=[[1, 2, 3]])
    tg = NDDataset([1, 3, 4], coordset=[[1, 2, 3]])
    proj.A350["IR"] = ir
    proj.A350["TG"] = tg
    script_source = (
        "set_loglevel(INFO)\n"
        'info_(f"samples contained in the project are {proj.projects_names}")'
    )

    proj["print_info"] = Script("print_info", script_source)
    return proj


def test_compare_ndarrays(IR_dataset_1D):

    nda1 = NDArray(IR_dataset_1D)
    nda2 = nda1.copy()

    assert not nda1.implements("NDDataset")
    testing.assert_ndarray_equal(nda1, nda2)

    # equality does not depend on title
    nda3 = nda1.copy()
    nda3.title = "xxx"
    testing.assert_ndarray_equal(nda1, nda3)

    # should have same units
    nda3.ito("km", force=True)
    with pytest.raises(AssertionError):
        testing.assert_ndarray_equal(nda1, nda3)

    # almost equal ndarrays
    nda4 = nda1.copy()
    nda4.data += 1.0e-6
    with pytest.raises(AssertionError):
        testing.assert_ndarray_equal(nda1, nda4)
    testing.assert_ndarray_almost_equal(nda1, nda4)
    with pytest.raises(AssertionError):
        testing.assert_ndarray_equal(nda1, nda4, decimal=7)

    # data only
    nda5 = nda1.copy()
    nda5.ito("km", force=True)
    with pytest.raises(AssertionError):
        testing.assert_ndarray_equal(nda1, nda5)
    testing.assert_ndarray_equal(nda1, nda5, data_only=True)


def test_compare_coords(IR_dataset_2D):

    x1 = IR_dataset_2D.x
    x2 = x1.copy()

    assert x1.implements("LinearCoord")
    testing.assert_coord_equal(x1, x2)

    y1 = IR_dataset_2D.y
    y2 = y1.copy()

    assert y2.implements("Coord")
    testing.assert_coord_equal(y1, y2)

    # equality do depend on title for coordinates
    y3 = y1.copy()
    y3.title = "xxx"
    with pytest.raises(AssertionError):
        testing.assert_coord_equal(y1, y3)

    # should have same units
    with testing.assert_produces_warning(check_stacklevel=False):
        y2.ito("km", force=True)  # <- this has no effects on datetime coordinate!

    x2.ito("km", force=True)
    with pytest.raises(AssertionError):
        testing.assert_coord_equal(x1, x2)

    # almost equal coords
    x4 = x1.copy()
    x4.data += 1.0e-6
    with pytest.raises(AssertionError):
        testing.assert_coord_equal(x1, x4)
    testing.assert_coord_almost_equal(x1, x4)
    with pytest.raises(AssertionError):
        testing.assert_coord_equal(x1, x4, decimal=7)


def test_compare_dataset(IR_dataset_1D):
    # dataset comparison

    nd1 = IR_dataset_1D.copy()
    nd2 = nd1.copy()

    testing.assert_dataset_equal(nd1, nd2)

    nd3 = nd1.copy()
    nd3.title = "ddd"

    with pytest.raises(AssertionError):
        testing.assert_dataset_equal(nd1, nd3)

    nd4 = nd1.copy()
    nd4.data += 0.001

    with pytest.raises(AssertionError):
        testing.assert_dataset_equal(nd1, nd4)

    testing.assert_dataset_almost_equal(nd1, nd4, decimal=3)

    with pytest.raises(AssertionError):
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

    with pytest.raises(AssertionError):
        testing.assert_project_equal(proj1, proj3)
