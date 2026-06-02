# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.core.units import ur
from spectrochempy.utils.exceptions import SpectroChemPyError
from spectrochempy.utils.testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_coord_almost_equal,
    assert_coord_equal,
    assert_equal,
    raises,
)


# ################################## TEST
# SLICING################################
def test_nddataset_loc2index(ref_ds, ds1):
    da = ds1
    ref = ref_ds
    assert da.shape == ref.shape
    coords = da.coordset
    assert len(coords) == 3
    assert da._loc2index(3990.0, dim=0) == 0
    assert da._loc2index("j", dim=0) == 9
    with pytest.raises(IndexError):
        da._loc2index("z", 0)  # labels doesn't exist
    da._loc2index(5000, 0)
    assert da._loc2index(5000, 0) == (0, "out_of_limits")  # return the low limit
    assert da._loc2index(0, 0) == (9, "out_of_limits")  # return the high limit index


def test_nddataset_slicing_by_index(ref_ds, ds1):
    da = ds1
    ref = ref_ds
    # case where the index is an integer: the selection is by index starting
    # at zero
    assert da.shape == ref.shape
    coords = da.coordset
    plane0 = da[0]
    # should return a dataset of with dimension x of size 1
    assert type(plane0) == type(da)
    assert plane0.ndim == 3
    assert plane0.shape == (1, 100, 3)
    assert plane0.size == 300
    assert plane0.dims == ["z", "y", "x"]
    assert_coord_equal(plane0.z, da.z[0])
    da1 = plane0.squeeze()
    assert da1.shape == (100, 3)
    assert da1.dims == ["y", "x"]
    plane1 = da[:, 0]
    assert type(plane1) == type(da)
    assert plane1.ndim == 3
    assert plane1.shape == (10, 1, 3)
    assert plane1.size == 30
    assert plane1.dims == ["z", "y", "x"]
    da2 = plane1.squeeze()
    assert da2.dims == ["z", "x"]
    assert_coord_almost_equal(da2.z, coords[-1], decimal=2)  # remember
    # coordinates
    # are ordered by name!
    assert_coord_almost_equal(da2.x, coords[0])
    # another selection
    row0 = plane0[:, 0]
    assert type(row0) == type(da)
    assert row0.shape == (1, 1, 3)
    # and again selection
    element = row0[..., 0]
    assert type(element) == type(da)
    assert element.dims == ["z", "y", "x"]
    # squeeze
    row1 = row0.squeeze()
    assert row1.mask == scp.NOMASK
    row1[0] = scp.MASKED
    assert row1.dims == ["x"]
    assert row1.shape == (3,)
    assert row1.mask.shape == (3,)
    element = row1[..., 0]
    assert element.x == coords[0][0]
    # now a slicing in multi direction
    da[1:4, 10:40:2, :2]
    # now again a slicing in multi direction (full selection)
    da[:, :, -2]
    # now again a slicing in multi direction (ellipsis)
    da[..., -1]
    da[-1, ...]


def test_nddataset_slicing_by_label(ds1):
    da = ds1
    # selection
    planeb = da["b"]
    assert type(planeb) == type(da)
    plane1 = da[1]
    assert_equal(planeb.data, plane1.data)
    assert planeb.ndim == 3
    assert planeb.size == 300
    bd = da["b":"f"]  # the last index is included
    assert bd.shape == (5, 100, 3)
    b1 = da[1:6]
    assert_equal(bd.data, b1.data)
    bc = da["b":"f", :, "hot"]
    assert bc.shape == (5, 100, 1)
    assert bc.z.labels[0] == "b"
    da[
        ..., "hot"
    ]  # TODO: find a way to use such syntax  # hot2 = da[  #  # "x.hot"]  # assert hot == hot2


def test_nddataset_slicing_by_values(ds1):
    da = ds1
    x = da[3000.0]
    assert x.shape == (1, 100, 3)
    y = da[3000.0:2000.0, :, 210.0]
    assert y.shape == (4, 100, 1)
    # slicing by values should also work using reverse order
    da[2000.0:3000.0, :, 210.0]


def test_nddataset_slicing_out_limits(caplog, ds1):
    import logging

    logger = logging.getLogger("SpectroChemPy")
    logger.propagate = True
    caplog.set_level(logging.DEBUG)
    da = ds1
    y1 = da[2000.0]
    assert str(y1) == "NDDataset: [float64] a.u. (shape: (z:1, y:100, x:3))"
    y2 = da[2000]
    assert y2 is None  # as we are out of limits
    assert caplog.records[-1].levelname == "ERROR"
    assert caplog.records[-1].message.startswith(
        "IndexError: Empty array of shape (0, 100, 3) resulted from slicing."
    )
    y3 = da[:, 95:105]
    assert str(y3) == "NDDataset: [float64] a.u. (shape: (z:10, y:5, x:3))"
    da[5000.0:4001.0]
    assert y2 is None  # as we are out of limits
    assert caplog.records[-1].levelname == "ERROR"
    assert caplog.records[-1].message.startswith(
        "IndexError: Empty array of shape (0, 100, 3) resulted from slicing."
    )
    y5 = da[5000.0:3000.0]
    assert str(y5) == "NDDataset: [float64] a.u. (shape: (z:4, y:100, x:3))"


@raises(IndexError)
def test_nddataset_slicing_toomanyindex(ds1):
    da = ds1
    da[:, 3000.0:2000.0, :, 210.0]


def test_nddataset_slicing_by_index_nocoords(ds1):
    da = ds1
    # case where the index is an integer:
    # the selection is by index starting at zero
    da.delete_coordset()  # clear coords
    plane0 = da[1]
    assert type(plane0) == type(da)  # should return a dataset
    assert plane0.ndim == 3
    assert plane0.size == 300


def test_nddataset_slicing_by_location_but_nocoords(ref_ds, ds1):
    da = ds1
    # case where the index is an integer:
    # the selection is by index starting at zero
    da.delete_coordset()  # clear coords
    # this cannot work (no coords for location)
    with pytest.raises(SpectroChemPyError):
        _ = da[3666.7]


# slicing tests
def test_nddataset_simple_slicing():
    d1 = scp.NDDataset(np.ones((5, 5)))
    assert d1.data.shape == (5, 5)
    assert d1.shape == (5, 5)
    d2 = d1[2:3, 2:3]
    assert d2.shape == (1, 1)
    assert d1 is not d2
    d3 = d1[2, 2]
    assert d3.shape == (1, 1)
    assert d3.squeeze().shape == ()
    d3 = d1[0]
    assert d3.shape == (1, 5)

    with pytest.raises(SpectroChemPyError) as exc:
        _ = d1[0 * ur.cm]
    assert (
        exc.value.args[0]
        == "No coords have been defined. Slicing or selection by location (0.0) needs coords definition."
    )


def test_nddataset_slicing_with_mask():
    mask = np.zeros((5, 5)).astype(bool)
    mask[1, 1] = True
    d1 = scp.NDDataset(np.ones((5, 5)), mask=mask)
    assert d1[1].shape == (1, 5)
    assert d1[1, 1].mask


def test_nddataset_slicing_with_mask_units():
    mask = np.zeros((5, 5)).astype(bool)
    mask[1, 1] = True
    d1 = scp.NDDataset(np.ones((5, 5)), mask=mask, units="m")
    assert d1[0].shape == (1, 5)


def test_nddataset_slicing_with_coords(ds1):
    da = ds1.copy()
    da00 = da[0, 0]
    assert da00.shape == (1, 1, 3)
    assert da00.coordset["x"] == da00.coordset[0]
    assert da00.coordset["x"] == da.coordset[0]


def test_slicing_with_quantities(ds1):
    da = ds1.copy()

    da00 = da[1000.0 * ur("cm^-1"), 0]
    assert da00.shape == (1, 1, 3)
    assert da00.coordset["x"] == da00.coordset[0]
    assert da00.coordset["x"] == da.coordset[0]

    with pytest.raises(ValueError):
        _ = da[1000.0 * ur.K, 0]  # wrong units


def test_nddataset_complex_dataset_slicing_by_index():
    na0 = np.array([1.0 + 2.0j, 2.0, 0.0, 0.0, -1.0j, 1j] * 4)
    nd = scp.NDDataset(na0)
    assert nd.shape == (24,)
    assert nd.data.shape == (24,)
    coords = (np.linspace(-10.0, 10.0, 24),)
    nd.set_coordset(coords)
    x1 = nd.x.copy()
    nd.coordset = coords
    x2 = nd.x.copy()
    assert x1 == x2
    # slicing
    nd1 = nd[0]
    assert nd1.shape == (1,)
    assert nd1.data.shape == (1,)
    # slicing range
    nd2 = nd[1:6]
    assert nd2.shape == (5,)
    assert nd2.data.shape == (5,)
    na0 = na0.reshape(6, 4)
    nd = scp.NDDataset(na0)
    coords = scp.CoordSet(np.linspace(-10.0, 10.0, 6), np.linspace(-1.0, 1.0, 4))
    nd.set_coordset(**coords)
    assert nd.shape == (6, 4)
    assert nd.data.shape == (6, 4)
    nd.coordset = coords
    # slicing 2D
    nd1 = nd[0]
    assert nd1.shape == (1, 4)
    assert nd1.data.shape == (1, 4)
    # slicing range
    nd1 = nd[1:3]
    assert nd1.shape == (2, 4)
    assert nd1.data.shape == (2, 4)
    # slicing range
    nd1 = nd[1:3, 0:2]
    assert nd1.shape == (2, 2)
    assert nd1.data.shape == (2, 2)
    nd.set_complex()
    assert nd.shape == (6, 4)
    assert nd.data.shape == (6, 4)


if __name__ == "__main__":
    pytest.main([__file__])
