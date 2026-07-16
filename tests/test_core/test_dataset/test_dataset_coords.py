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
from spectrochempy.utils.testing import (
    assert_array_almost_equal,
    assert_array_equal,
)


def test_nddataset_coordset():
    # init coordinates set at NDDataset initialization
    dx = np.random.random((10, 7, 3))
    coord0 = np.arange(10)
    coord1 = np.arange(7)
    coord2 = np.arange(3) * 100.0
    da = scp.NDDataset(
        dx,
        coordset=(coord0, coord1, coord2),
        title="absorbance",
        coordtitles=["wavelength", "time-on-stream", "temperature"],
        coordunits=["cm^-1", "s", "K"],
    )
    assert da.shape == (10, 7, 3)
    assert da.coordset.titles == ["temperature", "time-on-stream", "wavelength"]
    assert da.coordset.names == ["x", "y", "z"]
    assert da.coordunits == [ur.Unit("K"), ur.Unit("s"), ur.Unit("cm^-1")]
    # order of dims follow data shape, but not necessarily the coord list (
    # which is ordered by name)
    assert da.dims == ["z", "y", "x"]
    assert da.coordset.names == sorted(da.dims)
    # transpose
    dat = da.T
    assert dat.dims == ["x", "y", "z"]
    # dims changed but did not change coords order !
    assert dat.coordset.names == sorted(dat.dims)
    assert dat.coordtitles == da.coordset.titles
    assert dat.coordunits == da.coordset.units

    # too many coordinates
    cadd = scp.Coord(labels=["d%d" % i for i in range(6)])
    coordtitles = ["wavelength", "time-on-stream", "temperature"]
    coordunits = ["cm^-1", "s", None]
    data = scp.NDDataset(
        dx,
        coordset=[coord0, coord1, coord2, cadd, coord2.copy()],
        title="absorbance",
        coordtitles=coordtitles,
        coordunits=coordunits,
    )
    assert data.coordset.titles == coordtitles[::-1]
    assert data.dims == ["z", "y", "x"]
    # with a CoordSet
    c0, c1 = (
        scp.Coord(labels=["d%d" % i for i in range(6)]),
        scp.Coord(data=[1, 2, 3, 4, 5, 6]),
    )
    cc = scp.CoordSet(c0, c1)
    cd = scp.CoordSet(x=cc, y=c1)
    ds = scp.NDDataset([1, 2, 3, 6, 8, 0], coordset=cd, units="m")
    assert ds.dims == ["x"]
    assert ds.x == cc
    ds.history = "essai: 1"
    ds.history = "try:2"
    # wrong type
    with pytest.raises(TypeError):
        ds.coord[1.3]
    # extra coordinates
    with pytest.raises(AttributeError):
        ds.y
    # invalid_length
    coord1 = scp.Coord(np.arange(9), title="wavelengths")  # , units='m')
    coord2 = scp.Coord(np.arange(20), title="time")  # , units='s')
    with pytest.raises(ValueError):
        scp.NDDataset(np.random.random((10, 20)), coordset=(coord1, coord2))


def test_nddataset_coords_indexer():
    dx = np.random.random((10, 100, 10))
    coord0 = np.linspace(4000, 1000, 10)
    coord1 = np.linspace(0, 60, 10)  # wrong length
    coord2 = np.linspace(20, 30, 10)
    with pytest.raises(ValueError):  # wrong length
        da = scp.NDDataset(
            dx,
            coordset=[coord0, coord1, coord2],
            title="absorbance",
            coordtitles=["wavelength", "time-on-stream", "temperature"],
            coordunits=["cm^-1", "s", "K"],
        )
    coord1 = np.linspace(0, 60, 100)
    da = scp.NDDataset(
        dx,
        coordset=[coord0, coord1, coord2],
        title="absorbance",
        coordtitles=["wavelength", "time-on-stream", "temperature"],
        coordunits=["cm^-1", "s", "K"],
    )
    assert da.shape == (10, 100, 10)
    coords = da.coordset
    assert len(coords) == 3
    assert_array_almost_equal(
        da.coordset[2].data, coord0, decimal=2, err_msg="get axis by index failed"
    )
    # we use almost as SpectroChemPy round the coordinate numbers
    assert_array_almost_equal(
        da.coordset["wavelength"].data,
        coord0,
        decimal=2,
        err_msg="get axis by title failed",
    )
    assert_array_almost_equal(
        da.coordset["time-on-stream"].data,
        coord1,
        decimal=3,
        err_msg="get axis by title failed",
    )
    assert_array_almost_equal(da.coordset["temperature"].data, coord2, decimal=3)
    da.coordset["temperature"] += 273.15 * ur.K
    assert_array_almost_equal(
        da.coordset["temperature"].data, coord2 + 273.15, decimal=3
    )


def test_nddataset_set_coordinates(nd2d, ds1):
    # set coordinates all together
    nd = nd2d.copy()
    ny, nx = nd.shape
    nd.set_coordset(x=np.arange(nx), y=np.arange(ny))
    assert nd.dims == ["y", "x"]
    assert nd.x == np.arange(nx)
    nd.transpose(inplace=True)
    assert nd.dims == ["x", "y"]
    assert nd.x == np.arange(nx)
    # set coordinates from tuple
    nd = nd2d.copy()
    ny, nx = nd.shape
    nd.set_coordset(np.arange(ny), np.arange(nx))
    assert nd.dims == ["y", "x"]
    assert nd.x == np.arange(nx)
    nd.transpose(inplace=True)
    assert nd.dims == ["x", "y"]
    assert nd.x == np.arange(nx)
    # set coordinate with one set to None: should work!
    # set coordinates from tuple
    nd = nd2d.copy()
    ny, nx = nd.shape
    nd.set_coordset(np.arange(ny), None)
    assert nd.dims == ["y", "x"]
    assert nd.y == np.arange(ny)
    assert nd.x.is_empty
    nd.transpose(inplace=True)
    assert nd.dims == ["x", "y"]
    assert nd.y == np.arange(ny)
    assert nd.x.is_empty
    assert nd.coordset == scp.CoordSet(np.arange(ny), None)
    nd = nd2d.copy()
    ny, nx = nd.shape
    nd.set_coordset(None, np.arange(nx))
    assert nd.dims == ["y", "x"]
    assert nd.x == np.arange(nx)
    assert nd.y.is_empty
    nd.set_coordset(y=np.arange(ny), x=None)
    # set up a single coordinates
    nd = nd2d.copy()
    ny, nx = nd.shape
    nd.x = np.arange(nx)
    nd.x = np.arange(nx)  # do it again - fix  a bug
    nd.set_coordtitles(y="intensity", x="time")
    assert repr(nd.coordset) == "CoordSet: [x:time, y:intensity]"
    # validation
    with pytest.raises(ValueError):
        nd.x = np.arange(nx + 5)
    with pytest.raises(AttributeError):
        nd.z = None
    # set coordinates all together
    nd = nd2d.copy()
    ny, nx = nd.shape
    nd.coordset = scp.CoordSet(u=np.arange(nx), v=np.arange(ny))
    assert nd.dims != ["u", "v"]  # dims = ['y','x']
    # set dim names
    nd.dims = ["u", "v"]
    nd.set_coordset(**scp.CoordSet(u=np.arange(ny), v=np.arange(nx)))
    assert nd.dims == ["u", "v"]


def test_nddataset_coords_manipulation(dsm):
    dataset = dsm.copy()
    coord0 = dataset.coordset["y"]
    coord0 -= coord0[0]  # remove first element


def test_nddataset_slice_preserves_empty_coord_metadata():
    ds = scp.NDDataset(np.zeros((10, 5)))
    ds.set_coordset(None, None)
    ds.set_coordtitles(y="time", x="wavelength")
    ds.set_coordunits(y="s", x="cm^-1")

    sliced = ds[:5]

    assert sliced.coordtitles == ds.coordtitles
    assert sliced.coordunits == ds.coordunits


def test_nddataset_square_dataset_with_identical_coordinates():
    a = np.random.rand(3, 3)
    c = scp.Coord(np.arange(3) * 0.25, title="time", units="us")
    nd = scp.NDDataset(a, coordset=scp.CoordSet(x=c, y="x"))
    assert nd.x == nd.y


# ### multiple axis
def test_nddataset_multiple_axis(
    ref_ds, coord0, coord1, coord2, coord2b, dsm
):  # dsm is defined in conftest
    ref = ref_ds
    da = dsm.copy()
    coordm = scp.CoordSet(coord2, coord2b)
    # check indexing
    assert da.shape == ref.shape
    coords = da.coordset
    assert len(coords) == 3
    assert coords["z"] == coord0
    assert da.z == coord0
    assert da.coordset["wavenumber"] == coord0
    assert da.wavenumber == coord0
    assert da["wavenumber"] == coord0
    # for multiple coordinates
    assert da.coordset["x"] == coordm
    assert da["x"] == coordm
    assert da.x == coordm
    # but we can also specify, which axis should be returned explicitly
    # by an index or a label
    assert da.coordset["x_1"] == coord2b
    assert da.coordset["x_2"] == coord2
    assert da.coordset["x"].coords[1] == coord2  # if we want to get it by
    # numerical index use coords attribute
    assert da.coordset["x"]._1 == coord2b
    assert da.x["_1"] == coord2b
    assert da["x_1"] == coord2b
    assert da.x_1 == coord2b
    x = da.coordset["x"]
    assert x["temperature"] == coord2
    assert da.coordset["x"]["temperature"] == coord2
    # even simpler we can specify any of the axis title and get it ...
    assert da.coordset["time-on-stream"] == coord1
    assert da.coordset["temperature"] == coord2
    da.coordset["magnetic field"] += 100 * ur.millitesla
    assert da.coordset["magnetic field"] == coord2b + 100 * ur.millitesla


# ### sorting
def test_nddataset_sorting(ds1):  # ds1 is defined in conftest
    dataset = ds1[:3, :3, 0].copy()
    dataset.sort(inplace=True, dim="z")
    labels = np.array(list("abc"))
    assert_array_equal(dataset.coordset["z"].labels, labels)
    # no change because the axis is naturally inverted to force it
    # we need to specify descend
    dataset.sort(
        inplace=True, descend=False, dim="z"
    )  # order value in increasing order
    labels = np.array(list("cba"))
    assert_array_equal(dataset.coordset["z"].labels, labels)
    dataset.sort(inplace=True, dim="z")
    new = dataset.copy()
    new = new.sort(descend=False, inplace=False, dim="z")
    assert_array_equal(new.data, dataset.data[::-1])
    assert new[0, 0] == dataset[-1, 0]
    assert_array_equal(new.coordset["z"].labels, labels)
    assert_array_equal(new.coordset["z"].data, dataset.coordset["z"].data[::-1])
    # check for another dimension
    dataset = ds1.copy()
    new = ds1.copy()
    new.sort(dim="y", inplace=True, descend=False)
    assert_array_equal(new.data, dataset.data)
    assert new[0, 0, 0] == dataset[0, 0, 0]
    new = dataset.copy()
    new.sort(dim="y", inplace=True, descend=True)
    assert_array_equal(new.data, dataset.data[:, ::-1, :])
    assert new[0, -1, 0] == dataset[0, 0, 0]


# ## issue 29
def test_nddataset_issue_29_mulitlabels():
    DS = scp.NDDataset(np.random.rand(3, 4))
    with pytest.raises(ValueError):
        # shape data and label mismatch
        DS.set_coordset(
            DS.y,
            scp.Coord(
                title="xaxis", units="s", data=[1, 2, 3, 4], labels=["a", "b", "c"]
            ),
        )
    c = scp.Coord(
        title="xaxis", units="s", data=[1, 2, 3, 4], labels=["a", "b", "c", "d"]
    )
    DS.set_coordset(x=c)
    c = scp.Coord(
        title="xaxis",
        units="s",
        data=[1, 2, 3, 4],
        labels=[["a", "c", "b", "d"], ["e", "f", "g", "h"]],
    )
    d = DS.y
    DS.set_coordset(d, c)
    DS.x.labels = ["alpha", "beta", "omega", "gamma"]
    assert DS.x.labels.shape == (4, 3)
    # sort
    DS1 = DS.sort(axis=1, by="value", descend=True)
    assert_array_equal(DS1.x, [4, 3, 2, 1])
    # sort
    assert DS.dims == ["y", "x"]
    DS1 = DS.sort(dim="x", by="label", descend=False)
    assert_array_equal(DS1.x, [1, 3, 2, 4])
    DS1 = DS.sort(dim="x", by="label", pos=2, descend=False)
    assert_array_equal(DS1.x, [1, 2, 4, 3])
    DS.sort(dim="y")
    DS.y.labels = ["alpha", "omega", "gamma"]
    DS2 = DS.sort(dim="y")
    assert_array_equal(DS2.y.labels, ["alpha", "gamma", "omega"])


if __name__ == "__main__":
    pytest.main([__file__])
