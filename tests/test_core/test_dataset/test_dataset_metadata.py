# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from zoneinfo import ZoneInfoNotFoundError

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.utils.meta import Meta


def test_nddataset_str():
    arr1d = scp.NDDataset([1, 2, 3])
    assert "[float64]" in str(arr1d)
    arr2d = scp.NDDataset(np.array([[1, 2], [3, 4]]))
    assert str(arr2d) == "NDDataset: [float64] unitless (shape: (y:2, x:2))"


def test_nddataset_str_repr(ds1):
    arr1d = scp.NDDataset(np.array([1, 2, 3]))
    assert repr(arr1d).startswith("NDDataset")
    arr2d = scp.NDDataset(np.array([[1, 2], [3, 4]]))
    assert repr(arr2d).startswith("NDDataset")


def test_nddataset_repr_html():
    dx = np.random.random((10, 100, 3))
    coord0 = scp.Coord(
        data=np.linspace(4000.0, 1000.0, 10),
        labels="a b c d e f g h i j".split(),
        mask=None,
        units="cm^-1",
        title="wavelength",
    )
    coord1 = scp.Coord(
        data=np.linspace(0.0, 60.0, 100),
        labels=None,
        mask=None,
        units="s",
        title="time-on-stream",
    )
    coord2 = scp.Coord(
        data=np.linspace(200.0, 300.0, 3),
        labels=["cold", "normal", "hot"],
        mask=None,
        units="K",
        title="temperature",
    )
    da = scp.NDDataset(
        dx, coordset=[coord0, coord1, coord2], title="absorbance", units="absorbance"
    )
    html = da._repr_html_()
    assert html is not None
    assert "NDDataset" in html
    assert "absorbance" in html
    assert "cm" in html or "cm⁻¹" in html


# ### Metadata
def test_nddataset_with_meta(ds1):
    da = ds1.copy()
    meta = Meta()
    meta.essai = ["try_metadata", 10]
    da.meta = meta
    # check copy of meta
    dac = da.copy()
    assert dac.meta == da.meta


# additional tests made following some bug fixes
def test_nddataset_repr_html_bug_undesired_display_complex():
    da = scp.NDDataset([1, 2, 3])
    da.title = "intensity"
    da.description = "Some experimental measurements"
    da.units = "dimensionless"
    assert "(complex)" not in da._repr_html_()


def test_nddataset_timezone():
    from zoneinfo import ZoneInfo

    nd = scp.NDDataset(np.ones((1, 3, 1, 2)), name="value")
    assert nd.timezone is not None
    assert nd.timezone == nd.local_timezone

    # Skip named-timezone checks when the IANA database is absent
    # (e.g. minimal Windows containers or stripped-down Linux images)
    try:
        ZoneInfo("Pacific/Honolulu")
    except ZoneInfoNotFoundError:
        pytest.skip("IANA timezone database not available on this system")

    nd.timezone = "Pacific/Honolulu"
    assert nd.timezone != nd.local_timezone
    with pytest.raises(ZoneInfoNotFoundError):
        nd.timezone = "XXX"


if __name__ == "__main__":
    pytest.main([__file__])
