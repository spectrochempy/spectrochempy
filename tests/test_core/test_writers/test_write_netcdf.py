#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

import pytest

import spectrochempy as scp
from spectrochempy.core.units import ur

xr = scp.optional.import_optional_dependency("xarray", errors="ignore")


@pytest.mark.skipif(xr is None, reason="need xarray package to run")
def test_write_netcdf(IR_dataset_2D):

    nd2 = IR_dataset_2D
    nd2[:, 1230.0:920.0] = scp.MASKED

    # add some attribute
    nd2.meta.pression = 34 * ur.pascal
    nd2.meta.temperature = 3000 * ur.K
    assert nd2.meta.temperature == 3000 * ur.K
    assert nd2.temperature == 3000 * ur.K  # alternative way to get the meta attribute

    assert nd2.meta.essai is None  # do not exist in dict
    with pytest.raises(AttributeError):
        nd2.essai2  # can not find this attribute

    # also for the coordinates
    nd2.y.meta.pression = 3 * ur.torr
    assert nd2.y.meta["pression"] == 3 * ur.torr
    assert nd2.y.pression == 3 * ur.torr  # alternative way to get the meta attribute

    assert nd2.y.meta.essai is None  # not found so it is None
    with pytest.raises(AttributeError):
        nd2.y.essai  # can't find such attribute

    nd2.write_netcdf("simple.nc", confirm=False)

    # test opening by xarray
    xnd2 = xr.open_dataarray("simple.nc")
    print(xnd2)
    # assert xnd2.y.attrs["units"] == nd2.y.units
    # assert xnd2.y.attrs["pression"] == nd2.y.meta["pression"].m
    # assert xnd2.y.attrs["pression"] == nd2.y.pression.m
    # assert xnd2.y.attrs["units_pression"] == nd2.y.meta["pression"].u
    #
    # # test opening by the reader
    # nd = scp.read_netcdf("simple.nc")
    # assert nd.units == nd2.y.units
    # #    assert xnd.y.meta.pression == nd2.y.pression
    # assert_dataset_equal(nd, nd2)
