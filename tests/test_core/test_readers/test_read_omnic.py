# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import pytest

import spectrochempy as scp
from spectrochempy.core import preferences as prefs
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils.testing import assert_dataset_equal

DATADIR = prefs.datadir
IRDATA = DATADIR / "irdata"


# @pytest.mark.skipif(
#     not IRDATA.exists(),
#     reason="Experimental data not available for testing",
# )
def test_read_omnic():
    # Class method opening a dialog (but for test it is preset)
    nd1 = NDDataset.read_omnic(IRDATA / "nh4y-activation.spg")
    assert str(nd1) == "NDDataset: [float64] a.u. (shape: (y:55, x:5549))"

    # API method
    nd2 = scp.read_omnic(IRDATA / "nh4y-activation.spg")
    assert nd1 == nd2

    # It is also possible to use more specific reader function such as
    # `read_spg` , `read_spa` or `read_srs` - they are alias of the read_omnic function.
    l2 = scp.read_spg("wodger.spg", "irdata/nh4y-activation.spg")
    assert len(l2) == 2

    # test read_omnic with byte spg content
    filename_wodger = "wodger.spg"
    with open(DATADIR / filename_wodger, "rb") as fil:
        content = fil.read()
    nd1 = scp.read_omnic(filename_wodger)
    nd2 = scp.read_omnic({filename_wodger: content})
    assert nd1 == nd2

    # Test bytes contents for spa files
    filename = IRDATA / "subdir" / "7_CZ0-100_Pd_101.SPA"
    nds = scp.read_spa(filename)
    with open(IRDATA / "subdir" / filename, "rb") as fil:
        content = fil.read()
    nd = NDDataset.read_spa({filename: content})
    assert_dataset_equal(nd, nds)

    nd = scp.read_spa(IRDATA / "subdir" / "20-50" / "7_CZ0-100_Pd_21.SPA")
    assert str(nd) == "NDDataset: [float64] a.u. (shape: (y:1, x:5549))"

    nd2 = scp.read_spg(
        IRDATA / "subdir" / "20-50" / "7_CZ0-100_Pd_21.SPA"
    )  # wrong protocol but acceptable
    assert nd2 == nd

    # test import sample IFG
    nd = scp.read_spa(IRDATA / "carroucell_samp" / "2-BaSO4_0.SPA", return_ifg="sample")
    assert str(nd) == "NDDataset: [float64] V (shape: (y:1, x:16384))"

    # test import background IFG
    nd = scp.read_spa(
        IRDATA / "carroucell_samp" / "2-BaSO4_0.SPA", return_ifg="background"
    )
    assert str(nd) == "NDDataset: [float64] V (shape: (y:1, x:16384))"

    # import IFG from file without IFG
    a = scp.read_spa(
        IRDATA / "subdir" / "20-50" / "7_CZ0-100_Pd_21.SPA", return_ifg="sample"
    )
    assert a is None

    # rapid_sca series
    a = scp.read_srs("irdata/omnic_series/rapid_scan.srs")
    assert str(a) == "NDDataset: [float64] V (shape: (y:643, x:4160))"

    # rapid_sca series, import bg
    a = scp.read_srs("irdata/omnic_series/rapid_scan.srs", return_bg=True)
    assert str(a) == "NDDataset: [float64] V (shape: (y:1, x:4160))"

    # GC Demo
    a = scp.read_srs("irdata/omnic_series/GC_Demo.srs")
    assert str(a) == "NDDataset: [float64] % (shape: (y:788, x:1738))"

    # high speed series
    a = scp.read_srs("irdata/omnic_series/high_speed.srs")
    assert str(a) == "NDDataset: [float64] a.u. (shape: (y:897, x:13898))"

    # high speed series, import bg
    a = scp.read_srs("irdata/omnic_series/high_speed.srs", return_bg=True)
    assert str(a) == "NDDataset: [float64] unitless (shape: (y:1, x:13898))"
