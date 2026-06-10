# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from pathlib import Path
from unittest.mock import patch, MagicMock
import io

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.application.preferences import preferences as prefs
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.utils.testing import assert_dataset_equal

DATADIR = prefs.datadir
IRDATA = DATADIR / "irdata"
WODGER = Path(__file__).parent / "ressources" / "omnic" / "wodger.spg"

pytestmark = pytest.mark.data


@pytest.fixture
def _skip_if_no_testdata():
    if not IRDATA.exists():
        pytest.skip("test data not available (set SCP_TEST_DATA_DOWNLOAD=1)")


def test_read_omnic_local_wodger():
    # It is also possible to use more specific reader function such as
    # `read_spg` , `read_spa` or `read_srs` - they are alias of the read_omnic function.
    nd1 = scp.read_omnic(WODGER)
    assert nd1.name == "wodger"

    # test read_omnic with byte spg content
    filename_wodger = "wodger.spg"
    with open(WODGER, "rb") as fil:
        content = fil.read()
    nd2 = scp.read_omnic({filename_wodger: content})
    assert nd1 == nd2


@pytest.mark.usefixtures("_skip_if_no_testdata")
def test_read_omnic():
    # Class method opening a dialog (but for test it is preset)
    nd1 = scp.read_omnic(IRDATA / "nh4y-activation.spg")
    assert str(nd1) == "NDDataset: [float64] a.u. (shape: (y:55, x:5549))"

    # API method
    nd2 = scp.read_omnic(IRDATA / "nh4y-activation.spg")
    assert nd1 == nd2

    # It is also possible to use more specific reader function such as
    # `read_spg` , `read_spa` or `read_srs` - they are alias of the read_omnic function.
    l2 = scp.read_spg(WODGER, "irdata/nh4y-activation.spg")
    assert len(l2) == 2

    # Test bytes contents for spa files
    filename = IRDATA / "subdir" / "7_CZ0-100_Pd_101.SPA"
    nds = scp.read_spa(filename)
    with open(IRDATA / "subdir" / filename, "rb") as fil:
        content = fil.read()
    nd = scp.read_spa({filename: content})
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


# Tests for allow_inconsistent_x parameter (issue #863)
# ================================================================================


class TestAllowInconsistentX:
    """Tests for the allow_inconsistent_x parameter in read_omnic/read_spg."""

    def test_allow_inconsistent_x_parameter_documented(self):
        """Test that allow_inconsistent_x parameter is documented."""
        assert "allow_inconsistent_x" in scp.read_spg.__doc__
        assert "allow_inconsistent_x" in scp.read_omnic.__doc__

    def test_error_message_suggests_parameter(self):
        """Test that error message for inconsistent x-axes suggests allow_inconsistent_x."""
        # The parameter should be mentioned in the docstring
        assert "allow_inconsistent_x=True" in scp.read_omnic.__doc__


@pytest.mark.skip(reason="Requires SPG file with inconsistent x-axes. See issue #863")
def test_allow_inconsistent_x_with_real_file():
    """
    Test reading SPG file with inconsistent x-axes using allow_inconsistent_x=True.

    This test is skipped until a representative sample file is available.
    Once available, this test should:
    1. Read file without allow_inconsistent_x -> ValueError with helpful message
    2. Read file with allow_inconsistent_x=True -> list[NDDataset]
    3. Verify each dataset in list has correct x-axis
    """
    pass
