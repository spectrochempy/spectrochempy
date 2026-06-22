# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from pathlib import Path

import pytest

import spectrochempy as scp
from spectrochempy.application.preferences import preferences as prefs
from spectrochempy.core.dataset.nddataset import NDDataset
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
    assert nd1.origin == "omnic"
    assert nd1.acquisition_date is not None
    assert nd1.y.title == "acquisition timestamp (GMT)"
    assert str(nd1.y.units) == "s"


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
    assert nd.origin == "omnic"

    nd2 = scp.read_omnic(IRDATA / "subdir" / "20-50" / "7_CZ0-100_Pd_21.SPA")
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


def test_read_spg_history_appended():
    """Regression test for #1144: sort history should be appended, not overwrite
    the import history. The history setter appends string values — both entries
    are preserved."""
    nd = scp.read_spg(WODGER, sortbydate=True)
    # History is a list of timestamp-prefixed strings
    history_text = " ".join(nd.history)
    assert "Imported from spg file" in history_text
    assert "Sorted by date" in history_text


def test_return_ifg_validation(tmp_path):
    """Regression test for #1144: invalid return_ifg values must warn clearly.
    The Importer catches exceptions and re-emits them as warnings, so we check
    for the warning."""
    import warnings

    spa_file = tmp_path / "dummy.spa"
    spa_file.write_bytes(b"\x00" * 1024)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = scp.read_spa(spa_file, return_ifg="invalid")
    assert result is None
    assert len(w) >= 1
    assert any("Invalid return_ifg value" in str(warning.message) for warning in w)
