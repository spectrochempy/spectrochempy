import pytest

import spectrochempy as scp
from spectrochempy.application.preferences import preferences as prefs

WIREDATA = prefs.datadir / "ramandata" / "wire"

pytestmark = pytest.mark.data


@pytest.fixture(autouse=True)
def _skip_if_no_testdata():
    if not WIREDATA.exists():
        pytest.skip("test data not available (set SCP_TEST_DATA_DOWNLOAD=1)")


def _check_no_aux_m_coord(dataset, expected_ndim):
    """Verify the dataset has no spurious 'm' auxiliary coordinate."""
    assert (
        dataset.ndim == expected_ndim
    ), f"expected {expected_ndim}D, got {dataset.ndim}D"
    assert (
        "m" not in dataset.coordset.names
    ), f"unexpected 'm' coordinate in {dataset.coordset.names}"
    assert hasattr(dataset.meta, "ylst_data"), "YLST metadata missing"
    assert hasattr(dataset.meta, "ylst_title"), "YLST title missing"
    assert hasattr(dataset.meta, "ylst_units"), "YLST units missing"


def test_read_wire():
    # First read a single spectrum (measurement type : single)
    dataset = scp.read_wire("ramandata/wire/sp.wdf")
    _check_no_aux_m_coord(dataset, expected_ndim=2)
    _ = dataset.plot()

    # Now read a series of spectra (measurement type : series) from a Z-depth scan.
    dataset = scp.read_wire("ramandata/wire/depth.wdf")
    _check_no_aux_m_coord(dataset, expected_ndim=2)
    _ = dataset.plot_image()

    # extract a line scan data from a StreamLine HR measurement
    dataset = scp.read_wire("ramandata/wire/line.wdf")
    _check_no_aux_m_coord(dataset, expected_ndim=2)
    _ = dataset.plot_image()

    # finally extract grid scan data from a StreamLine HR measurement
    dataset = scp.read_wire("ramandata/wire/mapping.wdf")
    _check_no_aux_m_coord(dataset, expected_ndim=3)
    _ = dataset.sum(dim=2).plot_image()

    # show spectra if test run as a single pytest test
    # scp.show()
