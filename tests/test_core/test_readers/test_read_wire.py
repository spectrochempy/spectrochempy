import pytest

import spectrochempy as scp
from spectrochempy.application.preferences import preferences as prefs

WIREDATA = prefs.datadir / "ramandata" / "wire"

pytestmark = pytest.mark.data


@pytest.fixture(autouse=True)
def _skip_if_no_testdata():
    if not WIREDATA.exists():
        pytest.skip("test data not available (set SCP_TEST_DATA_DOWNLOAD=1)")


def test_read_wire():
    # First read a single spectrum (measurement type : single)
    dataset = scp.read_wire("ramandata/wire/sp.wdf")
    _ = dataset.plot()

    # Now read a series of spectra (measurement type : series) from a Z-depth scan.
    dataset = scp.read_wire("ramandata/wire/depth.wdf")
    _ = dataset.plot_image()

    # extract a line scan data from a StreamLine HR measurement
    dataset = scp.read_wire("ramandata/wire/line.wdf")
    _ = dataset.plot_image()

    # finally extract grid scan data from a StreamLine HR measurement
    dataset = scp.read_wire("ramandata/wire/mapping.wdf")
    _ = dataset.sum(dim=2).plot_image()

    # show spectra if test run as a single pytest test
    # scp.show()
