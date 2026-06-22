# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from datetime import datetime
from pathlib import Path

import pytest

import spectrochempy as scp
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.readers.read_labspec import _read_txt

RAMANDIR = scp.preferences.datadir / "ramandata/labspec"

pytestmark = pytest.mark.data


@pytest.fixture(autouse=True)
def _skip_if_no_testdata():
    if not RAMANDIR.exists():
        pytest.skip("test data not available (set SCP_TEST_DATA_DOWNLOAD=1)")


def test_read_labspec():
    # single file
    nd = scp.read_labspec("Activation.txt", directory=RAMANDIR)
    assert nd.shape == (532, 1024)

    # with read_dir
    # First download data as read_dir will not
    scp.read(RAMANDIR / "subdir", replace_existing=False)

    nd = scp.read_dir(directory=RAMANDIR / "subdir")
    assert nd.shape == (6, 1024)

    # empty txt file
    Path("i_am_empty.txt").touch()
    f = Path("i_am_empty.txt")
    nd = scp.read_labspec(f)
    f.unlink()
    assert nd is None


def test_read_labspec_latin1_content():
    content = (
        "#Acq. time (s)=1\n"
        "#Dark correction=No\n"
        "#Acquired=01.01.2024 00:00:01\n"
        "#Accumulations=1\n"
        "#Comment=20\xb0C\n"
        "100\t1\n"
        "101\t2\n"
    ).encode("latin-1")

    nd = _read_txt(NDDataset(), Path("latin_labspec.txt"), content=content)

    assert nd.shape == (1, 2)
    assert nd.meta["Comment"] == "20°C"
    assert nd.origin == "labspec"
    assert nd._acquisition_date == datetime(2024, 1, 1, 0, 0, 0)
    assert nd.y.title == "Time"
    assert str(nd.y.units) == "s"

    # non labspec txt file
    f = Path("i_am_not_labspec.txt")
    f.write_text("blah")
    nd = scp.read_labspec(f)
    f.unlink()
    assert nd is None
