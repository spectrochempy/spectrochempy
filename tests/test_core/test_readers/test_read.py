# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import pytest

import spectrochempy as scp
from spectrochempy.application.preferences import preferences as prefs

DATADIR = prefs.datadir
IRDATA = DATADIR / "irdata"


def _skip_if_eigenvector_unreachable(exc):
    import requests

    network_errors = (
        FileNotFoundError,
        OSError,
        TimeoutError,
        requests.exceptions.RequestException,
    )
    if isinstance(exc, network_errors):
        pytest.skip("eigenvector.com not reachable")
    raise exc


def test_read():
    filename = IRDATA / "CO@Mo_Al2O3.SPG"

    # read normally
    nd1 = scp.read(filename)
    assert str(nd1) == "NDDataset: [float64] a.u. (shape: (y:19, x:3112))"

    nd1 = scp.read_omnic(filename)
    assert str(nd1) == "NDDataset: [float64] a.u. (shape: (y:19, x:3112))"

    # delete file to simulate its absence:
    filename.unlink()

    # now try to download from github s not found locally (use _read_remote)
    nd2 = scp.read_omnic("irdata/CO@Mo_Al2O3.SPG")
    assert str(nd2) == "NDDataset: [float64] a.u. (shape: (y:19, x:3112))"

    # delete file to simulate its absence:
    filename.unlink()

    # now try to download from github s not found locally (use _read_remote)
    # but file doesn't exist on github
    with pytest.raises(FileNotFoundError):
        scp.read_omnic("irdata/nh4y-active.spg")

    # now try a using generic read
    assert not filename.exists()
    nd2 = scp.read("irdata/CO@Mo_Al2O3.SPG")
    assert str(nd2) == "NDDataset: [float64] a.u. (shape: (y:19, x:3112))"
    assert filename.exists()

    # now try a using generic read with a missing
    with pytest.raises(FileNotFoundError):
        scp.read("irdata/nh4y-acti.spg")

    # not a scpy readable type
    with pytest.raises(TypeError):
        scp.read("https://www.spectrochempy.fr/latest/index.html")


@pytest.mark.network
def test_read_eigenvector_corn():
    """Read corn.mat from eigenvector.com."""
    try:
        ds1 = scp.read("http://www.eigenvector.com/data/Corn/corn.mat", merge=False)
        assert len(ds1) == 7
    except Exception as exc:  # noqa: BLE001
        _skip_if_eigenvector_unreachable(exc)

    try:
        ds2 = scp.read_mat("http://www.eigenvector.com/data/Corn/corn.mat", merge=False)
        assert len(ds2) == 7
    except Exception as exc:  # noqa: BLE001
        _skip_if_eigenvector_unreachable(exc)


@pytest.mark.network
def test_read_eigenvector_corn_zip():
    """Read corn.mat_.zip from eigenvector.com."""
    try:
        ds3 = scp.read(
            "https://eigenvector.com/wp-content/uploads/2019/06/corn.mat_.zip",
            merge=False,
        )
        assert len(ds3) == 7
    except Exception as exc:  # noqa: BLE001
        _skip_if_eigenvector_unreachable(exc)


@pytest.mark.network
def test_read_eigenvector_missing():
    """Read a non-existent file from eigenvector.com."""
    try:
        with pytest.raises(FileNotFoundError):
            scp.read("http://www.eigenvector.com/does_not_exist.mat")
    except Exception as exc:  # noqa: BLE001
        _skip_if_eigenvector_unreachable(exc)
