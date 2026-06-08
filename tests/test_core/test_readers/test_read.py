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

DATADIR = prefs.datadir
IRDATA = DATADIR / "irdata"
WODGER = Path(__file__).parent / "ressources" / "omnic" / "wodger.spg"


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


def _skip_if_scpy_data_unreachable(exc):
    import requests

    network_errors = (
        FileNotFoundError,
        OSError,
        TimeoutError,
        requests.exceptions.RequestException,
    )
    if isinstance(exc, network_errors):
        pytest.skip("spectrochempy_data GitHub testdata not reachable")
    raise exc


def _read_scpy_data_or_skip(reader, path):
    try:
        dataset = reader(path)
    except Exception as exc:  # noqa: BLE001
        _skip_if_scpy_data_unreachable(exc)
    if dataset is None:
        pytest.skip("spectrochempy_data GitHub testdata not reachable")
    return dataset


def _requires_irdata():
    if not IRDATA.exists():
        pytest.skip("IR test data not available (set SCP_TEST_DATA_DOWNLOAD=1)")


@pytest.mark.data
def test_read_local_file():
    """Read a local SPG file and verify content."""
    filename = WODGER

    nd1 = scp.read(filename)
    assert nd1.name == "wodger"

    nd2 = scp.read_omnic(filename)
    assert nd1 == nd2


@pytest.mark.data
@pytest.mark.network
def test_read_remote_fallback(tmp_path):
    """Test remote download from GitHub when local file is moved away."""
    _requires_irdata()
    filename = IRDATA / "CO@Mo_Al2O3.SPG"
    backup = tmp_path / filename.name
    filename.replace(backup)
    try:
        nd2 = _read_scpy_data_or_skip(
            scp.read_omnic,
            "irdata/CO@Mo_Al2O3.SPG",
        )
        assert str(nd2) == "NDDataset: [float64] a.u. (shape: (y:19, x:3112))"

        if filename.exists():
            filename.unlink()
        with pytest.raises(FileNotFoundError):
            scp.read_omnic("irdata/nh4y-active.spg")

        if filename.exists():
            filename.unlink()
        nd2 = _read_scpy_data_or_skip(scp.read, "irdata/CO@Mo_Al2O3.SPG")
        assert str(nd2) == "NDDataset: [float64] a.u. (shape: (y:19, x:3112))"
        assert filename.exists()
    finally:
        if filename.exists():
            filename.unlink()
        if backup.exists():
            backup.replace(filename)


def test_read_missing_file(tmp_path):
    """Reading a non-existent local path should raise FileNotFoundError."""
    missing = tmp_path / "does-not-exist.spg"
    with pytest.raises(FileNotFoundError):
        scp.read(missing, local_only=True)


def test_read_invalid_url_type():
    """Reading a non-scpy-readable URL should raise TypeError."""
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
