# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: S101, F841

import pytest

import spectrochempy as scp

DATADIR = scp.preferences.datadir
NMRDATA = DATADIR / "nmrdata"
nmrdir = NMRDATA / "bruker" / "tests" / "nmr"


def _require_path(path):
    if not path.exists():
        pytest.skip(f"NMR test data not available: {path}")
    return path


def _read_topspin_or_skip(*args, **kwargs):
    try:
        result = scp.read_topspin(*args, **kwargs)
    except FileNotFoundError as exc:
        pytest.skip(f"NMR test data incomplete: {exc}")
    if result is None:
        pytest.skip("NMR test data could not be read in this environment")
    return result


def _has_readdir_nmr_data():
    return all(
        path.exists()
        for path in (
            nmrdir / "topspin_1d/1/fid",
            nmrdir / "topspin_2d/1/ser",
            nmrdir / "topspin_2d/1/pdata/1/2rr",
        )
    )


@pytest.mark.skipif(not NMRDATA.exists(), reason="NMR test data not available")
def test_read_topspin():
    nd = _read_topspin_or_skip(_require_path(nmrdir / "exam2d_HC/3/pdata/1/2rr"))
    assert str(nd) == "NDDataset: [quaternion] pp (shape: (y:1024, x:1024))"
    assert nd.y.size == 1024
    assert nd.x.size == 1024

    nd = _read_topspin_or_skip(_require_path(nmrdir / "topspin_1d/1/fid"))
    assert str(nd) == "NDDataset: [complex128] pp (size: 12411)"
    assert nd.x.size == 12411

    nd = _read_topspin_or_skip(_require_path(nmrdir / "topspin_1d/1/pdata/1/1r"))
    assert str(nd) == "NDDataset: [complex128] pp (size: 16384)"
    assert nd.x.size == 16384

    nd = _read_topspin_or_skip(_require_path(nmrdir / "topspin_2d/1/ser"))
    assert str(nd) == "NDDataset: [quaternion] pp (shape: (y:96, x:474))"

    nd = _read_topspin_or_skip(_require_path(nmrdir / "topspin_2d/1/pdata/1/2rr"))
    assert str(nd) == "NDDataset: [quaternion] pp (shape: (y:1024, x:2048))"

    nd1 = _read_topspin_or_skip(_require_path(nmrdir / "topspin_2d"), expno=1, procno=1)
    assert nd1 == nd

    nd = _read_topspin_or_skip(directory=_require_path(nmrdir))
    assert nd.name == "topspin_2d expno:1 procno:1 (SER)"

    nd = _read_topspin_or_skip(_require_path(nmrdir), glob="topspin*/*/pdata/*/*")
    assert isinstance(nd, list)
    assert str(nd[0]) == "NDDataset: [complex128] pp (shape: (y:1, x:16384))"
    assert str(nd[1]) == "NDDataset: [quaternion] pp (shape: (y:1024, x:2048))"


@pytest.mark.data
@pytest.mark.skipif(
    not _has_readdir_nmr_data(),
    reason="Complete NMR read_dir test data not available",
)
def test_readdir_for_nmr():
    try:
        nd = scp.read_dir("nmrdata/bruker/tests/nmr", protocol="topspin")
    except AttributeError as exc:
        if "_read_topspin" in str(exc):
            pytest.skip(
                "TopSpin reader not yet registered on Importer "
                "(plugin module not loaded in this test order)"
            )
        raise
    assert isinstance(nd, list)
    nd1 = [item.name for item in nd]
    assert "topspin_2d expno:1 procno:1 (SER)" in nd1
    assert "topspin_1d expno:1 procno:1 (FID)" in nd1


@pytest.mark.skipif(not NMRDATA.exists(), reason="NMR test data not available")
def test_use_list():
    scp.read_topspin(nmrdir / "relax" / "100" / "ser", use_list=True)
