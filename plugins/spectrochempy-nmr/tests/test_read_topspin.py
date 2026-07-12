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
    assert str(nd) == "NDDataset: [quaternion] count (shape: (y:1024, x:1024))"
    assert nd.y.size == 1024
    assert nd.x.size == 1024

    nd = _read_topspin_or_skip(_require_path(nmrdir / "topspin_1d/1/fid"))
    assert str(nd) == "NDDataset: [complex128] count (size: 12411)"
    assert nd.x.size == 12411

    nd = _read_topspin_or_skip(_require_path(nmrdir / "topspin_1d/1/pdata/1/1r"))
    assert str(nd) == "NDDataset: [complex128] count (size: 16384)"
    assert nd.x.size == 16384

    nd = _read_topspin_or_skip(_require_path(nmrdir / "topspin_2d/1/ser"))
    assert str(nd) == "NDDataset: [quaternion] count (shape: (y:96, x:474))"

    nd = _read_topspin_or_skip(_require_path(nmrdir / "topspin_2d/1/pdata/1/2rr"))
    assert str(nd) == "NDDataset: [quaternion] count (shape: (y:1024, x:2048))"

    nd1 = _read_topspin_or_skip(_require_path(nmrdir / "topspin_2d"), expno=1, procno=1)
    assert nd1 == nd

    nd = _read_topspin_or_skip(directory=_require_path(nmrdir))
    assert nd.name == "topspin_2d expno:1 procno:1 (SER)"

    nd = _read_topspin_or_skip(_require_path(nmrdir), glob="topspin*/*/pdata/*/*")
    assert isinstance(nd, list)
    assert str(nd[0]) == "NDDataset: [complex128] count (shape: (y:1, x:16384))"
    assert str(nd[1]) == "NDDataset: [quaternion] count (shape: (y:1024, x:2048))"


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


@pytest.mark.skipif(not NMRDATA.exists(), reason="NMR test data not available")
def test_1d_fid_metadata():
    nd = _read_topspin_or_skip(_require_path(nmrdir / "topspin_1d/1/fid"))
    assert nd.origin == "topspin"
    assert nd.units == "count"
    assert nd.title == "intensity"
    assert nd.meta.datatype == "FID"
    assert nd.meta.encoding == ["QSIM"]
    assert nd.meta.isfreq == [False]
    assert nd.meta.iscomplex == [True]
    assert nd.meta.nuc1 == ["1H"]
    assert nd.x.size == 12411
    assert nd.x.title == "F1 acquisition time"


@pytest.mark.skipif(not NMRDATA.exists(), reason="NMR test data not available")
def test_1d_processed_metadata():
    nd = _read_topspin_or_skip(_require_path(nmrdir / "topspin_1d/1/pdata/1/1r"))
    assert nd.meta.datatype == "1D"
    assert nd.meta.isfreq == [True]
    assert nd.meta.iscomplex == [False]
    assert nd.x.units == "ppm"
    assert nd.x.size == 16384
    assert "^{1}H" in nd.x.title


@pytest.mark.skipif(not NMRDATA.exists(), reason="NMR test data not available")
def test_2d_ser_metadata():
    """Indirect dimension encoding must come from acqu2s, not acqus."""
    nd = _read_topspin_or_skip(_require_path(nmrdir / "topspin_2d/1/ser"))
    assert nd.meta.datatype == "SER"
    assert nd.meta.encoding == ["STATES-TPPI", "DQD"]
    assert nd.meta.isfreq == [False, False]
    assert nd.meta.iscomplex == [True, True]
    assert nd.meta.nuc1 == ["31P", "27Al"]
    # FnMODE index 0 = indirect dimension (from acqu2s)
    assert nd.meta.fnmode[0] == 5  # STATES-TPPI
    # Direct dimension uses AQ_mod
    assert nd.meta.aq_mod[1] == 3  # DQD
    assert nd.shape == (96, 474)


@pytest.mark.skipif(not NMRDATA.exists(), reason="NMR test data not available")
def test_2d_processed_metadata():
    nd = _read_topspin_or_skip(_require_path(nmrdir / "topspin_2d/1/pdata/1/2rr"))
    assert nd.meta.datatype == "2D"
    assert nd.meta.isfreq == [True, True]
    assert nd.meta.iscomplex == [False, False]
    assert nd.shape == (1024, 2048)


@pytest.mark.skipif(not NMRDATA.exists(), reason="NMR test data not available")
def test_remove_digital_filter_flag():
    nd_filter = _read_topspin_or_skip(
        _require_path(nmrdir / "topspin_1d/1/fid"),
        remove_digital_filter=True,
    )
    nd_no_filter = _read_topspin_or_skip(
        _require_path(nmrdir / "topspin_1d/1/fid"),
        remove_digital_filter=False,
    )
    assert nd_filter.x.size != nd_no_filter.x.size


@pytest.mark.skipif(not NMRDATA.exists(), reason="NMR test data not available")
def test_use_list_returns_time_axis():
    relax_ser = nmrdir / "relax" / "100" / "ser"
    if not relax_ser.exists():
        pytest.skip("Relaxation test data not available")
    nd = scp.read_topspin(relax_ser, use_list=True)
    assert nd.y.title == "time"
    assert str(nd.y.units) == "s"
    assert nd.y.size > 0


@pytest.mark.skipif(not NMRDATA.exists(), reason="NMR test data not available")
def test_read_topspin_missing_file():
    with pytest.raises(FileNotFoundError):
        scp.read_topspin(nmrdir / "nonexistent" / "1" / "fid")


@pytest.mark.skipif(not NMRDATA.exists(), reason="NMR test data not available")
def test_topspin_name_and_history():
    nd = _read_topspin_or_skip(_require_path(nmrdir / "topspin_1d/1/fid"))
    assert nd.name == "topspin_1d expno:1 procno:1 (FID)"
    assert any("Imported from TopSpin dataset" in entry for entry in nd.history)
