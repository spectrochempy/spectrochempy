# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import struct
from datetime import datetime

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.application.preferences import preferences as prefs
from spectrochempy.core.readers.read_spc import _SpcFile

DATADIR = prefs.datadir


@pytest.fixture
def galacticdata():
    if not (DATADIR / "galacticdata").exists():
        pytest.skip("test data not available (set SCP_TEST_DATA_DOWNLOAD=1)")
    return DATADIR / "galacticdata"


@pytest.mark.data
def test_read_spc_merge_behavior(galacticdata):
    """Test that read_spc respects merge parameter for multi-subfile SPC files."""
    # BARBITUATES.SPC has 286 subfiles with different x-axis lengths
    # Default behavior (merge=False) should return all 286 subfiles individually
    A_default = scp.read_spc("galacticdata/BARBITUATES.SPC")
    assert (
        len(A_default) == 286
    ), "Default merge=False should return 286 individual datasets"
    assert A_default[90].shape == (1, 17)

    # Explicit merge=False should also return 286 datasets
    A_no_merge = scp.read_spc("galacticdata/BARBITUATES.SPC", merge=False)
    assert (
        len(A_no_merge) == 286
    ), "Explicit merge=False should return 286 individual datasets"

    # merge=True groups datasets by shape compatibility
    # Since BARBITUATES.SPC has subfiles with different shapes (incompatible x-axes),
    # they get grouped by shape but can't be merged into multi-row datasets
    A_merged = scp.read_spc("galacticdata/BARBITUATES.SPC", merge=True)
    assert len(A_merged) < 286, "merge=True should group datasets by shape"
    # Verify the merge operation actually occurred (reduced dataset count)
    assert len(A_merged) == 57, "merge=True should reduce to 57 shape groups"

    # Single subfile SPC should return single NDDataset regardless of merge setting
    B_default = scp.read_spc("galacticdata/BENZENE.SPC")
    assert hasattr(B_default, "shape"), "Single file should return NDDataset, not list"
    assert B_default.shape == (1, 1842)

    B_merged = scp.read_spc("galacticdata/BENZENE.SPC", merge=True)
    assert B_merged.shape == (
        1,
        1842,
    ), "Single file with merge=True should still be single dataset"

    B_no_merge = scp.read_spc("galacticdata/BENZENE.SPC", merge=False)
    assert B_no_merge.shape == (
        1,
        1842,
    ), "Single file with merge=False should still be single dataset"


@pytest.mark.data
def test_read_spc(galacticdata):
    A = scp.read_spc("galacticdata/BARBITUATES.SPC")
    assert len(A) == 286
    assert A[90].shape == (1, 17)

    C = scp.read_spc("galacticdata/BENZENE.SPC")
    assert C.shape == (1, 1842)

    D = scp.read_spc("galacticdata/CONTOUR.SPC")
    assert D.shape == (19, 179)

    E = scp.read_spc("galacticdata/DEMO_3D.SPC")
    assert E.shape == (32, 171)

    F = scp.read_spc("galacticdata/DRUG_SAMPLE.SPC")
    assert len(F) == 400
    assert F[0].shape == (1, 10)

    G = scp.read_spc("galacticdata/DRUG_SAMPLE_PEAKS.SPC")
    assert len(G) == 6
    assert G[0].shape == (1, 124)

    H = scp.read_spc("galacticdata/FID.SPC")
    assert H.shape == (1, 8192)

    I = scp.read_spc("galacticdata/HCL.SPC")
    assert I.shape == (1, 8361)

    J = scp.read_spc("galacticdata/HOLMIUM.SPC")
    assert J.shape == (1, 901)

    K = scp.read_spc("galacticdata/IG_BKGND.SPC")
    assert K.shape == (1, 4096)

    L = scp.read_spc("galacticdata/IG_MULTI.SPC")
    assert L.shape == (10, 4096)

    M = scp.read_spc("galacticdata/IG_SAMP.SPC")
    assert M.shape == (1, 4645)

    N = scp.read_spc("galacticdata/KKSAM.SPC")
    assert N.shape == (1, 751)

    # O = scp.read_spc("LC_DIODE_ARRAY.SPC")
    # assert O is None

    P = scp.read_spc("galacticdata/POLYR.SPC")
    assert P.shape == (1, 1844)

    Q = scp.read_spc("galacticdata/POLYS.SPC")
    assert Q.shape == (1, 1844)

    R = scp.read_spc("galacticdata/SINGLE_POLYMER_FILM.SPC")
    assert R.shape == (1, 1844)

    S = scp.read_spc("galacticdata/SPECTRUM_WITH_BAD_BASELINE.SPC")
    # no acquisition time
    if S is None:
        pytest.skip(
            "SPECTRUM_WITH_BAD_BASELINE.SPC is not readable in this test environment"
        )
    assert S.shape == (1, 1400)

    T = scp.read_spc("galacticdata/TOLUENE.SPC")
    assert T.shape == (1, 801)

    U = scp.read_spc("galacticdata/TUMIX.SPC")
    assert U.shape == (1, 1775)

    V = scp.read_spc("galacticdata/TWO_POLYMER_FILMS.SPC")
    assert V.shape == (1, 1844)

    W = scp.read_spc("galacticdata/XYTRACE.SPC")
    assert W.shape == (1, 3469)


@pytest.mark.data
def test_read_spc_single_subfile_sets_acquisition_date_without_changing_timestamp_axis(
    galacticdata,
):
    dataset = scp.read_spc("galacticdata/BENZENE.SPC")

    assert dataset._acquisition_date == datetime(1997, 3, 9, 8, 46, 0)
    assert dataset.y.title == "acquisition timestamp (GMT)"
    assert str(dataset.y.units) == "s"


@pytest.mark.data
def test_read_spc_multi_subfile_sets_acquisition_date_without_changing_existing_support_time(
    galacticdata,
):
    dataset = scp.read_spc("galacticdata/CONTOUR.SPC")

    assert dataset._acquisition_date == datetime(1997, 3, 9, 8, 46, 0)
    assert dataset.y.title == "axis title"
    assert dataset.y.units is None


@pytest.mark.data
def test_read_spc_without_collection_time_keeps_acquisition_date_empty(galacticdata):
    dataset = scp.read_spc("galacticdata/SPECTRUM_WITH_BAD_BASELINE.SPC")

    if dataset is None:
        pytest.skip(
            "SPECTRUM_WITH_BAD_BASELINE.SPC is not readable in this test environment"
        )
    assert dataset.acquisition_date is None
    assert dataset._acquisition_date is None
    assert dataset.y.title == "acquisition timestamp (GMT)"
    assert str(dataset.y.units) == "s"


def _build_spc_header(ftflgs=0x80, npts=0, nsub=0, first=0.0, last=0.0):
    """Build a 512-byte SPC header using the exact struct layout from the reader."""
    head_fmt = "<cccciddicccci9s9sh32s130s30siicchf48sfifc187s"
    return struct.pack(
        head_fmt,
        bytes([ftflgs]),  # ftflgs
        b"\x4b",  # fversn: new LSB 1st
        b"\x00",  # fexper
        b"\x80",  # fexp: float Y (0x80)
        npts,  # fnpts
        float(first),  # ffirst
        float(last),  # flast
        nsub,  # fnsub
        b"\x00",  # fxtype
        b"\x00",  # fytype
        b"\x00",  # fztype
        b"\x00",  # fpost
        0,  # fdate
        b"\x00" * 9,  # fres
        b"\x00" * 9,  # fsource
        0,  # fpeakpt
        b"\x00" * 32,  # fspare
        b"\x00" * 130,  # fcmnt
        b"\x00" * 30,  # fcatxt
        0,  # flogoff
        0,  # fmods
        b"\x00",  # fprocs
        b"\x01",  # flevel
        1,  # fsampin
        1.0,  # ffactor
        b"\x00" * 48,  # fmethod
        0.0,  # fzinc
        0,  # fwplanes
        0.0,  # fwinc
        b"\x00",  # fwtype
        b"\x00" * 187,  # freserv
    )


def _build_subheader(subindx=0, subtime=0.0, subnext=0.0, npts=0):
    """Build a 32-byte SPC subfile header."""
    subhdr_fmt = "<cchfffiif4s"
    return struct.pack(
        subhdr_fmt,
        b"\x00",  # subflgs
        b"\x80",  # subexp: float (0x80)
        subindx,  # subindx
        float(subtime),  # subtime
        float(subnext),  # subnext
        0.0,  # subnois
        npts,  # subnpts
        0,  # subscan
        0.0,  # subwlevel
        b"\x00" * 4,  # subresv
    )


def _make_xy_spc(npts=4, x_values=None, y_values=None):
    """Build a minimal X-Y SPC file (1 subfile, TXVALS flag)."""
    if x_values is None:
        x_values = np.array([10.0, 20.0, 30.0, 40.0], dtype="<f4")
    if y_values is None:
        y_values = np.array([1.1, 2.2, 3.3, 4.4], dtype="<f4")
    npts = len(x_values)
    header = _build_spc_header(ftflgs=0x80, npts=npts)
    subhdr = _build_subheader(subindx=0, subtime=0.0, subnext=1.0, npts=npts)
    return header + x_values.tobytes() + subhdr + y_values.tobytes()


def _make_xmy_spc(npts=3, nsub=3, shared_x=None, y_lists=None):
    """Build a minimal X-MY SPC file (shared X, multiple subfiles, TMULTI+TXVALS)."""
    if shared_x is None:
        shared_x = np.array([100.0, 200.0, 300.0], dtype="<f4")
    if y_lists is None:
        y_lists = [
            np.array([1.0, 2.0, 3.0], dtype="<f4"),
            np.array([4.0, 5.0, 6.0], dtype="<f4"),
            np.array([7.0, 8.0, 9.0], dtype="<f4"),
        ]
    nsub = len(y_lists)
    header = _build_spc_header(ftflgs=0x84, npts=npts, nsub=nsub)
    buf = bytearray(header)
    buf.extend(shared_x.tobytes())
    for i, y in enumerate(y_lists):
        subhdr = _build_subheader(
            subindx=i, subtime=float(i), subnext=float(i + 1), npts=npts
        )
        buf.extend(subhdr)
        buf.extend(y.tobytes())
    return bytes(buf)


def _make_mxy_spc(nsub=3, npts_per_sub=None):
    """Build a directory-based MXY SPC file (per-subfile X, TMULTI+TXVALS+TXYXYS).

    Layout: header(512) + directory(nsub*12) + subfiles.
    Each subfile: subhdr(32) + X(npts*4) + Y(npts*4).
    """
    if npts_per_sub is None:
        npts_per_sub = [4, 3, 5]
    header = _build_spc_header(ftflgs=0xC4, npts=512, nsub=nsub)
    dir_offset = 512
    dir_size = nsub * 12
    sub_start = dir_offset + dir_size
    buf = bytearray(header)
    all_x = []
    all_y = []
    sub_positions = []
    pos = sub_start
    for i, npts in enumerate(npts_per_sub):
        x = np.arange(100 * (i + 1), 100 * (i + 1) + npts, dtype="<f4")
        y = np.ones(npts, dtype="<f4") * (i + 1)
        all_x.append(x)
        all_y.append(y)
        ssfsize = 32 + npts * 4 + npts * 4
        sub_positions.append((pos, ssfsize, float(i)))
        pos += ssfsize
    for ssfposn, ssfsize, ssftime in sub_positions:
        buf.extend(struct.pack("<IIf", ssfposn, ssfsize, ssftime))
    for i, npts in enumerate(npts_per_sub):
        subhdr = _build_subheader(
            subindx=i, subtime=float(i), subnext=float(i + 1), npts=npts
        )
        buf.extend(subhdr)
        buf.extend(all_x[i].tobytes())
        buf.extend(all_y[i].tobytes())
    return bytes(buf), all_x, all_y


def test_extract_x_data_reads_from_supplied_offset():
    spc = object.__new__(_SpcFile)
    spc.head_size = 512
    spc.float32_dtype = "<f4"
    npts = 4
    x_at_offset = np.array([90.0, 91.0, 92.0, 93.0], dtype="<f4")
    offset = 700
    content = bytearray(offset + npts * 4)
    content[offset : offset + npts * 4] = x_at_offset.tobytes()
    x = np.asarray(spc._extract_x_data(offset, bytes(content), npts))
    np.testing.assert_array_equal(x, x_at_offset)


def test_extract_x_data_xy_synthetic():
    content = _make_xy_spc(npts=4)
    x_expected = np.array([10.0, 20.0, 30.0, 40.0], dtype="<f4")
    y_expected = np.array([1.1, 2.2, 3.3, 4.4], dtype="<f4")
    spc = _SpcFile(content)
    assert spc.format == "X-Y"
    assert len(spc.nds) == 1
    x, y, z = spc.nds[0]
    np.testing.assert_array_almost_equal(np.asarray(x), x_expected)
    np.testing.assert_array_almost_equal(np.asarray(y), y_expected)


def test_extract_xmy_synthetic_shared_x():
    shared_x = np.array([100.0, 200.0, 300.0], dtype="<f4")
    content = _make_xmy_spc(npts=3, nsub=3, shared_x=shared_x)
    spc = _SpcFile(content)
    assert spc.format == "X-MY"
    assert len(spc.nds) == 3
    for i, (x, y, z) in enumerate(spc.nds):
        assert x.shape == (3,)
        np.testing.assert_array_almost_equal(np.asarray(x), shared_x)
    x0 = spc.nds[0][0]
    for i in range(1, 3):
        assert spc.nds[i][0] is x0


def test_extract_mxy_synthetic_unique_x():
    npts_per_sub = [4, 3, 5]
    content, all_x, all_y = _make_mxy_spc(nsub=3, npts_per_sub=npts_per_sub)
    spc = _SpcFile(content)
    assert spc.format == "MXY"
    assert len(spc.nds) == 3
    for i, (x, y, z) in enumerate(spc.nds):
        npts = npts_per_sub[i]
        assert x.shape == (npts,)
        np.testing.assert_array_almost_equal(np.asarray(x), all_x[i])
        np.testing.assert_array_almost_equal(np.asarray(y), all_y[i])


def test_read_spc_barbituates_explicit_x_regression(galacticdata):
    fpath = galacticdata / "BARBITUATES.SPC"
    if not fpath.exists():
        pytest.skip("BARBITUATES.SPC not available")
    raw = fpath.read_bytes()
    assert raw[1] == 0x4B
    ftflgs = raw[0]
    assert ftflgs & 0x80, "TXVALS flag must be set"
    assert ftflgs & 0x04, "TMULTI flag must be set"
    assert ftflgs & 0x40, "TXYXYS flag must be set"
    nsub = struct.unpack_from("<I", raw, 24)[0]
    assert nsub == 286
    fnpts_dir_offset = struct.unpack_from("<I", raw, 4)[0]
    assert fnpts_dir_offset > 0, "directory offset must be non-zero"
    hdr_size = 512
    float_dtype = "<f4"
    expected_x_values = []
    dir_offset = fnpts_dir_offset
    for i in range(nsub):
        ssfposn, ssfsize, ssftime = struct.unpack_from("<IIf", raw, dir_offset)
        dir_offset += 12
        npts = int((ssfsize - 32) / 8)
        x = np.frombuffer(raw, dtype=float_dtype, offset=ssfposn + 32, count=npts)
        expected_x_values.append(x)
    spc = _SpcFile(raw)
    assert spc.format == "MXY"
    assert len(spc.nds) == nsub
    for i in range(nsub):
        x, y, z = spc.nds[i]
        assert x.shape[0] == expected_x_values[i].shape[0]
        np.testing.assert_array_almost_equal(np.asarray(x), expected_x_values[i])
        assert np.min(np.asarray(x)) >= 0.0
        assert np.max(np.asarray(x)) <= 300.0
