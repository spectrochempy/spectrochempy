# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

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
    assert S.shape == (1, 1400)

    T = scp.read_spc("galacticdata/TOLUENE.SPC")
    assert T.shape == (1, 801)

    U = scp.read_spc("galacticdata/TUMIX.SPC")
    assert U.shape == (1, 1775)

    V = scp.read_spc("galacticdata/TWO_POLYMER_FILMS.SPC")
    assert V.shape == (1, 1844)

    W = scp.read_spc("galacticdata/XYTRACE.SPC")
    assert W.shape == (1, 3469)


def test_extract_x_data_reads_from_head_size_not_offset():
    # AUDIT (#1151): pin the CURRENT behavior of ``_SpcFile._extract_x_data``.
    #
    # The method reads the explicit X array from the fixed header boundary
    # (``offset=self.head_size``) and discards the ``np.frombuffer`` read at the
    # supplied ``offset`` argument, so the returned X never depends on
    # ``offset``.  This synthetic, data-free test documents that behavior so any
    # future correction is a deliberate, reviewed change.
    #
    # Impact, verified against the Galactic SPC layout and the two call sites in
    # ``_get_sub_file``:
    #   * X-Y / X-MY files: the single explicit X block sits immediately after
    #     the header, so the call site passes ``offset == head_size`` and the
    #     returned X is correct (the discarded read would have produced the same
    #     bytes).
    #   * MXY (TXVALS + TXYXYS) files: each subfile owns a separate X array at a
    #     varying offset (after its 32-byte subheader), so reading from the fixed
    #     ``head_size`` returns the first block's bytes for every subfile -- a
    #     latent coordinate-extraction error that leaves the dataset shape intact
    #     while corrupting the X coordinates.
    spc = object.__new__(_SpcFile)
    spc.head_size = 512
    spc.float32_dtype = "<f4"

    npts = 4
    x_at_head_size = np.array([10.0, 11.0, 12.0, 13.0], dtype="<f4")
    x_at_subfile_offset = np.array([90.0, 91.0, 92.0, 93.0], dtype="<f4")

    # an offset like a later subfile's X array (after head + subheader + X/Y)
    offset = spc.head_size + 32 + npts * 4
    content = bytearray(offset + npts * 4)
    content[spc.head_size : spc.head_size + npts * 4] = x_at_head_size.tobytes()
    content[offset : offset + npts * 4] = x_at_subfile_offset.tobytes()
    content = bytes(content)

    x = np.asarray(spc._extract_x_data(offset, content, npts))

    # current behavior: X comes from head_size, the ``offset`` argument is unused
    np.testing.assert_array_equal(x, x_at_head_size)
    assert not np.array_equal(x, x_at_subfile_offset)
