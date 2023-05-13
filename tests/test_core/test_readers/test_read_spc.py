# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa

import spectrochempy as scp

DATADIR = scp.preferences.datadir
GALACTICDATA = DATADIR / "galacticdata"


# @pytest.mark.skipif(
#     not GALACTICDATA.exists(),
#     reason="Experimental data not available for testing",
# )
def test_read_spc():
    A = scp.read_spc("galacticdata/BARBITUATES.SPC")
    # "spc reader not implemented yet for multifiles"
    assert A is None

    B = scp.read_spc("galacticdata/barbsvd.spc")
    # multi file, can't be read yet
    assert B is None

    C = scp.read_spc("galacticdata/BENZENE.SPC")
    assert C.shape == (1842,)

    D = scp.read_spc("galacticdata/CONTOUR.SPC")
    # "The version b'M' is not yet supported"
    assert D is None

    E = scp.read_spc("galacticdata/DEMO_3D.SPC")
    # "The version b'M' is not yet supported"
    assert E is None

    F = scp.read_spc("galacticdata/DRUG_SAMPLE.SPC")
    # multi file, can't be read yet
    assert F is None

    G = scp.read_spc("galacticdata/DRUG_SAMPLE_PEAKS.SPC")
    # multi file, can't be read yet
    assert G is None

    H = scp.read_spc("galacticdata/FID.SPC")
    assert H.shape == (8192,)

    I = scp.read_spc("galacticdata/HCL.SPC")
    assert I.shape == (8361,)

    J = scp.read_spc("galacticdata/HOLMIUM.SPC")
    assert J.shape == (901,)

    K = scp.read_spc("galacticdata/IG_BKGND.SPC")
    assert K.shape == (4096,)

    L = scp.read_spc("galacticdata/IG_MULTI.SPC")
    # multi file, can't be read yet
    assert L is None

    M = scp.read_spc("galacticdata/IG_SAMP.SPC")
    assert M.shape == (4645,)

    N = scp.read_spc("galacticdata/KKSAM.SPC")
    assert N.shape == (751,)

    O = scp.read_spc("galacticdata/LC_DIODE_ARRAY.SPC")
    # "The version b'M' is not yet supported"
    assert O is None

    P = scp.read_spc("galacticdata/POLYR.SPC")
    assert P.shape == (1844,)

    Q = scp.read_spc("galacticdata/POLYS.SPC")
    assert Q.shape == (1844,)

    R = scp.read_spc("galacticdata/SINGLE_POLYMER_FILM.SPC")
    assert R.shape == (1844,)

    S = scp.read_spc("galacticdata/SPECTRUM_WITH_BAD_BASELINE.SPC")
    # no acquisition time
    assert S.shape == (1400,)

    T = scp.read_spc("galacticdata/TOLUENE.SPC")
    assert T.shape == (801,)

    U = scp.read_spc("galacticdata/TUMIX.SPC")
    assert U.shape == (1775,)

    V = scp.read_spc("galacticdata/TWO_POLYMER_FILMS.SPC")
    assert V.shape == (1844,)

    W = scp.read_spc("galacticdata/XYTRACE.SPC")
    assert W.shape == (3469,)
