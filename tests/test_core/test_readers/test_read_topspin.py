# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from pathlib import Path

import pytest

import spectrochempy as scp
from spectrochempy.utils.plots import show

nmrdir = Path("nmrdata/bruker/tests/nmr")

DATADIR = scp.preferences.datadir
NMRDATA = DATADIR / "nmrdata"


# @pytest.mark.skipif(
#     not NMRDATA.exists(),
#     reason="Experimental data not available for testing",
# )
def test_readtopspin():
    # TODO: revise this to make all dataset 2D even if shape[0]=1. Just for consistency.
    # A.plot()
    nd = scp.read_topspin(nmrdir / "exam2d_HC/3/pdata/1/2rr")
    assert str(nd) == "NDDataset: [quaternion] pp (shape: (y:1024, x:1024))"

    # Select a TOPSPIN spectra using the full name
    nd = scp.read_topspin(nmrdir / "topspin_1d/1/fid")
    assert str(nd) == "NDDataset: [complex128] pp (size: 12411)"

    nd = scp.read_topspin(nmrdir / "topspin_1d/1/pdata/1/1r")
    assert str(nd) == "NDDataset: [complex128] pp (size: 16384)"

    # Select a TOPSPIN spectra using the full name
    nd = scp.read_topspin(nmrdir / "topspin_2d/1/ser")
    assert str(nd) == "NDDataset: [quaternion] pp (shape: (y:96, x:948))"

    nd = scp.read_topspin(nmrdir / "topspin_2d/1/pdata/1/2rr")
    assert str(nd) == "NDDataset: [quaternion] pp (shape: (y:1024, x:2048))"

    # alternative syntax
    nd1 = scp.read_topspin(nmrdir / "topspin_2d", expno=1, procno=1)
    assert nd1 == nd

    # Open a dialog for selecting a Topspin directory
    nd = scp.read_topspin(directory=nmrdir)
    assert nd.name == "topspin_2d expno:1 procno:1 (SER)"

    nd = scp.read_topspin(
        nmrdir, glob="topspin*/*/pdata/*/*"
    )  # TODO: check this more deeply!
    assert isinstance(nd, list)
    assert str(nd[0]) == "NDDataset: [complex128] pp (shape: (y:1, x:16384))"
    assert str(nd[1]) == "NDDataset: [quaternion] pp (shape: (y:1024, x:2048))"


# @pytest.mark.skipif(
#     not NMRDATA.exists(),
#     reason="Experimental data not available for testing",
# )
def test_readdir_for_nmr():
    nd = scp.read_dir("nmrdata/bruker/tests/nmr", protocol="topspin")
    assert isinstance(nd, list)
    nd1 = [item.name for item in nd]
    assert "topspin_2d expno:1 procno:1 (SER)" in nd1
    assert "topspin_1d expno:1 procno:1 (FID)" in nd1


def test_use_list():
    # Importing a pseudo 2D NMR spectra
    # ---------------------------------
    # Define the folder where are the spectra
    datadir = scp.preferences.datadir
    nmrdir = datadir / "nmrdata" / "bruker" / "tests" / "nmr"

    dataset = scp.read_topspin(nmrdir / "relax" / "100" / "ser", use_list=True)
