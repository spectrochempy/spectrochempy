# -*- coding: utf-8 -*-
# flake8: noqa

import sys
import pytest
from pathlib import Path

import spectrochempy as scp
from spectrochempy.utils import show

nmrdir = Path("nmrdata/bruker/tests/nmr")

DATADIR = scp.preferences.datadir
NMRDATA = DATADIR / "nmrdata"


@pytest.mark.skipif(
    not NMRDATA.exists(),
    reason="Experimental data not available for testing",
)
def test_deprecated():
    with pytest.deprecated_call():
        scp.read_bruker_nmr(nmrdir / "topspin_2d", expno=1, remove_digital_filter=True)


@pytest.mark.skipif(
    not NMRDATA.exists(),
    reason="Experimental data not available for testing",
)
def test_readtopspin():

    # A.plot()
    nd = scp.read_topspin(nmrdir / "exam2d_HC/3/pdata/1/2rr")
    assert str(nd) == "NDDataset: [quaternion] unitless (shape: (y:1024, x:1024))"

    # Select a TOPSPIN spectra using the full name
    nd = scp.read_topspin(nmrdir / "topspin_1d/1/fid")
    assert str(nd) == "NDDataset: [complex128] unitless (size: 12411)"

    nd = scp.read_topspin(nmrdir / "topspin_1d/1/pdata/1/1r")
    assert str(nd) == "NDDataset: [complex128] unitless (size: 16384)"

    # Select a TOPSPIN spectra using the full name
    nd = scp.read_topspin(nmrdir / "topspin_2d/1/ser")
    assert str(nd) == "NDDataset: [quaternion] unitless (shape: (y:96, x:948))"

    nd = scp.read_topspin(nmrdir / "topspin_2d/1/pdata/1/2rr")
    assert str(nd) == "NDDataset: [quaternion] unitless (shape: (y:1024, x:2048))"

    # alternative syntax
    nd1 = scp.read_topspin(nmrdir / "topspin_2d", expno=1, procno=1)
    assert nd1 == nd

    # Open a dialog for selecting a Topspin directory
    nd = scp.read_topspin(directory=nmrdir)
    assert nd.name == "topspin_2d expno:1 procno:1 (SER)"

    nd = scp.read_topspin(nmrdir, glob="topspin*/*/pdata/*/*")
    assert isinstance(nd, list)
    assert str(nd[0]) == "NDDataset: [complex128] unitless (size: 16384)"
    assert str(nd[1]) == "NDDataset: [quaternion] unitless (shape: (y:1024, x:2048))"


@pytest.mark.skipif(
    not NMRDATA.exists(),
    reason="Experimental data not available for testing",
)
def test_readdir_for_nmr():
    nd = scp.read_dir("nmrdata/bruker/tests/nmr", protocol="topspin")
    assert isinstance(nd, list)
    nd1 = [item.name for item in nd]
    assert "topspin_2d expno:1 procno:1 (SER)" in nd1
    assert "topspin_1d expno:1 procno:1 (FID)" in nd1
