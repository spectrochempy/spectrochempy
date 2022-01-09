# -*- coding: utf-8 -*-
# flake8: noqa


import pytest
from pathlib import Path

import spectrochempy as scp
from spectrochempy.utils import show

nmrdir = Path("nmrdata/bruker/tests/nmr")


def test_deprecated():
    with pytest.deprecated_call():
        scp.read_bruker_nmr(nmrdir / "topspin_2d", expno=1, remove_digital_filter=True)


def test_read_topspin():

    # A.plot()
    A = scp.read_topspin(nmrdir / "exam2d_HC/3/pdata/1/2rr")
    A.plot_map()

    # Select a TOPSPIN spectra using the full name
    B = scp.read_topspin(nmrdir / "topspin_1d/1/fid")
    assert str(B) == "NDDataset: [complex128] unitless (size: 12411)"

    C = scp.read_topspin(nmrdir / "topspin_1d/1/pdata/1/1r")
    assert str(C) == "NDDataset: [complex128] unitless (size: 16384)"
    C.plot_map()

    # Select a TOPSPIN spectra using the full name
    B = scp.read_topspin(nmrdir / "topspin_2d/1/ser")
    assert str(B) == "NDDataset: [quaternion] unitless (shape: (y:96, x:948))"
    B.plot_surface()

    C = scp.read_topspin(nmrdir / "topspin_2d/1/pdata/1/2rr")
    assert str(C) == "NDDataset: [quaternion] unitless (shape: (y:1024, x:2048))"
    C.plot_image()

    # alternative syntax

    D = scp.read_topspin(nmrdir / "topspin_2d", expno=1, procno=1)
    assert D == C

    # Open a dialog for selecting a Topspin directory
    A = scp.read_topspin(directory=nmrdir)
    assert A.name == "topspin_2d expno:1 procno:1 (SER)"

    show()


def test_read_topspin_glob():

    D = scp.read_topspin(nmrdir, glob="topspin*/*/pdata/*/*")
    print(D)


# EOF
