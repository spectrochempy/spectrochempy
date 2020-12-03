# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

from pathlib import Path

import spectrochempy as scp


def test_read_topspin():

    nmrdir = Path('nmrdata/bruker/tests/nmr')

    # Open a dialog for selecting a Topspin directory
    A = scp.read_topspin(directory = nmrdir)

    #A.plot()

    A = scp.read_topspin(nmrdir / 'exam2d_HC/3/pdata/1/2rr')
    A.plot_map()


    # Select a TOPSPIN spectra using the full name
    B = scp.read_topspin(nmrdir / 'topspin_1d/1/fid')
    assert str(B) == 'NDDataset: [complex128] unitless (size: 12411)'

    C = scp.read_topspin(nmrdir / 'topspin_1d/1/pdata/1/1r')
    assert str(C) == 'NDDataset: [complex128] unitless (size: 16384)'
    C.plot_map()

    # Select a TOPSPIN spectra using the full name
    B = scp.read_topspin(nmrdir / 'topspin_2d/1/ser')
    assert str(B) == 'NDDataset: [quaternion] unitless (shape: (y:96, x:948))'
    B.plot_surface()

    C = scp.read_topspin(nmrdir / 'topspin_2d/1/pdata/1/2rr')
    assert str(C) == 'NDDataset: [quaternion] unitless (shape: (y:1024, x:2048))'
    C.plot_image()

    scp.show()

