# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

from pathlib import Path

import spectrochempy as scp


def test_read_labspec():

    datadir = scp.preferences.datadir
    ramandir = Path('ramandata')
    scp.info_(ramandir)

    A = scp.read_labspec('Activation.txt', directory=ramandir)
    A.plot()

    A = scp.read_labspec('532nm-191216-Si 200mu.txt', directory=ramandir)
    A.plot()

    A = scp.read_labspec('serie190214-1.txt', directory=ramandir)
    A.plot(colorbar=True)
    A.plot_map(colorbar=True)

    A = scp.read_labspec('SMC1-Initial RT.txt', directory=ramandir)
    A.plot()

    B = scp.read(protocol='labspec', directory=ramandir)

    # this pack all spectra of the subdir directory
    B = scp.read_dir(directory=ramandir / 'subdir')
    B.plot()

    scp.show()
