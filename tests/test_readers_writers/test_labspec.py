# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import pytest
from pathlib import Path

import spectrochempy as scp
from spectrochempy.utils.exceptions import ProtocolError

def test_read_labspec():

    datadir = scp.general_preferences.datadir
    ramandir = Path('ramandata')

    A = scp.read_labspec('Activation.txt', directory = ramandir)
    A.plot()

    A = scp.read_labspec('532nm-191216-Si 200µ.txt', directory = ramandir)
    A.plot()

    A = scp.read_labspec('serie190214-1.txt', directory = ramandir)
    A.plot(colorbar=True)
    A.plot_map(colorbar=True)

    A = scp.read_labspec('SMC1-Initial RT.txt', directory = ramandir)
    A.plot()



    B = scp.read(protocol='labspec', directory=ramandir)

    with pytest.raises(ProtocolError):
        B = scp.read(protocol='txt', directory=ramandir)

    B = scp.read(directory=datadir)
    B.plot()

    scp.show()
