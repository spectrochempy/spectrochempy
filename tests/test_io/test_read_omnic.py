# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import os
from spectrochempy import NDDataset, info_


def test_read_without_filename():
    A = NDDataset.read_omnic()
    info_(A)


def test_read_with_filename():
    A = NDDataset.read_omnic(os.path.join('irdata', 'nh4y-activation.spg'))
    info_(A)
