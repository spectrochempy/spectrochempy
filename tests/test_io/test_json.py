# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import os

from spectrochempy import NDDataset, general_preferences as prefs
from spectrochempy.utils.testing import assert_approx_equal

# datasets are defined in conftest as fixture

def test_load_json(IR_dataset_2D):
    ds = IR_dataset_2D
    ds.write('try2D.json')
    dsr = NDDataset.read('try2D.json')
    assert ds == dsr
    os.remove('try2D.json')
