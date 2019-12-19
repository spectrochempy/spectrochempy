# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================
"""
pytests file for 

"""

import numpy as np
import pytest
from spectrochempy.core.dataset.npy import *


def test_npy(ds1):

    df = full_like(ds1, dtype=np.complex128, fill_value=2.5)




# ============================================================================
if __name__ == '__main__':
    pass

# end of module