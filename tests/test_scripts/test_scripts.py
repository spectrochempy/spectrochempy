# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

from spectrochempy import *

def test_script():

    x = Script('name')
    print(x.name)

    try:
        x = Script('0name')
    except:
        print('name not valid')







# =============================================================================
if __name__ == '__main__':
    pass
