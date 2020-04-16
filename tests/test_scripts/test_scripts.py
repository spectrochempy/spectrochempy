# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

from spectrochempy import *

def test_script():

    x = Script('name','print(2)')
    print(x.name)

    try:
        x = Script('0name', 'print(3)')
    except:
        print('name not valid')







# ======================================================================================================================
if __name__ == '__main__':
    pass
