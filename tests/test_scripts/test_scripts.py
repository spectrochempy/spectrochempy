# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

from spectrochempy import Script, info_


def test_script():
    x = Script('name', 'print(2)')
    info_(x.name)

    try:
        Script('0name', 'print(3)')
    except Exception:
        info_('name not valid')
