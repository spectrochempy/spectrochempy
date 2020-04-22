# -*- coding: utf-8 -*-
#
# ======================================================================================================================

# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

from spectrochempy import Isotopes
import numpy as np

def test_isotopes():

    isotope = Isotopes('129Xe')

    assert(isotope.name == 'xenon')
    assert(isotope.spin == 1/2)
    assert(isotope.symbol == 'Xe')

    isotope.nucleus = '27Al' # we change the isotope`inplace`
    assert(isotope.name == 'aluminium')
    assert(isotope.spin == 5/2)
    assert(isotope.symbol == 'Al')

    isotope = Isotopes('129Xe')
    print((isotope.name))
    print((isotope.spin))
    print((isotope.symbol))
    isotope.nucleus = '27Al'  # we change the isotope`inplace`
    print((isotope.name))
    print((isotope.spin))
    print((isotope.symbol))
    print((isotope.H_2.Q))
    print((isotope.H_2.gamma.to('MHz/T') / 2. / np.pi))