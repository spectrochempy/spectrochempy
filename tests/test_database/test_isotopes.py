# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

from spectrochempy import Isotopes, ur


def test_isotopes():
    isotope = Isotopes('129Xe')

    assert (isotope.name == 'xenon')
    assert (isotope.spin == 1 / 2)
    assert (isotope.symbol == 'Xe')

    isotope.nucleus = '27Al'  # we change the isotope`inplace`
    assert (isotope.name == 'aluminium')
    assert (isotope.spin == 5 / 2)
    assert (isotope.symbol == 'Al')

    assert isinstance(isotope.H_2, Isotopes)
    assert round(isotope.H_2.Q, 2) == 2.86 * ur.millibarn
    assert round(isotope.H_1.gamma * 9.4 * ur.tesla, 0) == 400 * ur.MHz
