# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================


""" Tests for the  module

"""
import sys
import functools
import pytest
from tests.utils import (assert_equal, assert_array_equal,
                         assert_array_almost_equal, assert_equal_units,
                         raises, show_do_not_block)


from spectrochempy.api import *
from spectrochempy.utils import SpectroChemPyWarning


# nmr_processing
#-----------------------------
@show_do_not_block
def test_nmr_1D(NMR_source_1D):

    import sys
    source = NMR_source_1D.copy()

    # perform some analysis
    assert source.is_complex[-1]

    # test if we can plot on the same figure
    source.plot(hold=True, xlim=(0.,25000.))
    # we want to superpose a second spectrum
    source.plot(imag=True, data_only=True)

    # display the real and complex at the same time
    source.plot(show_complex=True, color='green',
                xlim=(0.,3000.), zlim=(-2.,2.))

@show_do_not_block
def test_nmr_1D_em_fft(NMR_source_1D_1H):

    source = NMR_source_1D_1H.copy()

    source.plot(hold=True)
    source = source.em(lb=100.*ur.Hz)
    source.plot(data_only=True)

    source1 = source.fft()
    source1.plot()

    pass


@show_do_not_block
def test_nmr_2D(NMR_source_2D):
    source = NMR_source_2D

    source.plot()

    # perform some anakysis
    assert source.is_complex[-1]

    ax = source.real().plot()
    source.imag().plot()

