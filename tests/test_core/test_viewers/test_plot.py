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
import pytest

from spectrochempy.api import plotoptions


# @pytest.mark.xfail(True, reason='not yet finished')

@pytest.fixture()
def DONOTBLOCK():
    return True  # True # True in principle for testing



def test_plot_generic(IR_source_1, DONOTBLOCK):

    source = IR_source_1.copy()
    plotoptions.do_not_block = DONOTBLOCK
    source.plot()

def test_plot_generic_1D(IR_source_1, DONOTBLOCK):
    source = IR_source_1[0].copy()
    plotoptions.do_not_block = DONOTBLOCK
    source.plot()

def test_plot_2D(IR_source_1, DONOTBLOCK):

    source = IR_source_1.copy()
    plotoptions.do_not_block = DONOTBLOCK
    source.plot_2D()

    source = IR_source_1.copy()
    plotoptions.do_not_block = DONOTBLOCK
    source.plot_2D(kind='map')

    source = IR_source_1.copy()
    plotoptions.do_not_block = DONOTBLOCK
    source.plot_2D(kind='image')

    source = IR_source_1.copy()
    plotoptions.do_not_block = DONOTBLOCK
    source.plot_2D(kind='stack')


def test_plot_map(IR_source_1, DONOTBLOCK):

    source = IR_source_1.copy()
    plotoptions.do_not_block = DONOTBLOCK
    source.plot_map()  # plot_map is an alias of plot_2D


def test_plot_stack(IR_source_1, DONOTBLOCK):

    source = IR_source_1.copy()
    plotoptions.do_not_block = DONOTBLOCK
    source.plot_stack()  # plot_map is an alias of plot_2D

def test_plot_stack_generic(IR_source_1, DONOTBLOCK):
    source = IR_source_1.copy()
    plotoptions.do_not_block = DONOTBLOCK
    source.plot(kind='stack')