# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL FREE SOFTWARE LICENSE AGREEMENT (Version B) 
# See full LICENSE agreement in the root directory
# =============================================================================




from spectrochempy.api import *

from tests.utils import assert_approx_equal, show_do_not_block
import os
import pytest

@show_do_not_block
def test_nmr():
    path = os.path.join(scpdata, 'nmrdata','bruker', 'tests', 'nmr','bruker_1d')

    # load the data in a new dataset
    ndd = NDDataset()
    ndd.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    ndd._repr_html_()
