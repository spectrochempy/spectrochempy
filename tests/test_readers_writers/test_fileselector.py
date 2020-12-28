# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""
This module is for testing fileselector.py

"""

from spectrochempy.widgets.fileselector import FileSelector
from spectrochempy.core import preferences


def test_fileselector():
    path = preferences.datadir
    fs = FileSelector(path=path, filters='spg')

    assert fs.path.endswith('testdata/')
    if fs.value is not None:
        assert fs.value.endswith('')


def test_issues_61():
    path = preferences.datadir + "/"
    fs = FileSelector(path=path, filters='spg')

    assert fs.path.endswith('testdata/')
