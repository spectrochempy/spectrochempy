# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""

"""
import os
import pytest
from spectrochempy.utils import *
from spectrochempy.core import info_

def test_readfilename_wo_filename_provided():

    # should read in the default prefs.datadir
    f = readfilename( filetypes=["OMNIC files (*.sp*)",
                                 "SpectroChemPy files (*.scp)",
                                 "all files (*)"] )
    info_(f)

def test_readfilename_w_directory_instead_of_filename():
    # should read in the specified directory
    f = readfilename(   os.path.expanduser("~/"),
                        filetypes=["OMNIC files (*.sp*)",
                                 "SpectroChemPy files (*.scp)",
                                 "all files (*)"] )
    info_(f)

def test_readfilename_w_bad_filename():
    # should raise an error
    with pytest.raises(IOError):
        f = readfilename(   os.path.expanduser("~/xxxx"),
                            filetypes=["OMNIC files (*.sp*)",
                                 "SpectroChemPy files (*.scp)",
                                 "all files (*)"] )

def test_readfilename_w_good_filename_in_tesdata():
    f = readfilename(   os.path.join('irdata','nh4y-activation.spg'),
                            filetypes=["OMNIC files (*.sp*)",
                                 "SpectroChemPy files (*.scp)",
                                 "all files (*)"] )
    info_(f)

# def test_readfilename(qtbot, monkeypatch):
# TODO: find  a way to test the user interaction with pytest-qt.
#
#     from spectrochempy.gui.widgets.commonwidgets import QFileDialog, QMessageBox
#     from pyqtgraph.Qt import QtGui
#
#     dialog = QFileDialog()
#
#     qtbot.add_widget(dialog)
#     monkeypatch.setattr(QFileDialog, "accept", lambda *args: QFileDialog.Accept)
#
#     f = readfilename()
#     f = readfilename(filetypes='directory')


