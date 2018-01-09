# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

"""
This module is the main entry-point for the GUI application launched from the
terminal command line

"""
def main_gui():

    from spectrochempy.gui.gui import MainWindow as gui
    gui.start()


# =============================================================================
if __name__ == '__main__':

    main_gui()

