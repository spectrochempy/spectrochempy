# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

from spectrochempy.gui.preferences import (Preferences,
                                           GeneralPreferencePageWidget,
                                           ProjectPreferencePageWidget,
                                           PlotPreferencePageWidget)

from spectrochempy.extern.pyqtgraph import mkQApp

app = mkQApp()

class testPreferences():

    def __init__(self):

        self.preference_pages = [GeneralPreferencePageWidget,
                                 ProjectPreferencePageWidget,
                                 PlotPreferencePageWidget]

        self.preferences = dlg = Preferences()

        for Page in self.preference_pages:
            page = Page(dlg)
            page.initialize()
            dlg.add_page(page)

        dlg.resize(1000, 400)
        dlg.exec_()

# =============================================================================
if __name__ == '__main__':

    tp = testPreferences()


