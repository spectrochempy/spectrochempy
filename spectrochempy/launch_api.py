# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

"""
This module is the main entry-point for the application launched from the
terminal command line

"""
import sys
import os

def main():

    import spectrochempy as sc
    from spectrochempy.application import app, WARNING, DEBUG

    log = sc.log
    preferences = sc.preferences

    preferences.log_level = WARNING
    fname = app.startup_filename

    if not fname:
        return

    try:
        log.info("Loading filename: '%s'" % fname)
        ds = sc.NDDataset.read(fname)
        ds.plot()
        sc.show()

    except:

        log.info("'%s' file doesn't exists"%fname)
        print()
        app.print_help()

# =============================================================================
if __name__ == '__main__':

    main()
