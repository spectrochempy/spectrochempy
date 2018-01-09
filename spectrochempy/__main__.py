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
    fname = sc.preferences.startup_filename

    if not fname:
        return

    if os.path.exists(fname):
        log.info("Loading filename: '%s'" % fname)
        ds = sc.nddataset.read(fname)
        #ds.print()
    else:
        log.info("'%s' file doesn't exists"%fname)
        app.print_help()

# =============================================================================
if __name__ == '__main__':

    main()
