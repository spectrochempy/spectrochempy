# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

"""This module is the main entry-point for the application launched from the
terminal command line

"""
import sys
import os

def main():
    """Main call
    """

    import spectrochempy as sc
    from spectrochempy.application import app, WARNING, DEBUG

    log = sc.log

    sc.set_loglevel(WARNING)
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
