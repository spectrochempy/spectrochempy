# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""This module is the main entry-point for the application launched from the
terminal command line

"""
import sys
import os


def main():
    """
    Main call
    """

    import spectrochempy as scp
    from spectrochempy.core import app, info_

    scp.set_loglevel("INFO")
    fname = app.startup_filename

    if not fname:
        return

    try:
        info_(f"Loading filename : '{fname}'")
        ds = scp.NDDataset.read(fname)
        ds.plot()
        scp.show()

    except:
        print(f"Sorry, but the '{fname}' file couldn't be read.")
        print()
        app.print_help()


# ======================================================================================================================
if __name__ == '__main__':
    main()
