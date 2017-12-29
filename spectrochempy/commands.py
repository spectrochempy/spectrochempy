# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

import sys
import os
import spectrochempy as scp, DEBUG

log = spc.log
preferences = spc.preferences

def main():
    preferences.log_level = DEBUG
    fname = spc.preferences.startup_filename
    log.info("Loading filename: '%s'"%fname)
    if os.path.exists(fname):
        ds = spc.nddataset.read(fname)
        print(ds)
    else:
        log.error("'%s' file doesn't exists"%fname)
        print(spc.app.print_help())

# =============================================================================
if __name__ == '__main__':
    main()
