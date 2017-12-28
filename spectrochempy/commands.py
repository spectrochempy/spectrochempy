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
from spectrochempy import scp, DEBUG

log = scp.log
preferences = scp.preferences

def main():
    preferences.log_level = DEBUG
    fname = scp.preferences.startup_filename
    log.info("Loading filename: '%s'"%fname)
    if os.path.exists(fname):
        ds = scp.nddataset.read(fname)
        print(ds)
    else:
        log.error("'%s' file doesn't exists"%fname)
        print(scp.app.print_help())

# =============================================================================
if __name__ == '__main__':
    main()
