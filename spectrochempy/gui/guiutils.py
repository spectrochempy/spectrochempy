# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================



import os

# ----------------------------------------------------------------------------
def geticon(name):

    return os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "ressources", name)