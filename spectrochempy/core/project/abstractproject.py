# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
__all__ = []

import traitlets as tr


# ======================================================================================
# AbstractProject class
# ======================================================================================
class AbstractProject(tr.HasTraits):
    """
    Project class will subclass it.

    It is mainly used for avoiding circular imports
    """
