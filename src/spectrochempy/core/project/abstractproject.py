# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import traitlets as tr


# ======================================================================================
# AbstractProject class
# ======================================================================================
class AbstractProject(tr.HasTraits):
    """
    Project class will subclass it.

    It is mainly used for avoiding circular imports
    """
