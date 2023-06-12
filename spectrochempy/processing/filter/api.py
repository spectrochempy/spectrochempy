# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
from spectrochempy.utils.packages import generate_api

# in analysis package some classes are configurbles: list them!
__all__, __configurables__ = generate_api(__file__, configurables=True)
pass
