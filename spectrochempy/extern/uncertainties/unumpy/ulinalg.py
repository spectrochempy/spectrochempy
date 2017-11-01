"""
This module provides uncertainty-aware functions that generalize some
of the functions from numpy.linalg.

(c) 2010-2016 by Eric O. LEBIGOT (EOL) <eric.lebigot@normalesup.org>.
"""

from spectrochempy.extern.uncertainties import __author__
from spectrochempy.extern.uncertainties.unumpy.core import inv, pinv

# This module cannot import unumpy because unumpy imports this module.

__all__ = ['inv', 'pinv']

