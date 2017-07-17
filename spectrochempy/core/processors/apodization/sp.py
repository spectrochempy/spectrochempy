# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2016 Christian Fernandez
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================

# =============================================================================
# Third party imports
# =============================================================================
import numpy as np

# =============================================================================
# Local imports
# =============================================================================
from .apodize import apodize
from spectrochempy.core.units import Quantity

# =============================================================================
# interface for the processing class
# =============================================================================
__all__ = ["sp"]


# =============================================================================
# sp function
# =============================================================================

def sp(self, *args, **kwargs):
    """Calculate a Shifted sine-bell apodization

    Functional form of apodization window:

    .. math::
        sp(x) = \\sin(\\frac{pi * off + pi * (end - off) * x} {size - 1})^{pow}

    Parameters
    ----------
    off : float

        offset - Specifies the starting point of the sine-bell in time units
        The default value is 0.0.

    end : float

        end - Specifies the ending point of the sine-bell in time units.

    pow : float

        pow - Specifies the exponent of the sine-bell; Non-integer values
        are allowed. Common values are 1.0 (for ordinary sine-bell) and 2.0
        (for squared-bell functions). The default value is 1.0.


    inv : bool, optional

        True for inverse apodization.  False (default) for standard.

    rev : bool, optional.

        True to reverse the apodization before applying it to the data.

    """
    args = list(args)  # important (args is a tuple)

    # off
    off = kwargs.get('off', 0)
    if off == 0:
        # let's try the args if the kwargs was not passed
        if len(args) > 0:
            off = args.pop(0)

    # end
    end = kwargs.get('end', 1.)
    if end == 1.:
        # let's try the second args if the kwargs was not passed
        if len(args) > 0:
            end = args.pop(0)

    # pow
    pow = kwargs.pop('pow', 1.)
    if pow == 1.:
        # let's try the args if the kwargs was not passed
        if len(args) > 0:
            pow = args.pop(0)

    # apod func (must be func(x, tc1, tc2, tc3) form
    def func(x, off, end, pow):
        w = x[-1].data - x[0].data
        i = np.arange(0, x.size, 1)
        if isinstance(pow, Quantity):
            pow = pow.magnitude
            off = off.magnitude
            end = end.magnitude
        return np.sin(
            np.pi * off / w + np.pi * ((end - off)/w) * i / (x.size - 1.)) ** pow

    kwargs['method'] = func
    kwargs['apod'] = off
    kwargs['apod2'] = end
    kwargs['apod3'] = pow

    return apodize(self, **kwargs)
