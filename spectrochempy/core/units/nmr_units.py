# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
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


"""

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from pint import Context
# =============================================================================
# scaling
# =============================================================================
from spectrochempy.core.units import U_

U_.define('ppm = 1. = ppm')

__all__ = ['set_nmr_context']

def set_nmr_context(larmor):
    """Set a NMR context ur\_ relative to the given Larmor frequency

    Parameters
    ----------
    larmor: Quantity or float
        The Larmor frequency of the current nucleus.
        If it is not a quantity it is assumed to be given i MHz


    Examples
    --------

    First we set the NMR context,

    >>> set_nmr_context(104.3 * U_.MHz)

    then, we can use the context as follow:

    >>> fhz = 10000 * U_.Hz
    >>> with U_.context('nmr'):
    ...    fppm = fhz.to('ppm')
    >>> print("{:~.3f}".format(fppm))
    95.877 ppm

    or in the opposite direction

    >>> with U_.context('nmr'):
    ...    fhz = fppm.to('kHz')
    >>> print("{:~.3f}".format(fhz))
    10.000 kHz

    Now we update the context:

    >>> with U_.context('nmr', larmor=100. * U_.MHz):
    ...    fppm = fhz.to('ppm')
    >>> print("{:~.3f}".format(fppm))
    100.000 ppm

    >>> set_nmr_context(75 * U_.MHz)
    >>> fhz = 10000 * U_.Hz
    >>> with U_.context('nmr'):
    ...    fppm = fhz.to('ppm')
    >>> print("{:~.3f}".format(fppm))
    133.333 ppm
    """

    if not isinstance(larmor, U_.Quantity):
        larmor = larmor * U_.MHz

    if 'nmr' not in U_._contexts.keys():
        c = Context('nmr', defaults={'larmor':larmor})

        c.add_transformation('[]', '[frequency]',
                             lambda U_, x, **kwargs:  x * kwargs.get('larmor') / 1.e6)
        c.add_transformation('[frequency]', '[]',
                             lambda U_, x, **kwargs: x * 1.e6 / kwargs.get('larmor') )
        U_.add_context(c)

    else:

        c = U_._contexts['nmr']
        c.defaults['larmor']=larmor

# if __name__ == '__main__':
#     set_nmr_context(104.3 * U_.MHz)
#     fhz = 10000 * U_.Hz
#     with U_.context('nmr', larmor=100. * U_.MHz):
#         fppm = fhz.to('ppm')
