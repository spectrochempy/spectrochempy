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


"""The core interface to the Pint library

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['Unit', 'Quantity', 'U_', 'Q_', 'M_']

__doctest_skip__ = ['*']

from warnings import warn

from pint import set_application_registry
from pint import UnitRegistry
from pint.unit import UnitsContainer
from pint.quantity import _Quantity as Quantity
from pint.unit import _Unit as Unit
from pint import formatting
from pint.formatting import siunitx_format_unit


PKG = 'spectrochempy.core.units'

# Modify the pint behaviour ####################################################
#  TODO: better ways ??

formats = {

    'K': {  # spectrochempy Compact format.
        'as_ratio': False,
        'single_denominator': False,
        'product_fmt': '.',  # TODO: Should this just be ''?
        'division_fmt': '/',
        'power_fmt': '{0}^{1}',
        'parentheses_fmt': r'({0})',
        },

    'T': {  # spectrochempy HTML format.
        'as_ratio': False,
        'single_denominator': False,
        'product_fmt': r'.',
        'division_fmt': r'{0}/{1}',
        'power_fmt': '{0}<sup>{1}</sup>',
        'parentheses_fmt': r'({0})',
    },

    'X': {  # Latex format.
        'as_ratio': False,
        'single_denominator': False,
        'product_fmt': r' \cdot ',
        'division_fmt': r'\frac[{0}][{1}]',
        'power_fmt': '{0}^[{1}]',
        'parentheses_fmt': r'\left({0}\right)',
    },

}

formatting._FORMATS.update(formats)
formatting._KNOWN_TYPES = frozenset(list(formatting._FORMATS.keys()) + ['~'])

setattr(Quantity, '_repr_html_', lambda cls: cls.__format__('~T'))
setattr(Quantity, '_repr_latex_', lambda cls:"$" + cls.__format__('X') + "$")

#TODO: work on this latex format

setattr(Unit, 'scaling', property(lambda u:
                  u._REGISTRY.Quantity(1., u._units).to_base_units().magnitude))

def __format__(self, spec):

    spec = spec or self.default_format

    # special cases
    if 'Lx' in spec:  # the LaTeX siunitx code
        opts = ''
        ustr = siunitx_format_unit(self)
        ret = r'\si[%s]{%s}' % (opts, ustr)
        return ret

    if '~' in spec or 'K' in spec or 'T' in spec or 'X' in spec: #spectrochempy modified
        if self.dimensionless:

            if self._units == 'ppm':
                units = UnitsContainer({'ppm':1})
            elif abs(self.scaling - 1.) < 1.e-10:
                units = UnitsContainer({'dimensionless':1})
            else:
                units = UnitsContainer({'scaled-dimensionless (%.2g)' % self.scaling
:1})
        else:
            units = UnitsContainer(dict((self._REGISTRY._get_symbol(key),
                                     value)
                                    for key, value in self._units.items()))
        spec = spec.replace('~', '')
    else:
        units = self._units

    return '%s' % (format(units, spec))
setattr(Unit, '__format__', __format__)

################################################################################

if globals().get('U_', None) is None:
    #filename = resource_filename(PKG, 'spectrochempy.txt')
    U_ = UnitRegistry(on_redefinition='ignore') #filename)
    set_application_registry(U_)
    U_.enable_contexts('spectroscopy','boltzmann', 'chemistry')
    del UnitRegistry # to avoid importing it

else:
    warn('Unit registry was already set up. Bypassed the new loading')


U_.define('__wrapped__ = 1')  #<- hack to avoid an error with pytest (doctest activated)
U_.default_format = '' #.2fK'
Q_ = U_.Quantity
Q_.default_format = '' #.2fK'
M_ = U_.Measurement
M_.default_format = 'uK'

