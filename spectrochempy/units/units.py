# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL FREE SOFTWARE LICENSE AGREEMENT (Version 2.1)
# See full LICENSE agreement in the root directory
# =============================================================================




"""The core interface to the Pint library

"""

__all__ = ['Unit', 'Quantity', 'Measurement', 'ur', 'set_nmr_context',
           'DimensionalityError']

from warnings import warn

from spectrochempy.extern.pint import set_application_registry
from spectrochempy.extern.pint import UnitRegistry, DimensionalityError
from spectrochempy.extern.pint.unit import UnitsContainer
from spectrochempy.extern.pint.quantity import _Quantity as Quantity
from spectrochempy.extern.pint.measurement import _Measurement as Measure
from spectrochempy.extern.pint.unit import _Unit as Unit
from spectrochempy.extern.pint import formatting
from spectrochempy.extern.pint.formatting import siunitx_format_unit
from spectrochempy.extern.pint import Context
from spectrochempy.extern.pint.converters import ScaleConverter
from spectrochempy.extern.pint.unit import UnitDefinition
from spectrochempy.extern import pint

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

    'H': {  # spectrochempy HTML format.
        'as_ratio': False,
        'single_denominator': False,
        'product_fmt': r'.',
        'division_fmt': r'{0}/{1}',
        'power_fmt': '{0}<sup>{1}</sup>',
        'parentheses_fmt': r'{0}',
    },

    'L': {  # Latex format.
        'as_ratio': False,
        'single_denominator': True,
        'product_fmt': r' \cdot ',
        'division_fmt': r'\frac[{0}][{1}]',
        'power_fmt': '{0}^[{1}]',
        'parentheses_fmt': r'\left({0}\right)',
    },

}

formatting._FORMATS.update(formats)
formatting._KNOWN_TYPES = frozenset(list(formatting._FORMATS.keys()) + ['~'])

setattr(Quantity, '_repr_html_', lambda cls: cls.__format__('~H'))
setattr(Quantity, '_repr_latex_', lambda cls: "$" + cls.__format__('~L') + "$")

setattr(Measure, '_repr_html_', lambda cls: cls.__format__('~H'))

# TODO: work on this latex format

setattr(Unit, 'scaling', property(lambda u:
                                  u._REGISTRY.Quantity(1.,
                                           u._units).to_base_units().magnitude))

def __format__(self, spec):
    spec = spec or self.default_format

    # special cases
    if 'Lx' in spec:  # the LaTeX siunitx code


        opts = ''
        ustr = siunitx_format_unit(self)
        ret = r'\si[%s]{%s}' % (opts, ustr)
        return ret

    if '~' in spec or 'K' in spec or 'T' in spec or 'L' in spec:  # spectrochempy modified
        if self.dimensionless:

            if self._units == 'ppm':
                units = UnitsContainer({'ppm': 1})
            elif self._units == 'percent':
                units = UnitsContainer({'%': 1})
            elif self._units == 'weight_percent':
                units = UnitsContainer({'wt.%': 1})
            elif self._units == 'absorbance':
                units = UnitsContainer({'a.u.': 1})
            elif abs(self.scaling - 1.) < 1.e-10:
                units = UnitsContainer({'dimensionless': 1})
            else:
                units = UnitsContainer(
                        {'scaled-dimensionless (%.2g)' % self.scaling
                         : 1})
        else:
            units = UnitsContainer(dict((self._REGISTRY._get_symbol(key),
                                         value)
                                        for key, value in list(self._units.items())))
        spec = spec.replace('~', '')
    else:
        units = self._units

    return '%s' % (format(units, spec))


setattr(Unit, '__format__', __format__)

################################################################################

if globals().get('U_', None) is None:
    # filename = resource_filename(PKG, 'spectrochempy.txt')
    U_ = UnitRegistry(on_redefinition='ignore')  # filename)
    set_application_registry(U_)
    U_.enable_contexts('spectroscopy', 'boltzmann', 'chemistry')
    del UnitRegistry  # to avoid importing it

else:
    warn('Unit registry was already set up. Bypassed the new loading')

U_.define(
    '__wrapped__ = 1')  # <- hack to avoid an error with pytest (doctest activated)
U_.define('ppm = 1. = ppm')
U_.define('absorbance = 1. = AU')

U_.define(UnitDefinition('percent', 'pct', (), ScaleConverter(1 / 100.0)))
U_.define(UnitDefinition('weight_percent', 'wt_pct', (), ScaleConverter(1 / 100.0)))

U_.default_format = ''  # .2fK'
Q_ = U_.Quantity
Q_.default_format = ''  # .2fK'
M_ = U_.Measurement
M_.default_format = 'uK'



# Context for NMR
# -----------------

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

    if 'nmr' not in list(U_._contexts.keys()):
        c = Context('nmr', defaults={'larmor': larmor})

        c.add_transformation('[]', '[frequency]',
                             lambda U_, x, **kwargs: x * kwargs.get(
                                 'larmor') / 1.e6)
        c.add_transformation('[frequency]', '[]',
                             lambda U_, x, **kwargs: x * 1.e6 / kwargs.get(
                                 'larmor'))
        U_.add_context(c)

    else:

        c = U_._contexts['nmr']
        c.defaults['larmor'] = larmor


# set alias for units and uncertainties
# --------------------------------------
ur = U_
Quantity = Q_
Measurement = M_

# ===========================================================================
if __name__ == '__main__':
    pass