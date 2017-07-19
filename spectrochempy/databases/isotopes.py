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


"""This module essentially define a class :class:`Isotopes` which handle all
necessary features of NMR nuclei, such as their spin, larmor frequency and so on.

"""

# =============================================================================
# Standard imports
# =============================================================================
import numpy as np
import pandas as pd
from pkg_resources import resource_filename
import re
from fractions import Fraction

from spectrochempy.application import log


# =============================================================================
# Third-party imports
# =============================================================================
from traitlets import (HasTraits,
                        Unicode, Int, Float, Instance)

# =============================================================================
# Local imports
# =============================================================================
from spectrochempy.core.units import ur

# =============================================================================
# Constants
# =============================================================================

# for the command : from .isotopes import *
__all__ = ['Isotopes']
_classes = __all__[:]

# This module's package.
PKG = 'spectrochempy.databases.isotopes'

# =============================================================================
# Isotopes class
# =============================================================================
class Isotopes(HasTraits):
    """
    This class defines useful properties of nuclei

    Parameters
    ----------
    nucleus : String, optional, default='1H'
        In the AX form where A is the atomic mass and X the atom symbol


    Examples
    --------

    How to use it?

    after the api has been loaded:

    >>> from spectrochempy.api import *


    one can use:

    >>> isotope = Isotopes('129Xe')
    >>> isotope.name
    'xenon'
    >>> isotope.spin
    Fraction(1, 2)
    >>> isotope.symbol
    'Xe'
    >>> isotope.nucleus = '27Al' # we change the isotope`inplace`
    >>> isotope.name
    'aluminium'
    >>> isotope.spin
    Fraction(5, 2)
    >>> isotope.symbol
    'Al'


    References
    ----------
    .. [1] Nuclear magnetic moments are taken from Stone, Table of Nuclear
        Magnetic Dipole and Electric Quadrupole Moments, Atomic Data
        and Nuclear Data Tables 90, 75-176 (2005).
        Nuclear quadrupole moments are taken from P.Pyykko, Mol.Phys.
        99, 1617-1629 (2001) and the 2002 edition of the CRC Handbook
        of Physics and Chemistry (which took it from Pyykko and others).

    """

    isotopes = Instance(pd.DataFrame)

    _nucleus = Unicode

    # -------------------------------------------------------------------------
    # nucleus (has an alias isotope, for the getter property only)
    # -------------------------------------------------------------------------
    @property
    def nucleus(self):
        """The current isotope(alias for nucleus)"""
        return self._nucleus

    @nucleus.setter
    def nucleus(self, value):
        self._nucleus = value

    # -------------------------------------------------------------------------
    # isotope / alias of nucleus
    # -------------------------------------------------------------------------

    @property
    def isotope(self):
        """The current isotope (alias for nucleus)"""
        return self.nucleus.strip()

    # -------------------------------------------------------------------------
    # spin
    # -------------------------------------------------------------------------
    @property
    def spin(self):
        "Spin quantum number of the current nucleus"
        return Fraction(self.isotopes.ix[self.nucleus]['spin'])

    # ===========================================================================
    # Z
    # ===========================================================================
    @property
    def Z(self):
        'Atomic number  of the current nucleus'
        return self.isotopes.ix[self.nucleus]['Z']

    # ===========================================================================
    # A
    # ===========================================================================
    @property
    def A(self):
        'Atomic mass  of the current nucleus'
        return self.isotopes.ix[self.nucleus]['A']

    # ===========================================================================
    # full name
    # ===========================================================================
    @property
    def name(self):
        'the name of the nucleus'
        return self.isotopes.ix[self.nucleus]['name'].strip()

    # ===========================================================================
    # gamma
    # ===========================================================================
    @property
    def gamma(self):
        'gyromagnetic ratio of the current nucleus'
        muN = ur.elementary_charge / ur.proton_mass / 2. / (2. * np.pi)
        return (self.isotopes.ix[self.nucleus]['gn']* muN).to('MHz/T')

    # ===========================================================================
    # _get_abundance
    # ===========================================================================
    @property
    def abundance(self):
        'natural abundance in percent of the current nucleus'
        return self.isotopes.ix[self.nucleus]['abundance']

    # ===========================================================================
    # _get_Q
    # ===========================================================================

    @property
    def Q(self):
        """
        Electric quadrupole moment in barn '
                      'of the current nucleus

        """
        try:
            return float(self.isotopes.ix[self.nucleus]['quadrupole'])*1000. * ur.mbarn
        except:
            return 0. * ur.barn

    # -------------------------------------------------------------------------
    # symbol
    # -------------------------------------------------------------------------
    @property
    def symbol(self):
        "Symbol of the current nucleus"
        return self.isotopes.ix[self._nucleus].symbol.strip()

    # -------------------------------------------------------------------------
    # Stability
    # -------------------------------------------------------------------------
    @property
    def stability(self):
        """
        The stability of the current nucleus

        """
        return self.isotopes.ix[self.nucleus].stability.strip()

    # -------------------------------------------------------------------------
    # initializer
    # -------------------------------------------------------------------------
    def __init__(self, nucleus='1H'):
        """
        Class constructor
        """
        filename = resource_filename(PKG, 'isotopes.csv')
        self.isotopes = pd.read_csv(filename, index_col=0)
        self._nucleus = nucleus

    # -------------------------------------------------------------------------
    # Private interface
    # -------------------------------------------------------------------------
    def __str__(self):
        return self.nucleus.strip()

    def __repr__(self):
        return "Isotopes < " + self.nucleus.strip() + " >"

    def _repr_html_(self):
        return "<sup>%s</sup>%s [%s]"%(self.A, self.symbol, self.spin)

    def __getattr__(self, item):
        """
        when an attribute is not found, try to interpret or retrun an informative message
        """
        # it may be a nucleus but in a inverted format
        #try:
        p = re.compile(r'^([A-Z,a-z]+)[_-]*([0-9]+$)')
        m= re.match(p, item).groups()
        nucleus = m[1]+m[0]  # transform "e.g., Al27->27Al, ou AL-27 to 27Al"
        if nucleus in self.isotopes.index.values:
            self.nucleus = nucleus
            return self

        log.warning('The isotope attribute {0} does not exists!'.format(item))
        return None


    # -------------------------------------------------------------------------
    # events
    # -------------------------------------------------------------------------
    def __nucleus_changed(self):
        log.info('Current nucleus has been set to {}'.format(self.nucleus))


if __name__ == '__main__':
    isotope = Isotopes('129Xe')
    print(isotope.name)
    print(isotope.spin)
    print(isotope.symbol)
    isotope.nucleus = '27Al' # we change the isotope`inplace`
    print(isotope.name)
    print(isotope.spin)
    print(isotope.symbol)
    print(isotope.H_2.Q)
    print(isotope.H_2.gamma.to('MHz/T')/2./np.pi)
# =============================================================================
# EOF
# =============================================================================
