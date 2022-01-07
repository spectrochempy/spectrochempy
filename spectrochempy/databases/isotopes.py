# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie,
#  Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in
#  the root directory                         =
# ======================================================================================================================
"""
Database isotopes for NMR.

This module essentially define a class :class:`Isotopes` which handle all
necessary features of NMR nuclei, such as their spin, larmor frequency and
so on.
"""
__all__ = ["Isotopes"]

import pathlib
import re
from fractions import Fraction
import numpy as np

from spectrochempy.units import ur
from spectrochempy.core import preferences as prefs
from spectrochempy.core.dataset.meta import Meta


# ======================================================================================================================
# Isotopes class
# ======================================================================================================================
class Isotopes(Meta):  # lgtm[py/missing-call-to-init]
    """
    This class defines useful properties of nuclei [#]_.

    Parameters
    ----------
    nucleus : String, optional, default='1H'
        In the AX form where A is the atomic mass and X the atom symbol.

    References
    ----------
    .. [#] Nuclear magnetic moments are taken from Stone, Table of Nuclear
           Magnetic Dipole and Electric Quadrupole Moments, Atomic Data
           and Nuclear Data Tables 90, 75-176 (2005).
           Nuclear quadrupole moments are taken from P.Pyykko, Mol.Phys.
           99, 1617-1629 (2001) and the 2002 edition of the CRC Handbook
           of Physics and Chemistry (which took it from Pyykko and others).

    Examples
    --------

    How to use it?

    >>> from spectrochempy import Isotopes
    >>> isotope = Isotopes('129Xe')
    >>> isotope.name
    'xenon'
    >>> isotope.spin
    Fraction(1, 2)
    >>> isotope.symbol
    'Xe'
    >>> isotope.nucleus = '27Al'  # we change the isotope `inplace`.
    >>> isotope.name
    'aluminium'
    >>> isotope.spin
    Fraction(5, 2)
    >>> isotope.symbol
    'Al'
    >>> isotope.nucleus = 'Al27'
    >>> isotope.name
    'aluminium'
    """

    _nucleus = ""

    # ------------------------------------------------------------------------
    # isotope / alias of nucleus
    # ------------------------------------------------------------------------

    @property
    def isotope(self):
        """The current isotope (alias for nucleus)."""
        return self.nucleus.strip()

    # ------------------------------------------------------------------------
    # spin
    # ------------------------------------------------------------------------
    @property
    def spin(self):
        """Spin quantum number of the current nucleus."""
        return Fraction(self[self.nucleus]["spin"])

    # ===========================================================================
    # Z
    # ===========================================================================
    @property
    def Z(self):
        """Atomic number  of the current nucleus."""
        return self[self.nucleus]["Z"]

    # ===========================================================================
    # A
    # ===========================================================================
    @property
    def A(self):
        """Atomic mass  of the current nucleus."""
        return self[self.nucleus]["A"]

    # ===========================================================================
    # full name
    # ===========================================================================
    @property
    def name(self):
        """the name of the nucleus."""
        return self[self.nucleus]["name"].strip()

    # ===========================================================================
    # gamma
    # ===========================================================================
    @property
    def gamma(self):
        """gyromagnetic ratio of the current nucleus."""
        muN = ur.elementary_charge / ur.proton_mass / 2.0 / (2.0 * np.pi)
        return (self[self.nucleus]["gamma"] * muN).to("MHz/T")

    # ===========================================================================
    # _get_abundance
    # ===========================================================================
    @property
    def abundance(self):
        """natural abundance in percent of the current nucleus."""
        return self[self.nucleus]["abundance"]

    # ===========================================================================
    # _get_Q
    # ===========================================================================

    @property
    def Q(self):
        """
        Electric quadrupole moment in barn of the current nucleus.
        """
        try:
            return float(self[self.nucleus]["Q"]) * 1000.0 * ur.mbarn
        except Exception:
            return 0.0 * ur.barn

    # ------------------------------------------------------------------------
    # symbol
    # ------------------------------------------------------------------------
    @property
    def symbol(self):
        """Symbol of the current nucleus."""
        return self[self.nucleus]["symbol"].strip()

    # ------------------------------------------------------------------------
    # Stability
    # ------------------------------------------------------------------------
    @property
    def stability(self):
        """
        The stability of the current nucleus.
        """
        return self[self.nucleus]["stability"].strip()

    @property
    def nucleus(self):
        return self._get_nucleus(self._nucleus)

    def _get_nucleus(self, nuc):

        if nuc in list(self.keys()):
            return nuc

        p = re.compile(r"^([A-Z,a-z]+)[_-]*([0-9]+$)")
        m = re.match(p, nuc).groups()

        nuc = m[1] + m[0]  # transform "e.g., Al27->27Al, ou AL-27 to 27Al"
        if nuc in list(self.keys()):
            return nuc

        else:
            raise KeyError(f"Unknown isotope symbol : {nuc}")

    # ------------------------------------------------------------------------
    # initializer
    # ------------------------------------------------------------------------
    def __init__(self, nucleus="1H"):

        # filename = resource_filename(PKG, 'isotopes.csv')
        DATABASES = pathlib.Path(prefs.databases_directory)
        filename = DATABASES / "isotopes.csv"
        txt = filename.read_text()
        arr = txt.replace(" ", "").split("\n")
        keys = arr[0].split(",")
        dic = {}
        for line in arr[1:]:
            vals = line.split(",")
            dic[vals[0]] = dict(zip(keys[1:], vals[1:]))
        self._data = self._isotopes_validate(dic)
        self.nucleus = nucleus

    # ------------------------------------------------------------------------
    # Private interface
    # ------------------------------------------------------------------------
    def __str__(self):
        return self.nucleus.strip()

    def __repr__(self):
        return "Isotopes < " + self.nucleus.strip() + " >"

    def _repr_html_(self):
        return "<sup>%s</sup>%s [%s]" % (self.A, self.symbol, self.spin)

    def __getattr__(self, key):
        key = self._get_nucleus(key)
        if key != self.nucleus:
            return Isotopes(key)
        else:
            return self[key]

    def __setattr__(self, key, value):
        if key not in [
            "nucleus",
            "readonly",
            "_data",
            "_trait_notifiers",
            "_trait_values",
            "_trait_validators",
            "_cross_validation_lock",
        ]:
            self[key] = value
        elif key == "nucleus":
            self.__dict__["_nucleus"] = value
        else:
            self.__dict__[
                key
            ] = value  # to avoid a recursive call  # we can not use  # self._readonly = value!

    def _isotopes_validate(self, pv):

        for key, item in pv.items():
            if not key or not item:
                continue

            pv[key] = {
                "name": item["name"],
                "symbol": item["symbol"],
                "A": int(item["A"]),
                "Q": float(item["quadrupole"]),
                "Z": int(item["Z"]),
                "gamma": float(item["gn"]),
                "spin": float(item["spin"]),
                "abundance": float(item["abundance"]),
                "stability": item["stability"],
            }

        return pv


#   ======================================================================================================================
if __name__ == "__main__":
    pass
