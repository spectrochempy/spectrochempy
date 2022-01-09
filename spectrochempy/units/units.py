# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
The core interface to the Pint library.
"""

__all__ = [
    "Unit",
    "Quantity",
    "ur",
    "set_nmr_context",
    "DimensionalityError",
    "remove_args_units",
]

from warnings import warn
from functools import wraps

from pint import (
    set_application_registry,
    UnitRegistry,
    DimensionalityError,
    formatting,
    Context,
)

# from pint.measurement import _Measurement as Measure
from pint.unit import UnitsContainer, _Unit as Unit, UnitDefinition
from pint.quantity import _Quantity as Quantity
from pint.formatting import siunitx_format_unit
from pint.converters import ScaleConverter

# ======================================================================================================================
# Modify the pint behaviour
# ======================================================================================================================
#  TODO: better ways ??

_PRETTY_EXPONENTS = "⁰¹²³⁴⁵⁶⁷⁸⁹"


# ------------------------------------------------------------------
def _pretty_fmt_exponent(num):
    """
    Format a number into a pretty printed exponent using unicode.
    """
    # work badly for decimals as superscript dot do not exist in unicode
    # (as far as we know)
    ret = "{0:n}".format(num).replace("-", "⁻").replace(".", "\u22C5")
    for n in range(10):
        ret = ret.replace(str(n), _PRETTY_EXPONENTS[n])
    return ret


formats = {
    "P": {  # Pretty format.
        "as_ratio": False,  # True in pint
        "single_denominator": False,
        "product_fmt": "·",
        "division_fmt": "/",
        "power_fmt": "{}{}",
        "parentheses_fmt": "({})",
        "exp_call": _pretty_fmt_exponent,
    },
    "L": {  # spectrochempy Latex format.
        "as_ratio": False,  # True in pint
        "single_denominator": True,
        "product_fmt": r" \cdot ",
        "division_fmt": r"\frac[{}][{}]",
        "power_fmt": "{}^[{}]",
        "parentheses_fmt": r"\left({}\right)",
    },
    "H": {  # spectrochempy HTML format.
        "as_ratio": False,  # True in pint
        "single_denominator": False,
        "product_fmt": r".",
        "division_fmt": r"{}/{}",
        "power_fmt": "{}<sup>{}</sup>",
        "parentheses_fmt": r"{}",
    },
    "K": {  # spectrochempy Compact format.
        "as_ratio": False,
        "single_denominator": False,
        "product_fmt": ".",
        "division_fmt": "/",
        "power_fmt": "{}^{}",
        "parentheses_fmt": r"({})",
    },
}

formatting._FORMATS.update(formats)
formatting._KNOWN_TYPES = frozenset(list(formatting._FORMATS.keys()) + ["~"])


def _repr_html_(cls):
    p = cls.__format__("~H")
    # attempt to solve a display problem in notebook (recent version of pint
    # have a strange way to handle HTML. For me it doesn't work
    p = p.replace(r"\[", "").replace(r"\]", "").replace(r"\ ", " ")
    return p


setattr(Quantity, "_repr_html_", _repr_html_)
setattr(Quantity, "_repr_latex_", lambda cls: "$" + cls.__format__("~L") + "$")

# TODO: work on this latex format

setattr(
    Unit,
    "scaling",
    property(lambda u: u._REGISTRY.Quantity(1.0, u._units).to_base_units().magnitude),
)


# ------------------------------------------------------------------
def __format__(self, spec):
    # modify Pint unit __format__

    spec = spec or self.default_format

    # special cases
    if "Lx" in spec:  # the LaTeX siunitx code
        return r"\si[]{%s}" % siunitx_format_unit(self)

    if (
        "~" in spec or "K" in spec or "T" in spec or "L" in spec
    ):  # spectrochempy modified
        if self.dimensionless and "absorbance" not in self._units:
            if self._units == "ppm":
                units = UnitsContainer({"ppm": 1})
            elif self._units in ["percent", "transmittance"]:
                units = UnitsContainer({"%": 1})
            elif self._units == "weight_percent":
                units = UnitsContainer({"wt.%": 1})
            elif self._units == "radian":
                units = UnitsContainer({"rad": 1})
            elif self._units == "degree":
                units = UnitsContainer({"deg": 1})
            # elif self._units == 'absorbance':
            #    units = UnitsContainer({'a.u.': 1})
            elif abs(self.scaling - 1.0) < 1.0e-10:
                units = UnitsContainer({"": 1})
            else:
                units = UnitsContainer(
                    {"scaled-dimensionless (%.2g)" % self.scaling: 1}
                )
        else:
            units = UnitsContainer(
                dict(
                    (self._REGISTRY._get_symbol(key), value)
                    for key, value in list(self._units.items())
                )
            )
        spec = spec.replace("~", "")
    else:
        units = self._units

    if "H" in spec:
        # HTML / Jupyter Notebook (
        return r"\[" + format(units, spec).replace(" ", r"\ ") + r"\]"

    return "%s" % (format(units, spec))


setattr(Unit, "__format__", __format__)

if globals().get("U_", None) is None:

    # filename = resource_filename(PKG, 'spectrochempy.txt')
    U_ = UnitRegistry(on_redefinition="ignore")  # filename)

    U_.define(
        "__wrapped__ = 1"
    )  # <- hack to avoid an error with pytest (doctest activated)
    U_.define("@alias point = count")
    U_.define("transmittance = 1. / 100.")
    U_.define("absolute_transmittance = 1.")
    U_.define("absorbance = 1. = a.u.")
    U_.define("Kubelka_Munk = 1. = K.M.")

    U_.define("ppm = 1. = ppm")

    U_.define(UnitDefinition("percent", "pct", (), ScaleConverter(1 / 100.0)))
    U_.define(UnitDefinition("weight_percent", "wt_pct", (), ScaleConverter(1 / 100.0)))

    U_.default_format = ""  # .2fK'
    Q_ = U_.Quantity
    Q_.default_format = ""  # .2fK'

    set_application_registry(U_)
    del UnitRegistry  # to avoid importing it

else:
    warn("Unit registry was already set up. Bypassed the new loading")

U_.enable_contexts("spectroscopy", "boltzmann", "chemistry")


# Context for NMR
# ------------------------------------------------------------------
def set_nmr_context(larmor):
    """
    Set a NMR context relative to the given Larmor frequency.

    Parameters
    ----------
    larmor : |Quantity| or float
        The Larmor frequency of the current nucleus.
        If it is not a quantity it is assumed to be given in MHz.

    Examples
    --------

    First we set the NMR context,

    >>> from spectrochempy.units import ur, set_nmr_context
    >>>
    >>> set_nmr_context(104.3 * ur.MHz)

    then, we can use the context as follow

    >>> fhz = 10000 * ur.Hz
    >>> with ur.context('nmr'):
    ...     fppm = fhz.to('ppm')
    >>> print("{:~.3f}".format(fppm))
    95.877 ppm

    or in the opposite direction

    >>> with ur.context('nmr'):
    ...     fhz = fppm.to('kHz')
    >>> print("{:~.3f}".format(fhz))
    10.000 kHz

    Now we update the context :

    >>> with ur.context('nmr', larmor=100. * ur.MHz):
    ...     fppm = fhz.to('ppm')
    >>> print("{:~.3f}".format(fppm))
    100.000 ppm

    >>> set_nmr_context(75 * ur.MHz)
    >>> fhz = 10000 * ur.Hz
    >>> with ur.context('nmr'):
    ...     fppm = fhz.to('ppm')
    >>> print("{:~.3f}".format(fppm))
    133.333 ppm
    """

    if not isinstance(larmor, U_.Quantity):
        larmor = larmor * U_.MHz

    if "nmr" not in list(U_._contexts.keys()):
        c = Context("nmr", defaults={"larmor": larmor})

        c.add_transformation(
            "[]",
            "[frequency]",
            lambda U_, x, **kwargs: x * kwargs.get("larmor") / 1.0e6,
        )
        c.add_transformation(
            "[frequency]",
            "[]",
            lambda U_, x, **kwargs: x * 1.0e6 / kwargs.get("larmor"),
        )
        U_.add_context(c)

    else:

        c = U_._contexts["nmr"]
        c.defaults["larmor"] = larmor


# set alias for units and uncertainties
# ------------------------------------------------------------------
ur = U_
Quantity = Q_

# utilities


def remove_args_units(func):
    """
    Decorator which remove units of arguments of a function
    """

    def _remove_units(val):

        if isinstance(val, Quantity):
            val = val.m
        elif isinstance(val, (list, tuple)):
            val = type(val)([_remove_units(v) for v in val])
        return val

    @wraps(func)
    def new_func(*args, **kwargs):

        args = tuple([_remove_units(arg) for arg in args])
        kwargs = {key: _remove_units(val) for key, val in kwargs.items()}
        return func(*args, **kwargs)

    return new_func


# ======================================================================================================================
if __name__ == "__main__":
    pass
