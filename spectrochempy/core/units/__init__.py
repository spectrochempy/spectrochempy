# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
The core interface to the Pint library.


 - % sign (used e.g. for absorbance) leads to an error when plotting with pint 0.24
"""

__all__ = [
    "Unit",
    "Quantity",
    "ur",
    "set_nmr_context",
    "DimensionalityError",
]

from functools import wraps
from warnings import warn

from pint import (
    Context,
    DimensionalityError,
    Quantity,
    Unit,
    UnitRegistry,
    __version__,
    formatting,
    set_application_registry,
)
from pint.facets.plain import ScaleConverter, UnitDefinition
from pint.util import UnitsContainer

# check pint version
pint_version = int(__version__.split(".")[1])

# ======================================================================================
# Modify the pint behaviour
# ======================================================================================
formats = {
    "P": {  # Pretty format.
        "as_ratio": False,  # True in pint
        "single_denominator": False,
        "product_fmt": "·",
        "division_fmt": "/",
        "power_fmt": "{}{}",
        "parentheses_fmt": "({})",
        "exp_call": formatting._pretty_fmt_exponent,
    },
    "L": {  # Latex format.
        "as_ratio": False,  # True in pint
        "single_denominator": True,
        "product_fmt": r" \cdot ",
        "division_fmt": r"\frac[{}][{}]",
        "power_fmt": "{}^[{}]",
        "parentheses_fmt": r"\left({}\right)",
    },
    "H": {  # HTML format.
        "as_ratio": False,  # True in pint
        "single_denominator": False,
        "product_fmt": r" ",
        "division_fmt": r"{}/{}",
        "power_fmt": r"{}<sup>{}</sup>",
        "parentheses_fmt": r"({})",
    },
    "": {  # Default format.
        "as_ratio": True,
        "single_denominator": False,
        "product_fmt": " * ",
        "division_fmt": " / ",
        "power_fmt": "{} ** {}",
        "parentheses_fmt": r"({})",
    },
    "C": {  # Compact format.
        "as_ratio": False,
        "single_denominator": False,
        "product_fmt": "*",  # TODO: Should this just be ''?
        "division_fmt": "/",
        "power_fmt": "{}**{}",
        "parentheses_fmt": r"({})",
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

if pint_version < 24:
    del formatting._FORMATTERS["P"]
    del formatting._FORMATTERS["L"]
    del formatting._FORMATTERS["H"]
    del formatting._FORMATTERS["D"]
    del formatting._FORMATTERS["C"]

    @formatting.register_unit_format("P")
    def format_pretty(unit, registry, **options):
        return formatting.formatter(
            unit.items(),
            as_ratio=False,
            single_denominator=False,
            product_fmt=".",
            division_fmt="/",
            power_fmt="{}{}",
            parentheses_fmt="({})",
            exp_call=formatting._pretty_fmt_exponent,
            **options,
        )

    @formatting.register_unit_format("K")
    def format_spectrochempy_compact(unit, registry, **options):
        return formatting.formatter(
            unit.items(),
            as_ratio=False,
            single_denominator=False,
            product_fmt=".",
            division_fmt="/",
            power_fmt="{}^{}",
            parentheses_fmt=r"({})",
            **options,
        )

    @formatting.register_unit_format("L")
    def format_latex(unit, registry, **options):
        preprocessed = {
            r"\mathrm{{{}}}".format(u.replace("_", r"\_")): p for u, p in unit.items()
        }
        formatted = formatting.formatter(
            preprocessed.items(),
            as_ratio=False,
            single_denominator=True,
            product_fmt=r" \cdot ",
            division_fmt=r"\frac[{}][{}]",
            power_fmt="{}^[{}]",
            parentheses_fmt=r"\left({}\right)",
            **options,
        )
        return formatted.replace("[", "{").replace("]", "}")

    @formatting.register_unit_format("H")
    def format_html(unit, registry, **options):
        return formatting.formatter(
            unit.items(),
            as_ratio=False,
            single_denominator=True,
            product_fmt=r".",
            division_fmt=r"{}/{}",
            power_fmt=r"{}<sup>{}</sup>",
            parentheses_fmt=r"({})",
            **options,
        )

    @formatting.register_unit_format("D")
    def format_default(unit, registry, **options):
        return formatting.formatter(
            unit.items(),
            as_ratio=False,
            single_denominator=False,
            product_fmt="*",
            division_fmt="/",
            power_fmt="{}^{}",
            parentheses_fmt=r"({})",
            **options,
        )

    @formatting.register_unit_format("C")
    def format_compact(unit, registry, **options):
        return formatting.formatter(
            unit.items(),
            as_ratio=False,
            single_denominator=False,
            product_fmt="*",
            division_fmt="/",
            power_fmt="{}**{}",
            parentheses_fmt=r"({})",
            **options,
        )

else:  # pint > 0.24
    # temoporary fix for pint >= 0.24
    # the following should be changed as it fails in giving
    # negative powers, e.g. as_ratio=False should yield "10 s.km⁻¹" while
    # "10 s/km" is obtained...
    # also the "K" format does not seem to be taken into account....

    del formatting.REGISTERED_FORMATTERS["P"]
    del formatting.REGISTERED_FORMATTERS["L"]
    del formatting.REGISTERED_FORMATTERS["H"]
    del formatting.REGISTERED_FORMATTERS["D"]
    del formatting.REGISTERED_FORMATTERS["C"]

    @formatting.register_unit_format("P")
    def format_pretty(unit, registry, **options):
        return formatting.formatter(
            items=unit.items(),
            as_ratio=False,
            single_denominator=False,
            product_fmt=".",
            division_fmt="/",
            power_fmt="{}{}",
            parentheses_fmt="({})",
            exp_call=formatting._pretty_fmt_exponent,
            **options,
        )

    @formatting.register_unit_format("K")
    def format_spectrochempy_compact(unit, registry, **options):
        return formatting.formatter(
            unit.items(),
            as_ratio=False,
            single_denominator=False,
            product_fmt=".",
            division_fmt="/",
            power_fmt="{}^{}",
            parentheses_fmt=r"({})",
            **options,
        )

    @formatting.register_unit_format("L")
    def format_latex(unit, registry, **options):
        preprocessed = {
            r"\mathrm{{{}}}".format(u.replace("_", r"\_")): p for u, p in unit.items()
        }
        formatted = formatting.formatter(
            preprocessed.items(),
            as_ratio=False,
            single_denominator=True,
            product_fmt=r" \cdot ",
            division_fmt=r"\frac[{}][{}]",
            power_fmt="{}^[{}]",
            parentheses_fmt=r"\left({}\right)",
            **options,
        )
        return formatted.replace("[", "{").replace("]", "}")

    @formatting.register_unit_format("H")
    def format_html(unit, registry, **options):
        return formatting.formatter(
            unit.items(),
            as_ratio=False,
            single_denominator=True,
            product_fmt=r".",
            division_fmt=r"{}/{}",
            power_fmt=r"{}<sup>{}</sup>",
            parentheses_fmt=r"({})",
            **options,
        )

    @formatting.register_unit_format("D")
    def format_default(unit, registry, **options):
        return formatting.formatter(
            unit.items(),
            as_ratio=False,
            single_denominator=False,
            product_fmt="*",
            division_fmt="/",
            power_fmt="{}^{}",
            parentheses_fmt=r"({})",
            **options,
        )

    @formatting.register_unit_format("C")
    def format_compact(unit, registry, **options):
        return formatting.formatter(
            unit.items(),
            as_ratio=False,
            single_denominator=False,
            product_fmt="*",
            division_fmt="/",
            power_fmt="{}**{}",
            parentheses_fmt=r"({})",
            **options,
        )


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


# --------------------------------------------------------------------------------------
def __format__(self, spec):
    # modify Pint unit __format__

    spec = formatting.extract_custom_flags(spec or self.default_format)
    if "~" in spec:
        if not self._units:
            return ""

        # Spectrochempy
        if self.dimensionless and "absorbance" not in self._units:
            if self._units == "ppm":
                units = UnitsContainer({"ppm": 1})
            elif self._units in ["percent", "transmittance"]:
                if pint_version >= 24:
                    # to avoid an error when plotting with % sign and pint 0.24
                    units = UnitsContainer({"pct": 1})
                else:
                    units = UnitsContainer({"%": 1})
            elif self._units == "weight_percent":
                if pint_version >= 24:
                    # to avoid an error when plotting with % sign and pint 0.24
                    units = UnitsContainer({"wt.pct": 1})
                else:
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
                    for key, value in self._units.items()
                )
            )
        spec = spec.replace("~", "")
    else:
        units = self._units

    return formatting.format_unit(units, spec, registry=self._REGISTRY)


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

    U_.define(
        UnitDefinition(
            "percent", "pct", (), ScaleConverter(1 / 100.0), UnitsContainer()
        )
    )
    U_.define(
        UnitDefinition(
            "weight_percent",
            "wt_pct",
            (),
            ScaleConverter(1 / 100.0),
            UnitsContainer(),
        )
    )

    if pint_version < 24:
        U_.default_format = "~P"
    else:
        U_.formatter.default_format = "~P"
    Q_ = U_.Quantity
    Q_.default_format = "~P"

    set_application_registry(U_)
    del UnitRegistry  # to avoid importing it

else:
    warn("Unit registry was already set up. Bypassed the new loading")

U_.enable_contexts("spectroscopy", "boltzmann", "chemistry")


# Context for NMR
# --------------------------------------------------------------------------------------
def set_nmr_context(larmor):
    """
    Set a NMR context relative to the given Larmor frequency.

    Parameters
    ----------
    larmor : `Quantity` or `float`
        The Larmor frequency of the current nucleus.
        If it is not a quantity it is assumed to be given in MHz.

    Examples
    --------
    First we set the NMR context,

    >>> from spectrochempy.core.units import ur, set_nmr_context
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
# --------------------------------------------------------------------------------------
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
