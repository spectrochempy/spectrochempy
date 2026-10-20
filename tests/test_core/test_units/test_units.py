# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from spectrochempy import NDDataset, Quantity, ur


def test_ppm():
    x = 1 * ur.ppm
    assert x.units == ur.ppm


def test_dotted_symbol_parsing():
    # custom display symbols that contain dots must parse back to the unit
    # they are the symbol of. See #913.
    assert ur.Unit("a.u.") == ur.absorbance
    assert ur.Unit("a.u") == ur.absorbance
    assert ur.Unit("K.M.") == ur.Kubelka_Munk
    assert ur.Unit("K.M") == ur.Kubelka_Munk
    assert Quantity("a.u..cm^-1").units == ur.absorbance * ur.cm**-1
    assert Quantity("K.M..cm^-1").units == ur.Kubelka_Munk * ur.cm**-1
    # the full names keep working and separator dots are normalized
    assert ur.Unit("absorbance") == ur.absorbance
    assert ur.Unit("kg.m") == ur.kg * ur.m
    assert ur.Unit("kg . m") == ur.kg * ur.m
    assert ur.Unit("cm.s^-1") == ur.cm / ur.s
    assert ur.Unit("kg.m.s^-2") == ur.kg * ur.m / ur.s**2
    assert ur.Unit("kg.(m/s)") == ur.kg * ur.m / ur.s
    assert Quantity("10 cm.s^-1") == 10 * ur.cm / ur.s
    # decimals and scientific notation are left untouched
    assert Quantity("1.5 m") == 1.5 * ur.m
    assert Quantity("1.e3 m") == 1000 * ur.m


def test_nddataset_dotted_unit_parsing():
    assert NDDataset([1.0], units="cm.s^-1").units == ur.cm / ur.s
    assert NDDataset([1.0], units="a.u..cm^-1").units == ur.absorbance * ur.cm**-1


def test_units():
    assert 10 * ur.km == 10000 * ur.m
    assert ur.km / ur.m == 1000.0
    x = ur.km / ur.m
    assert x.dimensionless


def test_count_is_canonical_and_point_remains_an_alias():
    assert str(ur.Unit("count")) == "count"
    assert str(ur.Unit("point")) == "count"
    assert ur.Unit("count") == ur.Unit("point")


def test_repr_html():
    a = Quantity(10, "s/km")
    assert "{}".format(a) == "10 s⋅km⁻¹"
    assert a._repr_html_() == "10 s⋅km<sup>-1</sup>"


def test_unit_dimensionality():
    a = Quantity(1.0, "cm")
    b = a / Quantity(1.0, "km")
    assert b.dimensionless
    # assert b._repr_html_() == "1.0 scaled-dimensionless (1e-05)"


# def test_matplotlib():
#
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import pint
#
#     ureg = pint.UnitRegistry()
#     ureg.setup_matplotlib(True)
#
#     y = np.linspace(0, 30) * ureg.miles
#     x = np.linspace(0, 5) * ureg.hours
#
#     fig, ax = plt.subplots()
#     ax.yaxis.set_units(ureg.inches)
#     ax.xaxis.set_units(ureg.seconds)
#
#     ax.plot(x, y, 'tab:blue')
#
#     ax.axhline(26400 * ureg.feet, color='tab:red')
#     ax.axvline(120 * ureg.minutes, color='tab:green')
#
#     # here we just test that we can add some label to the default unit labeling
#     ax.set_xlabel('xxx ({})'.format(ax.get_xlabel()))
#     assert ax.get_xlabel() == 'xxx (second)'
#
#     show()
