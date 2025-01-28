# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from spectrochempy import Quantity, set_nmr_context, ur


def test_ppm():
    x = 1 * ur.ppm
    assert x.units == ur.ppm


def test_nmr_context():
    set_nmr_context(larmor=104.3 * ur.MHz)
    fhz = 10000 * ur.Hz
    with ur.context("nmr"):
        fppm = fhz.to("ppm")
    assert "{:~.3f}".format(fppm) == "95.877 ppm"
    print("{:.1f}".format(fppm))

    with ur.context("nmr"):
        fhz = fppm.to("Hz")
    assert "{:~.3f}".format(fhz) == "10000.000 Hz"
    print("{:.1f}".format(fhz))


def test_units():
    assert 10 * ur.km == 10000 * ur.m
    assert ur.km / ur.m == 1000.0
    x = ur.km / ur.m
    assert x.dimensionless


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
