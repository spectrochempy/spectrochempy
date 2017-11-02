# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
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
from spectrochempy.api import ur, set_nmr_context, Quantity, np
from tests.utils import raises

def test_ppm():

    x = 1 * ur.ppm
    assert x.units == ur.ppm

def test_nmr_context():

    set_nmr_context(larmor=104.3 * ur.MHz)

    fhz = 10000 * ur.Hz
    with ur.context('nmr'):
        fppm = fhz.to('ppm')

    assert "{:~.3f}".format(fppm) == '95.877 ppm'
    print("{:.1f}".format(fppm))

    with ur.context('nmr'):
        fhz = fppm.to('Hz')

    assert "{:~.3f}".format(fhz) == '10000.000 Hz'
    print("{:.1f}".format(fhz))

def test_units():

    assert 10 * ur.km == 10000 * ur.m

    assert ur.km / ur.m == 1000.

    x = (ur.km / ur.m)
    assert x.dimensionless

    assert type(x) == type(ur.km)

def test_repr_html():
    a = Quantity(10, 's/km')
    assert "{}".format(a) == "10 second / kilometer"
    assert a._repr_html_() == "10 s.km<sup>-1</sup>"
    #print(a)

def test_unit_dimensionality():
    a = Quantity(1., 'cm')
    b = a/Quantity(1., 'km')
    print(b._repr_html_())

def test_unit_measurement():
    a = Quantity(1., 'cm')
    b = a.plus_minus(.1)
    print(b ** 2)
    print(b._repr_html_())

    xa = Quantity(np.array((1, 2)), 'km')
    with raises(AttributeError):
        ba = xa.plus_minus(.1)

def test_matplotlib():

    import matplotlib.pyplot as plt
    import numpy as np
    from spectrochempy.extern import pint

    ureg = pint.UnitRegistry()
    ureg.setup_matplotlib(True)

    y = np.linspace(0, 30) * ureg.miles
    x = np.linspace(0, 5) * ureg.hours

    fig, ax = plt.subplots()
    ax.yaxis.set_units(ureg.inches)
    ax.xaxis.set_units(ureg.seconds)

    ax.plot(x, y, 'tab:blue')

    ax.axhline(26400 * ureg.feet, color='tab:red')
    ax.axvline(120 * ureg.minutes, color='tab:green')

    # here we just test that we can add some label to the default unit labeling
    ax.set_xlabel('xxx ({})'.format(ax.get_xlabel()))
    assert ax.get_xlabel() == 'xxx (second)'

    plt.show()