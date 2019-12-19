# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

import pytest

from spectrochempy.utils import SpectroChemPyWarning
import spectrochempy.core as sc
import numpy as np

import matplotlib.pyplot as plt


# ............................................................................
def test_lstsq_from_scratch():

    t = sc.NDDataset(data = [0, 1, 2, 3],
                     title='time',
                     units='hour')

    d = sc.NDDataset(data = [-1, 0.2, 0.9, 2.1],
                     title='distance',
                     units='kilometer')

    # We would like v and d0 such as
    #    d = v.t + d0

    lstsq = sc.LSTSQ(t, d)
    v, d0 = lstsq.transform()

    assert np.around(v.magnitude) == 1
    assert np.around(d0.magnitude,2) == -0.95
    assert v.units == d.units/t.units

    plt.plot(t.data, d.data, 'o', label='Original data', markersize=5)

    dfit = lstsq.inverse_transform()
    plt.plot(t.data, dfit.data, ':r', label='Fitted line')
    plt.legend()
    sc.show()


# ............................................................................
def test_implicit_lstsq():

    t = sc.Coord(data = [0, 1, 2, 3],
                 units='hour',
                 title='time')

    d = sc.NDDataset(data = [-1, 0.2, 0.9, 2.1],
                     coords=[t],
                     units='kilometer',
                     title='distance')

    assert d.ndim == 1

    # We would like v and d0 such as
    #    d = v.t + d0

    lstsq= sc.LSTSQ(d)  #
    v, d0 = lstsq.transform()

    print(v, d0)

    d.plot_scatter(pen=False, markersize=10, mfc='r', mec='k')
    dfit = lstsq.inverse_transform()
    dfit.plot_pen(clear=False, color='g')

    sc.show()

def test_lstq_2D():
    pass

#    St_i = np.linalg.lstsq(C_i, X.data)[0]




# ======================================================================================================================
if __name__ == '__main__':

    pass
