# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

import pytest

from spectrochempy.utils import SpectroChemPyWarning
import spectrochempy.core as sc

"""
Fit a line, y = mx + c, through some noisy data-points:

>>> x = np.array([0, 1, 2, 3])
>>> y = np.array([-1, 0.2, 0.9, 2.1])
By examining the coefficients, we see that the line should have a gradient of 
roughly 1 and cut the y-axis at, more or less, -1.

We can rewrite the line equation as y = Ap, where A = [[x 1]] 
and p = [[m], [c]]. Now use lstsq to solve for p:

>>> A = np.vstack([x, np.ones(len(x))]).T
>>> A
array([[ 0.,  1.],
       [ 1.,  1.],
       [ 2.,  1.],
       [ 3.,  1.]])
>>> m, c = np.linalg.lstsq(A, y)[0]
>>> print(m, c)
1.0 -0.95
Plot the data along with the fitted line:

>>> import matplotlib.pyplot as plt
>>> plt.plot(x, y, 'o', label='Original data', markersize=10)
>>> plt.plot(x, m*x + c, 'r', label='Fitted line')
>>> plt.legend()
>>> plt.show()


"""

def test_lstsq_from_scratch():

    t = sc.NDDataset(data = [0, 1, 2, 3],
                     title='time',
                     units='hour')

    d = sc.NDDataset(data = [-1, 0.2, 0.9, 2.1],
                     title='distance',
                     units='kilometer')

    # We would like v and d0 such as
    #    d = v.t + d0

    v, d0 = sc.lstsq(t, d)    #

    print(v, d0)

    import matplotlib.pyplot as plt

    plt.plot(t.data, d.data, 'o', label='Original data', markersize=5)
    plt.plot(t.data, (v * t + d0).data, ':r',
             label='Fitted line')
    plt.legend()
    sc.show()

def test_implicit_lstsq():

    t = sc.Coord(data = [0, 1, 2, 3],
                 units='hour',
                 title='time')

    d = sc.NDDataset(data = [-1, 0.2, 0.9, 2.1],
                     coordset=[t],
                     units='kilometer',
                     title='distance')

    assert d.ndim == 1

    # We would like v and d0 such as
    #    d = v.t + d0

    v, d0 = sc.lstsq(d)  #

    print(v, d0)

    d.plot_scatter(pen=False, markersize=10, mfc='r', mec='k')
    dfit = (v*d.x + d0)
    dfit.title = 'distance'
    dfit.coordset = [d.x]
    dfit.plot_pen(hold=True, color='g')

    sc.show()






# =============================================================================
if __name__ == '__main__':

    pass
