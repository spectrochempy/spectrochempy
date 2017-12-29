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

def test_lstsq():

    t = sc.NDDataset([0, 1, 2, 3], units='hour')
    d = sc.NDDataset([-1, 0.2, 0.9, 2.1], units='kilometer')

    with pytest.warns(SpectroChemPyWarning):
        assert len(t) == t.ndim # nb of rows = 1

    assert t.size == 4
    assert d.size == 4
    assert t.ndim == 1
    assert d.ndim == 1

    aones = sc.ones(t.shape, units=t.units)
    assert aones.size == 4
    assert aones.ndim == 1
    assert aones.shape == (4,)

    A = sc.stack([t, aones]).T





    v, d0 = sc.lstsq(A, d)[0]

    print(v, d0)

    import matplotlib.pyplot as plt

    plt.plot(t, d, 'o', label='Original data', markersize=10)
    plt.plot(t, v * t + d0, 'r', label='Fitted line')
    plt.legend()
    plt.show()








# =============================================================================
if __name__ == '__main__':

    pass
