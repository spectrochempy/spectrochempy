# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: sphinx
#       format_version: '1.1'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
Solve a linear equation using LSTSQ
-----------------------------------
In this example, we find the least  square solution of a simple linear
equation.

"""
# sphinx_gallery_thumbnail_number = 2

import os
from spectrochempy import *
from spectrochempy import show


##############################################################################
# Let's take a similar example to the one given in the `numpy.linalg`
# documentation
#
# We have some noisy data that represent the distance `d` traveled by some
# objects versus time `t`:

t = NDDataset(data=[0, 1, 2, 3],
                  title='time',
                  units='hour')

d = NDDataset(data=[-1, 0.2, 0.9, 2.1],
                  coords =[t],
                  title='distance',
                  units='kilometer')

##############################################################################
# Here is a plot of these data-points:

_ = d.plot_scatter(markersize=7, mfc='red')

##############################################################################
# We want to fit a line through these data-points of equation
#
# .. math::
#
#    d = v.t + d_0
#
# By examining the coefficients, we see that the line should have a
# gradient of roughly 1 km/h and cut the y-axis at, more or less, -1 km.
#
# Using LSTSQ, the solution is found very easily:

lst = LSTSQ(t, d)

v, d0 = lst.transform()
print('speed : {:.3fK},  d0 : {:.3fK}'.format(v, d0))


##############################################################################
# Final plot

d.plot_scatter(markersize=10,
               mfc='red', mec='black',
               label='Original data', suptitle='Least-square fitting '
                                                 'example')
dfit = lst.inverse_transform()

dfit.plot_pen(clear=False, color='g', label='Fitted line', legend=True)


##############################################################################
# Note: The same result can be obtained directly using `d` as a single
# parameter on LSTSQ (as `t` is the `x` coordinate axis!)

lst = LSTSQ(d)

v, d0 = lst.transform()
print('speed : {:.3fK},  d0 : {:.3fK}'.format(v, d0))

""
import numpy as np
npoints = 20
slope = 2
offset = 3
x = np.arange(npoints)
y1 = slope * x + offset + np.random.normal(size=npoints)
y2 = np.vstack([y1, 
               2* slope * x + offset + np.random.normal(size=npoints)]).T

""
import matplotlib.pyplot as plt # So we can plot the resulting fit
A = np.vstack([x,np.ones(npoints)]).T
print(A, x, y1)
m, c = np.linalg.lstsq(A, y1)[0] # Don't care about residuals right now
#fig = plt.figure()
#ax  = fig.add_subplot(111)
#plt.plot(x, y, 'bo', label="Data")
#plt.plot(x, m*x+c, 'r--',label="Least Squares")
#plt.show()
M, R, _, _ = np.linalg.lstsq(A, y1)
print(M, R)
print(A.shape, M.shape, y1.shape)
np.sum((np.dot(A,M) - y1)**2, axis=0)

""
import matplotlib.pyplot as plt # So we can plot the resulting fit
A = np.vstack([x,np.ones(npoints)]).T
print(A, x, y2)
m, c = np.linalg.lstsq(A, y2)[0] # Don't care about residuals right now
#fig = plt.figure()
#ax  = fig.add_subplot(111)
#plt.plot(x, y, 'bo', label="Data")
#plt.plot(x, m*x+c, 'r--',label="Least Squares")
#plt.show()
M, R, _, _ = np.linalg.lstsq(A, y2)
print(M, R)
print(A.shape, M.shape, y.shape)
np.sum((np.dot(A,M) - y2)**2, axis=0)

""
S = np.arange(40).reshape(10,4)
S

""
X = np.arange(30).reshape(10,3)
X

""
np.linalg.lstsq(S, X)[0]

""
Sd = NDDataset(S)
Xd = NDDataset(X)
lst = LSTSQ(Sd, Xd)

v, d0 = lst.transform()

""

