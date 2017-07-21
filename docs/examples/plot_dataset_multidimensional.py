# coding: utf-8
"""
Multidimensional datasets
=========================

Multidimensional array are defined in Spectrochempy using the NDDataset object.

"""

from spectrochempy.api import *


##############################################################################
# Define a ND-Dataset
#

nd_data = np.random.random((10, 100, 3))

axe0 = Axis(coords = np.linspace(4000., 1000., 10),
            labels = 'a b c d e f g h i j'.split(),
            mask = None,
            units = "cm^-1",
            title = 'wavelength')

axe1 = Axis(coords = np.linspace(0., 60., 100),
            labels = None,
            mask = None,
            units = "s",
            title = 'time-on-stream')

axe2 = Axis(coords = np.linspace(200., 300., 3),
            labels = ['cold', 'normal', 'hot'],
            mask = None,
            units = "K",
            title = 'temperature')

mydataset = NDDataset(nd_data,
               axes = [axe0, axe1, axe2],
               title='Absorbance',
               units='absorbance'
              )
mydataset.description = """Dataset example created for this tutorial. 
It's a 3-D dataset (with dimensionless intensity)"""


##############################################################################
# NDDataset can be sliced like conventional numpy-array...
#


new = mydataset[...,0]
new = new.squeeze() #inplace=True)

new.plot()


#

