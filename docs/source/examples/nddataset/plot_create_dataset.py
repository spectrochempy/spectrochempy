# coding: utf-8
"""
Simple example for NDDataset creation and plotting
==================================================

In this example, we create a 3D NDDataset from scratch,
and then we plot one section

"""

######################
# As usual, we start by loading the scp
import numpy as np
import matplotlib.pyplot as plt

import spectrochempy as scp

###############################################
# Now we create a 3D NDDataset from scratch

axe0 = scp.Axis(coords = np.linspace(200., 300., 3),
            labels = ['cold', 'normal', 'hot'],
            mask = None,
            units = "K",
            title = 'temperature')

axe1 = scp.Axis(coords = np.linspace(0., 60., 100),
            labels = None,
            mask = None,
            units = "minutes",
            title = 'time-on-stream')

axe2 = scp.Axis(coords = np.linspace(4000., 1000., 100),
            labels = None,
            mask = None,
            units = "cm^-1",
            title = 'wavenumber')

nd_data=np.array([np.array([np.sin(axe2.data*2.*np.pi/4000.)*np.exp(-y/60.) for y in axe1.data])*float(t)
         for t in axe0.data])**2


mydataset = scp.NDDataset(nd_data,
               axes = [axe0, axe1, axe2],
               title='Absorbance',
               units='absorbance'
              )

mydataset.description = """Dataset example created for this tutorial. 
It's a 3-D dataset (with dimensionless intensity : absorbance )"""

mydataset.name = 'An example from scratch'

mydataset.author = 'Blake and Mortimer'

##################################################################
# We want to plot a section of this 3D NDDataset:
#
# NDDataset can be sliced like conventional numpy-array...

new = mydataset[..., 0]

##################################################################
# or maybe more conveniently in this case, using an axis labels:

new = mydataset['hot']

##################################################################
# The dataset is still 3D (but with a dimension containing a single element...
#
# It can be squeezed easily:

new = new.squeeze()

##################################################################
# To plot a dataset, use the `plot` command (generic plot).
# As the section NDDataset is 2D, a contour plot is displayed by default.

new.plot()

##################################################################
# But it is possible to display image
#
# sphinx_gallery_thumbnail_number = 2

new.plot(kind='image')

##################################################################
# or stacked plot

new.plot(kind='stack')