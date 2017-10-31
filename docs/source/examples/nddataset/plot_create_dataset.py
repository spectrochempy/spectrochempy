# coding: utf-8
"""
NDDataset creation and plotting example
=======================================

In this example, we create a 3D NDDataset from scratch,
and then we plot one section (a 2D plane)

"""

###############################################################################
# As usual, we start by loading the api
from spectrochempy.api import *

###############################################################################
# Creation
# --------
# Now we will create a 3D NDDataset from scratch
#
# Data
# ++++++
# here we make use of numpy array functions to create the data fot coordinates
# axis and the array of data

c0 = np.linspace(200., 300., 3)
c1 = np.linspace(0., 60., 100)
c2 = np.linspace(4000., 1000., 100)
nd_data = np.array([np.array(
        [np.sin(2. * np.pi * c2 / 4000.) * np.exp(-y / 60) for y in
         c1]) * t for t in c0])

###############################################################################
# Coordinates
# +++++++++++
# The `Coord` object allow making an array of coordinates
# with additional metadata such as units, labels, title, etc

coord0 = Coord(data=c0,
               labels=['cold', 'normal', 'hot'],
               units="K",
               title='temperature')

coord1 = Coord(data=c1,
               labels=None,
               units="minutes",
               title='time-on-stream')

coord2 = Coord(data=c2,
               labels=None,
               units="cm^-1",
               title='wavenumber')

###############################################################################
# Labels can be useful for instance for indexing

a = coord0['normal']
print(a)

####################################################
# nd-Dataset
# +++++++++++
# The `NDDataset` object allow making the array of data with units, etc...

mydataset = NDDataset(nd_data,
                      coordset=[coord0, coord1, coord2],
                      title='Absorbance',
                      units='absorbance'
                      )

mydataset.description = """Dataset example created for this tutorial. 
It's a 3-D dataset (with dimensionless intensity : absorbance )"""

mydataset.name = 'An example from scratch'

mydataset.author = 'Blake and Mortimer'

print(mydataset)

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

figure()
new.plot()

##################################################################
# But it is possible to display image
#

# sphinx_gallery_thumbnail_number = 2
figure()
new.plot(kind='image')

##################################################################
# or stacked plot

figure()
new.plot(kind='stack')

if __name__ == '__main__':
    show()
