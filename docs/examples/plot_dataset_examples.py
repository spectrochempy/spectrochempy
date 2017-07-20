# coding: utf-8
"""
Dataset creation and plotting
==============================

essai

"""

import sys
from spectrochempy.api import *

def main():

    # Create a ND-Dataset from scratch

    axe0 = Axis(coords = np.linspace(200., 300., 3),
                labels = ['cold', 'normal', 'hot'],
                mask = None,
                units = "K",
                title = 'temperature')

    axe1 = Axis(coords = np.linspace(0., 60., 100),
                labels = None,
                mask = None,
                units = "minutes",
                title = 'time-on-stream')

    axe2 = Axis(coords = np.linspace(4000., 1000., 10),
                labels = None,
                mask = None,
                units = "cm^-1",
                title = 'wavelength')

    nd_data=np.array([np.array([np.sin(axe2.data*2.*np.pi/4000.)*np.exp(-y/60.) for y in axe1.data])*float(t)
             for t in axe0.data])**2


    # The dataset is now create with these data and defined axis:

    mydataset = NDDataset(nd_data,
                   axes = [axe0, axe1, axe2],
                   title='Absorbance',
                   units='absorbance'
                  )

    mydataset.description = """Dataset example created for this tutorial. 
    It's a 3-D dataset (with dimensionless intensity)"""

    mydataset.author = 'Tintin and Milou'

    # NDDataset can be sliced like conventional numpy-array...

    new = mydataset[..., 0]

    # or using the axes labels:

    new = mydataset['hot']


    # Single-element dimension are kept but can also be squeezed easily:

    new = new.squeeze()


    # To plot a dataset, use the `plot` command (generic plot).
    #  As the NDDataset is 2D, a contour plot is displayed by default.

    new.plot()

if __name__ == '__main__':
    main()
