# coding: utf-8
"""
Introduction to the plotting librairie
===========================================


"""


import spectrochempy as sp
import os


sp.set_loglevel('DEBUG')

dataset = sp.NDDataset.read_omnic(
        os.path.join(sp.datadir.path, 'irdata', 'nh4y-activation.spg'))


# plot generic
ax = dataset[0].plot()

# plot generic style
ax = dataset[0].plot(style='lcs')

# check that style reinit to default
# should be identical to the first
ax = dataset[0].plot()

dataset = dataset[:,::100]

datasets = [dataset[0], dataset[10], dataset[20], dataset[50], dataset[53]]
labels = ['sample {}'.format(label) for label in
          ["S1", "S10", "S20", "S50", "S53"]]

# plot multiple
sp.plot_multiple(method = 'scatter',
                datasets=datasets, labels=labels, legend='best')

# plot mupltiple with  style
sp.plot_multiple(method='scatter', style='sans',
              datasets=datasets, labels=labels, legend='best')

# check that style reinit to default
sp.plot_multiple(method='scatter',
              datasets=datasets, labels=labels, legend='best')



