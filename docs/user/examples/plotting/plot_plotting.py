# coding: utf-8
"""
Introduction to the plotting librairie
===========================================


"""
import spectrochempy as scp
import os

# this also import the os namespace

#sp.set_loglevel('DEBUG')
datadir = scp.general_preferences.datadir
dataset = scp.NDDataset.read_omnic(
        os.path.join(datadir, 'irdata', 'nh4y-activation.spg'))

########################################################################################################################
# plot generic
ax = dataset[0].plot()

########################################################################################################################
# plot generic style
ax = dataset[0].plot(style='classic')

########################################################################################################################
# check that style reinit to default
# should be identical to the first
ax = dataset[0].plot()

########################################################################################################################
# Multiple plots
dataset = dataset[:,::100]

datasets = [dataset[0], dataset[10], dataset[20], dataset[50], dataset[53]]
labels = ['sample {}'.format(label) for label in
          ["S1", "S10", "S20", "S50", "S53"]]

scp.plot_multiple(method = 'scatter',
                datasets=datasets, labels=labels, legend='best')

########################################################################################################################
# plot mupltiple with style
scp.plot_multiple(method='scatter', style='sans',
              datasets=datasets, labels=labels, legend='best')

########################################################################################################################
# check that style reinit to default
scp.plot_multiple(method='scatter',
              datasets=datasets, labels=labels, legend='best')


#show() # uncomment to show plot if needed()
