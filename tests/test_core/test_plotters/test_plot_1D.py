# -*- coding: utf-8 -*-
# flake8: noqa

import os

from spectrochempy import NDDataset
from spectrochempy.utils import show

# TODO: from spectrochempy.utils.testing import figures_dir, same_images

# https://stackoverflow.com/questions/27948126/how-can-i-write-unit-tests-against-code-that-uses-matplotlib


# @pytest.mark.skip
def test_plot_1D():

    dataset = NDDataset.read_omnic(os.path.join("irdata", "nh4y-activation.spg"))

    # get first 1D spectrum
    nd0 = dataset[0, 1550.0:1600.0]

    # plot generic 1D
    nd0.plot()
    nd0.plot_scatter(plottitle=True)
    nd0.plot_scatter(marker="^", markevery=10, title="scatter+marker")
    prefs = nd0.preferences
    prefs.method_1D = "scatter+pen"

    nd0.plot(title="xxxx")
    prefs.method_1D = "pen"
    nd0.plot(marker="o", markevery=10, title="with marker")

    # plot 1D column
    col = dataset[:, 3500.0]  # note the indexing using wavenumber!
    _ = col.plot_scatter()

    _ = col.plot_scatter(uselabel=True)

    # nd0.plot(output=os.path.join(figures_dir, 'IR_dataset_1D'),
    #          savedpi=150)
    #
    # # plot generic style
    # nd0.plot(style='poster',
    #          output=os.path.join(figures_dir, 'IR_dataset_1D_poster'),
    #          savedpi=150)
    #
    # # check that style reinit to default
    # nd0.plot(output='IR_dataset_1D', savedpi=150)
    # # try:
    # #     assert same_images('IR_dataset_1D.png',
    # #                        os.path.join(figures_dir, 'IR_dataset_1D.png'))
    # # except AssertionError:
    # #     os.remove('IR_dataset_1D.png')
    # #     raise AssertionError('comparison fails')
    # # os.remove('IR_dataset_1D.png')
    #
    # # try other type of plots
    # nd0.plot_pen()
    # nd0[:, ::100].plot_scatter()
    # nd0.plot_lines()
    # nd0[:, ::100].plot_bar()
    #
    # show()
    #
    # # multiple
    # d = dataset[:, ::100]
    # datasets = [d[0], d[10], d[20], d[50], d[53]]
    # labels = ['sample {}'.format(label) for label in
    #           ["S1", "S10", "S20", "S50", "S53"]]
    #
    # # plot multiple
    # plot_multiple(method='scatter',
    #               datasets=datasets, labels=labels, legend='best',
    #               output=os.path.join(figures_dir,
    #                                   'multiple_IR_dataset_1D_scatter'),
    #               savedpi=150)
    #
    # # plot mupltiple with style
    # plot_multiple(method='scatter', style='sans',
    #               datasets=datasets, labels=labels, legend='best',
    #               output=os.path.join(figures_dir,
    #                                   'multiple_IR_dataset_1D_scatter_sans'),
    #               savedpi=150)
    #
    # # check that style reinit to default
    # plot_multiple(method='scatter',
    #               datasets=datasets, labels=labels, legend='best',
    #               output='multiple_IR_dataset_1D_scatter',
    #               savedpi=150)
    # try:
    #     assert same_images('multiple_IR_dataset_1D_scatter',
    #                        os.path.join(figures_dir,
    #                                     'multiple_IR_dataset_1D_scatter'))
    # except AssertionError:
    #     os.remove('multiple_IR_dataset_1D_scatter.png')
    #     raise AssertionError('comparison fails')
    # os.remove('multiple_IR_dataset_1D_scatter.png')

    show()


# ======================================================================================================================
if __name__ == "__main__":
    pass
