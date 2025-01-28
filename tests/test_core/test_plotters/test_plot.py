# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

# import matplotlib.pyplot as mpl
#
# # from spectrochempy.utils.testing import image_comparison
# from spectrochempy import set_loglevel, INFO, WARNING, MASKED, show, multiplot, plot_multiple
#
# set_loglevel(INFO)
#
# # To regenerate the reference figures, set FORCE to True
# FORCE = True
#
#
# # for this regeneration it is advised to set non parallel testing.
# # (remove option -nauto in pytest.ini)
# #
# # @image_comparison(reference=['IR_dataset_2D_stack', 'IR_dataset_2D_map',
# #                              'IR_dataset_2D_image'], force_creation=FORCE)
# # def test_plot_generic_2D(IR_dataset_2D):
# #     for method in ['stack', 'map', 'image']:
# #         dataset = IR_dataset_2D.copy()
# #         dataset.plot(method=method)
# #
# #
# # @image_comparison(reference=['IR_dataset_1D', 'IR_dataset_1D_sans'],
# #                   force_creation=FORCE)
# # def test_plot_generic_1D(IR_dataset_1D):
# #     dataset = IR_dataset_1D.copy()
# #     dataset.plot()
# #     assert mpl.rcParams['figure.figsize'] == [6.8, 4.4]
# #     dataset.plot(style='sans')
# #     assert mpl.rcParams['font.family'] == ['sans-serif']
# #
# #
# # @image_comparison(reference=['IR_dataset_2D_stack'])
# # def test_plot_stack(IR_dataset_2D):
# #     dataset = IR_dataset_2D.copy()
# #     dataset.plot_stack()  # plot_stack is an alias of plot(method='stack')
# #
# #
# # @image_comparison(reference=['IR_dataset_2D_map'])
# # def test_plot_map(IR_dataset_2D):
# #     dataset = IR_dataset_2D.copy()
# #     dataset.plot_map()  # plot_map is an alias of plot(method='map')
# #
# #
# # @image_comparison(reference=['IR_dataset_2D_image',
# #                              'IR_dataset_2D_image_sanspaper'],
# #                   force_creation=FORCE)
# # def test_plot_image(IR_dataset_2D):
# #     dataset = IR_dataset_2D.copy()
# #     dataset.plot_image()  # plot_image is an alias of plot(method='image')
# #     dataset.plot_image(style=['sans', 'paper'], fontsize=9)
# #
# #
# # @image_comparison(reference=['IR_dataset_2D_image',
# #                              'IR_dataset_2D_image_sanspaper'],
# #                   min_similarity=85.0)
# # def test_plot_image_offset(IR_dataset_2D):
# #     dataset = IR_dataset_2D.copy() + .0001
# #     dataset.plot_image()  # plot_image with offset
# #     dataset.plot_image(style=['sans', 'paper'])
# #
# #
# # @image_comparison(reference=['IR_dataset_2D_stack'])
# # def test_plot_stack_generic(IR_dataset_2D):
# #     dataset = IR_dataset_2D.copy()
# #     dataset.plot()  # generic plot default to stack
#
#
# def test_plot_stack_masked(IR_dataset_2D):
#     # just to see if masked area do not appear on the figure
#     dataset = IR_dataset_2D.copy() * 2.
#     dataset[1300.:900.] = MASKED
#     dataset.plot_stack(colorbar=False)
#     dataset.plot_map(colorbar=False)
#     show()
#
#
# def test_plot_stack_multiple(IR_dataset_2D):
#     dataset = IR_dataset_2D.copy()
#     s1 = dataset[-10:]
#     s2 = dataset[0:5]
#     row = s1[-1]
#     row.plot()
#     # two on the same plot
#     s1.plot_stack()
#     s2.plot_stack(data_only=True, clear=False)
#     show()
#
#
# # BUG FIXES IN PLOTS
#
#
# def test_successive_plot_bug_1a3_28(IR_dataset_2D):
#     dataset = IR_dataset_2D.copy() * 2.
#     dataset[:, 1300.:900.] = MASKED
#     dataset.plot_stack(colorbar=False)
#     dataset.plot()  # in 0.1a3.28 bug because key colorbar is missing.
#     show()
#
#
# def test_successive_plot_bug_with_colorbars(IR_dataset_2D):
#     dataset = IR_dataset_2D.copy() * 2.
#     dataset[1300.:900.] = MASKED
#     dataset.plot_stack()
#     dataset.plot()
#     dataset.plot()  # bug colorbars stacked on the first plot
#     dataset.plot(method='map')  # bug: no colorbar
#     show()
#
#
# @image_comparison(reference=['multiplot1', 'multiplot2'], force_creation=FORCE)
# def test_multiplot(IR_dataset_2D):
#     dataset = IR_dataset_2D.copy()
#
#     datasets = [dataset, dataset * 1.1, dataset * 1.2, dataset * 1.3]
#     labels = ['sample {}'.format(label) for label in
#               ["1", "2", "3", "4"]]
#
#     multiplot(datasets=datasets, method='stack', labels=labels, nrow=2, ncol=2,
#               figsize=(9, 5), sharex=True, sharey=True)
#
#     multiplot(datasets=datasets, method='map', labels=labels, nrow=2, ncol=2,
#               figsize=(9, 5), sharex=True, sharey=True)
#
#
# @image_comparison(reference=['IR_dataset_1D',
#                              'IR_dataset_1D_sans',
#                              'IR_dataset_1D',
#                              'multiple_IR_dataset_1D_scatter',
#                              'multiple_IR_dataset_1D_scatter_sans',
#                              'multiple_IR_dataset_1D_scatter',
#                              ], force_creation=FORCE)
# def tests_multipleplots_and_styles(IR_dataset_1D, IR_dataset_2D):
#     dataset = IR_dataset_1D
#
#     # plot generic
#     dataset.copy().plot()
#
#     # plot generic style
#     dataset.copy().plot(style='sans')
#
#     # check that style reinit to default
#     dataset.copy().plot()
#
#     dataset = IR_dataset_2D
#
#     datasets = [dataset[0], dataset[10], dataset[20], dataset[50], dataset[53]]
#     labels = ['sample {}'.format(label) for label in
#               ["S1", "S10", "S20", "S50", "S53"]]
#
#     # plot multiple
#     plot_multiple(method='scatter',
#                   datasets=datasets, labels=labels, legend='best')
#
#     # plot mupltiple with  style
#     plot_multiple(method='scatter', style='sans',
#                   datasets=datasets, labels=labels, legend='best')
#
#     # check that style reinit to default
#     plot_multiple(method='scatter',
#                   datasets=datasets, labels=labels, legend='best')
#
#
# # #### debugging ####
# set_loglevel(WARNING)
