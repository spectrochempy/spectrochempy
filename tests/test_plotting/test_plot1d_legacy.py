# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from spectrochempy import NDDataset
from spectrochempy.utils.mplutils import show

# TODO: from spectrochempy.utils.testing import figures_dir, same_images

# https://stackoverflow.com/questions/27948126/how-can-i-write-unit-tests-against-code-that-uses-matplotlib


def test_plot_1D(sample_1d_dataset, sample_2d_dataset):
    nd0 = sample_1d_dataset

    # plot generic 1D
    nd0.plot()
    nd0.plot_scatter(plottitle=True)
    nd0.plot_scatter(marker="^", markevery=10, title="scatter+marker")

    nd0.plot(title="xxxx")
    nd0.plot(marker="o", markevery=10, title="with marker")

    # plot 1D column
    col = sample_2d_dataset[:, 0]
    _ = col.plot_scatter()

    _ = col.plot_scatter(uselabel=True)


def test_issue_375():
    # minimal example
    n_pc = 3

    color1, color2 = "b", "r"

    ratio = NDDataset([1, 2, 3])
    cum = ratio.cumsum()

    ax1 = ratio.plot_bar(color=color1, title="Scree plot")
    assert len(ax1.lines) == 0, "no lines"
    assert len(ax1.patches) == 3, "bar present"
    ax2 = cum.plot_scatter(color=color2, pen=True, markersize=7.0, twinx=ax1)
    assert len(ax2.lines) == 1, "1 lines"
    assert len(ax2.patches) == 0, "no bar present on the second plot"
    # TODO: Don't know yet how to get the marker present.
    ax1.set_title("Scree plot")
    show()

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


def test_plot_multiple_single_axes():
    """Regression test: plot_multiple should produce exactly one axes with multiple lines."""
    import matplotlib.pyplot as plt
    from spectrochempy import NDDataset
    from spectrochempy.plotting.plot1d import plot_multiple

    # Create test datasets
    d1 = NDDataset([1, 2, 3, 4, 5], name="A")
    d2 = NDDataset([2, 3, 4, 5, 6], name="B")
    d3 = NDDataset([3, 4, 5, 6, 7], name="C")

    datasets = [d1, d2, d3]

    # Call plot_multiple
    ax = plot_multiple(
        datasets,
        method="pen",
        legend="best",
        labels=["A", "B", "C"],
        shift=1,
    )

    # Verify: exactly one figure with one axes
    fig = ax.figure
    assert len(fig.get_axes()) == 1, f"Expected 1 axes, got {len(fig.get_axes())}"

    # Verify: three lines (one per dataset)
    assert len(ax.lines) == 3, f"Expected 3 lines, got {len(ax.lines)}"

    plt.close("all")


def test_plot_multiple_scatter_uses_markers():
    """plot_multiple(method='scatter') should not silently become a plain line plot."""
    import matplotlib.pyplot as plt
    from spectrochempy import NDDataset
    from spectrochempy.plotting.plot1d import plot_multiple

    datasets = [
        NDDataset([1, 2, 3], name="A"),
        NDDataset([2, 3, 4], name="B"),
    ]

    ax = plot_multiple(datasets, method="scatter", show=False)

    assert len(ax.lines) == 2
    assert len(ax.collections) == 0
    for line in ax.lines:
        assert line.get_marker() not in (None, "None", "")

    plt.close("all")


def test_plot_multiple_scatter_pen_false_has_no_connecting_lines():
    """pen=False should keep plot_multiple scatter traces as marker-only lines."""
    import matplotlib.pyplot as plt
    from spectrochempy import NDDataset
    from spectrochempy.plotting.plot1d import plot_multiple

    datasets = [
        NDDataset([1, 2, 3], name="A"),
        NDDataset([2, 3, 4], name="B"),
    ]

    ax = plot_multiple(datasets, method="scatter", pen=False, show=False)

    assert len(ax.lines) == 2
    for line in ax.lines:
        assert line.get_marker() not in (None, "None", "")
        assert line.get_linestyle() == "None"

    plt.close("all")


def test_plot_multiple_single_dataset_forwards_method():
    """The single-dataset fallback should keep the requested plotting method."""
    import matplotlib.pyplot as plt
    from spectrochempy import NDDataset
    from spectrochempy.plotting.plot1d import plot_multiple

    dataset = NDDataset([1, 2, 3], name="A")

    ax = plot_multiple(dataset, method="scatter", pen=False, show=False)

    assert len(ax.lines) >= 1
    assert len(ax.collections) == 0
    for line in ax.lines:
        assert line.get_marker() not in (None, "None", "")
        assert line.get_linestyle() == "None"

    plt.close("all")


def test_plot_multiple_show_true_calls_display_helper(mocker):
    """plot_multiple(show=True) should own one final explicit display step."""
    import matplotlib.pyplot as plt
    from spectrochempy import NDDataset
    from spectrochempy.plotting.plot1d import plot_multiple

    display = mocker.patch("spectrochempy.utils.mplutils.show")
    datasets = [
        NDDataset([1, 2, 3], name="A"),
        NDDataset([2, 3, 4], name="B"),
    ]

    _ = plot_multiple(datasets, method="scatter", show=True)

    display.assert_called_once()
    plt.close("all")


def test_plot_multiple_show_false_skips_display_helper(mocker):
    """plot_multiple(show=False) should suppress the explicit display step."""
    import matplotlib.pyplot as plt
    from spectrochempy import NDDataset
    from spectrochempy.plotting.plot1d import plot_multiple

    display = mocker.patch("spectrochempy.utils.mplutils.show")
    datasets = [
        NDDataset([1, 2, 3], name="A"),
        NDDataset([2, 3, 4], name="B"),
    ]

    _ = plot_multiple(datasets, method="scatter", show=False)

    display.assert_not_called()
    plt.close("all")
