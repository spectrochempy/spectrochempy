# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

# flake8: noqa

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import show
from spectrochempy.utils.testing import assert_array_equal


# TODO: from spectrochempy.utils.testing import figures_dir, same_images

# https://stackoverflow.com/questions/27948126/how-can-i-write-unit-tests-against
# -code-that-uses-matplotlib


# @pytest.mark.skip
def test_plot_1D(IR_dataset_2D):
    # get first 1D spectrum
    nd0 = IR_dataset_2D[0, 1550.0:2000.0]
    # plot generic 1D
    ax = nd0.plot()
    assert_array_equal(ax.lines[0].get_ydata(), nd0.data.squeeze())
    nd0.plot_scatter(plottitle=True)
    nd0.plot_scatter(marker="^", markevery=10, title="scatter+marker")
    prefs = nd0.preferences
    prefs.method_1D = "scatter+pen"
    nd0.plot(title="xxxx")
    prefs.method_1D = "pen"
    nd0.plot(marker="o", markevery=10, title="with marker")
    show()


def test_issue_375():
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


def test_nmr_1D_show_complex(NMR_dataset_1D):
    dataset = NMR_dataset_1D.copy()
    dataset.plot(xlim=(0.0, 25000.0))
    dataset.plot(imag=True, color="r", data_only=True, clear=False)
    # display the real and complex at the same time
    dataset.plot(
        show_complex=True, color="green", xlim=(0.0, 30000.0), zlim=(-200.0, 200.0)
    )
    show()


def test_plot_time_axis(IR_dataset_2D):
    # plot 1D
    col = IR_dataset_2D[:, 3500.0]  # note the indexing using wavenumber!
    _ = col.plot_scatter()
    # _ = col.plot_scatter(uselabel=True)
    show()


# ======================================================================================================================
if __name__ == "__main__":
    pass
