# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from spectrochempy import NDDataset
from spectrochempy import preferences as prefs
from spectrochempy import show


def test_plot2D():
    A = NDDataset.read_omnic("irdata/nh4y-activation.spg")
    A.y -= A.y[0]
    A.y.to("hour", inplace=True)
    A.y.title = "Acquisition time"
    A.copy().plot_stack()
    A.copy().plot_stack(data_transposed=True)
    A.copy().plot_image(style=["sans", "paper"], fontsize=9)

    # use preferences
    prefs = A.preferences
    prefs.reset()
    prefs.image.cmap = "magma"
    prefs.font.size = 10
    prefs.font.weight = "bold"
    prefs.axes.grid = True
    A.plot()
    A.plot(style=["sans", "paper", "grayscale"], colorbar=False)

    # set ls and lw
    prefs.reset()
    A.plot(lw=0.1)
    A.plot(ls="dashed")

    # use a cyclecolor
    A.plot(cmap=None)

    # use a single color
    A.plot(cmap=None, color="red")

    show()
