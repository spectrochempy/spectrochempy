# -*- coding: utf-8 -*-
# flake8: noqa
import pytest

from spectrochempy import NDDataset, show, preferences as prefs


def test_plot2D():
    A = NDDataset.read_omnic("irdata/nh4y-activation.spg")

    ax = A.copy().plot_stack()
    show()
    assert len(ax.lines) == A.shape[0]

    ax = A.copy().plot_stack(transposed=True)
    show()
    assert len(ax.lines) == 1110  # because we display only a subset

    ax = A.copy().plot_image(style=["sans", "paper"], fontsize=9)
    show()
    assert len(ax.lines) == 0

    A.y -= A.y[0]
    A.y.to("hour", inplace=True)
    A.y.title = u"Acquisition time"
    A.copy().plot_stack()
    A.copy().plot_stack(transposed=True)
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

    show()


@pytest.mark.skip("not yet ready")
def test_plotly2D():
    A = NDDataset.read_omnic("irdata/nh4y-activation.spg", directory=prefs.datadir)
    A.y -= A.y[0]
    A.y.to("hour", inplace=True)
    A.y.title = u"Acquisition time"

    # TODO: A.copy().plot(use_plotly=True)


# ======================================================================================================================

if __name__ == "__main__":
    pass

# EOF
