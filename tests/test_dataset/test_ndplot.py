# -*- coding: utf-8 -*-
# flake8: noqa

import pytest

from spectrochempy import preferences, plt, os
from spectrochempy.utils import show


def test_plot_generic_1D(IR_dataset_1D):
    for method in ["scatter", "pen", "scatter+pen"]:
        dataset = IR_dataset_1D.copy()
        dataset.plot(method=method)

    # show()


def test_plot_generic_2D(IR_dataset_2D):
    for method in ["stack", "map", "image"]:
        dataset = IR_dataset_2D.copy()
        dataset.plot(method=method)

    # show()


prefs = preferences

styles = ["poster", "talk", "scpy", "sans", "serif", "grayscale", "notebook", "paper"]


@pytest.mark.parametrize("style", styles)
def test_styles(style):
    try:
        plt.style.use(style)
    except OSError:
        plt.style.use(os.path.join(prefs.stylesheets, style + ".mplstyle"))
