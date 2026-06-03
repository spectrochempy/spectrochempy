# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import os

import matplotlib.pyplot as plt
import pytest

from spectrochempy import preferences
from spectrochempy.utils.mplutils import show


@pytest.mark.data
def test_plot_generic_1D(IR_dataset_1D):
    for method in ["scatter", "pen", "scatter+pen"]:
        dataset = IR_dataset_1D.copy()
        ax = dataset.plot(method=method)
        assert ax is not None
        assert len(ax.get_lines()) > 0

    show()


@pytest.mark.data
def test_plot_generic_2D(IR_dataset_2D):
    for method in ["stack", "map", "image"]:
        dataset = IR_dataset_2D.copy()
        ax = dataset.plot(method=method)
        assert ax is not None
        if method == "stack":
            assert len(ax.get_lines()) > 0
        else:
            assert ax.images or ax.collections

    show()


prefs = preferences

styles = ["poster", "talk", "scpy", "sans", "serif", "grayscale", "notebook", "paper"]


@pytest.mark.parametrize("style", styles)
def test_styles(style):
    try:
        plt.style.use(style)
    except OSError:
        plt.style.use(os.path.join(prefs.stylesheets, style + ".mplstyle"))
