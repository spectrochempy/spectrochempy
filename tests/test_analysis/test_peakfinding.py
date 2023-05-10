# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
import os

import pytest

import spectrochempy as scp
from spectrochempy.core.units import ur
from spectrochempy.utils import docstrings as chd


# test docstring
# but this is not intended to work with the debugger - use run instead of debug!
@pytest.mark.skipif(
    os.environ.get("PYDEVD_LOAD_VALUES_ASYNC", None),
    reason="debug mode cause error when checking docstrings",
)
def test_findpeaks_docstrings():
    chd.PRIVATE_CLASSES = []  # do not test private class docstring
    module = "spectrochempy.analysis.peakfinding"
    chd.check_docstrings(
        module,
        obj=scp.find_peaks,
        # exclude some errors - remove whatever you want to check
        exclude=["SA01", "EX01", "ES01", "GL11", "GL08", "PR01"],
    )


def test_findpeaks(IR_dataset_1D):

    dataset = IR_dataset_1D.copy()

    # use_coord is True
    X = dataset[1800.0:1300.0]
    peaks, properties = X.find_peaks(height=1.5, distance=50.0, width=0.0)
    assert len(peaks.x) == 2
    assert peaks.x.units == 1 / ur.centimeter
    assert peaks.x.data[0] == pytest.approx(1644.044, 0.001)
    assert properties["peak_heights"][0].m == pytest.approx(2.267, 0.001)
    assert properties["widths"][0].m == pytest.approx(38.729, 0.001)

    # use_coord is False
    peaks, properties = X.find_peaks(
        height=1.5, distance=50.0, width=0.0, use_coord=False
    )
    assert len(peaks.x) == 2
    assert peaks.x.data[0] == pytest.approx(162, 0.1)
    assert properties["peak_heights"][0] == pytest.approx(2.267, 0.001)
    assert properties["widths"][0] * abs(X.x.spacing.m) == pytest.approx(38.729, 0.001)
