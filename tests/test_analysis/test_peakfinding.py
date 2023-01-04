# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa

import pytest

from spectrochempy.core.units import ur


def test_findpeaks(IR_dataset_1D):

    dataset = IR_dataset_1D.copy()

    # use_coord is True
    X = dataset[1800.0:1300.0]
    peaks, properties = X.find_peaks(height=1.5, distance=50.0, width=0.0)
    assert len(peaks.x) == 2
    assert peaks.x.units == 1 / ur.centimeter
    assert peaks.x.data[0] == pytest.approx(1644.044, 0.001)
    assert properties["peak_heights"][0].m == pytest.approx(2.267, 0.001)
    assert properties["widths"][0].m == pytest.approx(38.7309, 0.001)

    # use_coord is False
    peaks, properties = X.find_peaks(
        height=1.5, distance=50.0, width=0.0, use_coord=False
    )
    assert len(peaks.x) == 2
    assert peaks.x.data[0] == pytest.approx(162, 0.1)
    assert properties["peak_heights"][0] == pytest.approx(2.267, 0.001)
    assert properties["widths"][0] * abs(X.x.increment) == pytest.approx(38.7309, 0.001)
