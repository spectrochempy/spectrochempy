# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import pytest

from spectrochempy.core import info_


def test_findpeaks(IR_dataset_1D):
    dataset = IR_dataset_1D.copy()
    info_(dataset)

    peaks, properties = dataset[1800.0:1300.0].find_peaks(height=1.5, distance=50.0, width=0.0)
    assert len(peaks.x) == 2
    assert properties['peak_heights'][0] == pytest.approx(2.267, 0.001)
    assert properties['widths'][0] == pytest.approx(38.73, 0.001)
