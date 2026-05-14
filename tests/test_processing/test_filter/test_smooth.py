# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import numpy as np

from spectrochempy.utils.mplutils import show


def test_smooth_2D(IR_dataset_2D):
    dataset = IR_dataset_2D.copy()
    dataset /= dataset.real.data.max()  # nromalize
    dataset += np.random.random(dataset.shape[-1]) * 0.02

    s = dataset.smooth(size=21)
    (dataset + 0.25).plot(xlim=[4000, 3000])
    s.plot(cmap="copper", clear=False, xlim=[4000, 3000])

    s2 = s.smooth(size=21)
    (s2 - 0.25).plot(cmap="jet", clear=False, xlim=[4000, 3000])

    show()
