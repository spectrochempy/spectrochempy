# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from numpy.testing import assert_allclose

import spectrochempy as scp


def test_fit_single_dataset(synthetic_two_peak_dataset, optimize_script):
    dataset = synthetic_two_peak_dataset

    f1 = scp.Optimize()
    f1.script = optimize_script
    f1.autobase = True
    f1.max_iter = 10
    result = f1.fit(dataset)

    assert result is f1
    assert f1.n_components == 2
    assert f1.components.shape == (3, dataset.size)
    assert f1.predict().shape == (1, dataset.size)
    assert f1.transform().shape == (1, 2)

    residual = f1.predict().squeeze() - dataset
    assert abs(residual.data).max() < 1e-6
    assert_allclose(
        [
            f1.fp["pos_line_1"],
            f1.fp["pos_line_2"],
            f1.fp["width_line_1"],
            f1.fp["width_line_2"],
        ],
        [3620.0, 3520.0, 200.0, 200.0],
        rtol=0.02,
        atol=3.0,
    )
