# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""
Tests for the ndplugin module.

"""

import numpy as np
import pytest

from spectrochempy import Coord
from spectrochempy import NDDataset
from spectrochempy.processing.transformation.autosub import autosub
from spectrochempy.utils.mplutils import show

# autosub
# ------


def _make_synthetic_autosub_dataset():
    x = Coord(np.linspace(0.0, 5.0, 6), title="x")
    y = Coord(np.linspace(0.0, 3.0, 4), title="y")

    ref_x = np.array([0.5, -1.0, 2.0, 1.5, -0.25, 0.75])
    coef_y = np.array([1.0, 2.0, -0.5, 0.25])
    data = np.outer(coef_y, ref_x)

    dataset = NDDataset(data, coordset=[y, x])
    ref_last = NDDataset(ref_x, coordset=[x])
    ref_first = NDDataset(coef_y, coordset=[y])

    return dataset, ref_last, ref_first


@pytest.mark.data
def test_autosub(IR_dataset_2D):
    dataset = IR_dataset_2D

    ranges = [5000.0, 5999.0], [1940.0, 1820.0]

    s1 = dataset.copy()
    ref = s1[-1].squeeze()

    dataset.plot_stack()
    ref.plot(clear=False, linewidth=2.0, color="r")

    s2 = dataset.copy()

    s3 = s2.autosub(ref, *ranges, dim=-1, method="vardiff", inplace=False)
    s3.plot()

    # inplace = False
    assert np.round(s2.data[-1, 0], 4) != 0.0000
    assert np.round(s3.data[-1, 0], 4) == 0.0000
    s3.name = "vardiff"

    s3.plot_stack()

    s4 = dataset.copy()
    s4.autosub(ref, *ranges, method="ssdiff", inplace=True)
    s4.name = "ssdiff, inplace"
    assert np.round(s4.data[-1, 0], 4) == 0.0000

    s4.plot_stack()  # true avoid blocking due to graphs

    s4 = dataset.copy()
    s = autosub(s4, ref, *ranges, method="ssdiff")
    assert np.round(s4.data[-1, 0], 4) != 0.0000
    assert np.round(s.data[-1, 0], 4) == 0.0000
    s.name = "ssdiff direct call"

    s.plot_stack()

    # s5 = dataset.copy()
    # ref2 = s5[:, 0].squeeze()
    # ranges2 = [0, 5], [45, 54]

    # TODO: not yet implemented
    # s6 = s5.autosub(ref2, *ranges2, dim='y', method='varfit', inplace=False)
    # s6.plot()

    show()


def test_autosub_synthetic_last_dimension():
    dataset, ref_x, _ = _make_synthetic_autosub_dataset()

    result = dataset.autosub(
        ref_x,
        [0.0, 2.0],
        [3.0, 5.0],
        dim="x",
        method="ssdiff",
        inplace=False,
    )

    np.testing.assert_allclose(
        dataset.data, np.outer([1.0, 2.0, -0.5, 0.25], ref_x.data)
    )
    np.testing.assert_allclose(result.data, 0.0, atol=1.0e-10)


def test_autosub_synthetic_non_last_dimension():
    dataset, _, ref_y = _make_synthetic_autosub_dataset()

    result = dataset.autosub(
        ref_y,
        [0.0, 1.0],
        [2.0, 3.0],
        dim="y",
        method="ssdiff",
        inplace=False,
    )

    np.testing.assert_allclose(result.data, 0.0, atol=1.0e-10)
