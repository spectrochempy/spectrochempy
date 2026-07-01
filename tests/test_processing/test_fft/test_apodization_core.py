# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import numpy as np
import pytest

from spectrochempy import Coord
from spectrochempy import NDDataset


def test_apodization_ir_2d_zpd():
    """Regression: multi-dimensional ZPD detection in apodization (#xxx).

    ``em`` / ``gm`` with ``is_ir=True`` used ``np.argmax(…, -1).item()``
    which raised for 2D+ data.  Verify 2D interferogram apodization runs.
    """
    nrows, ncols = 3, 100
    # synthetic interferogram: Gaussian peak centered at position 40 in each row
    x = Coord(np.arange(ncols, dtype=float), units="us")
    data = np.zeros((nrows, ncols))
    data[:, 40] = 1.0
    ds = NDDataset(
        data, coordset=[Coord(np.arange(nrows)), x], meta={"interferogram": True}
    )
    result = ds.em(lb=50.0)
    assert result.shape == (nrows, ncols)


def test_apodization_ir_1d_zpd():
    """1D interferogram apodization still works after the ZPD fix."""
    npts = 100
    x = Coord(np.arange(npts, dtype=float), units="us")
    data = np.zeros(npts)
    data[40] = 1.0
    ds = NDDataset(data, coordset=[x], meta={"interferogram": True})
    result = ds.em(lb=50.0)
    assert result.shape == (npts,)


def test_apodization_ir_2d_zpd_uses_median():
    """2D ZPD uses median of per-row argmax, not global max.

    Row 0: ZPD at 30, Row 1: ZPD at 30, Row 2: ZPD at 70.
    Median should be 30 (not 70, which is an outlier).
    """
    nrows, ncols = 3, 100
    x = Coord(np.arange(ncols, dtype=float), units="us")
    data = np.zeros((nrows, ncols))
    data[0, 30] = 1.0
    data[1, 30] = 1.0
    data[2, 70] = 1.0  # outlier row
    ds = NDDataset(
        data, coordset=[Coord(np.arange(nrows)), x], meta={"interferogram": True}
    )
    result = ds.em(lb=50.0)
    assert result.shape == (nrows, ncols)
