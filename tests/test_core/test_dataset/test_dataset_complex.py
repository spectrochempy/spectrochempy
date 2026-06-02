# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.core.units import ur
from spectrochempy.utils.testing import assert_array_equal


# ################ Complex and NMR ##################
# ======================================================================================
# Complex NDDataset tests
# ======================================================================================


def test_nddataset_real_imag():
    na = np.array(
        [[1.0 + 2.0j, 2.0 + 0j], [1.3 + 2.0j, 2.0 + 0.5j], [1.0 + 4.2j, 2.0 + 3j]]
    )
    nd = scp.NDDataset(na)
    # in the last dimension
    assert_array_equal(nd.real, na.real)
    assert_array_equal(nd.imag, na.imag)


def test_nddataset_create_from_complex_data():
    # 1D (complex)
    nd = scp.NDDataset([1.0 + 2.0j, 2.0 + 0j])
    assert nd.data.size == 2
    assert nd.size == 2
    assert nd.data.shape == (2,)
    assert nd.shape == (2,)

    # 2D (complex in the last dimension - automatic detection)
    nd = scp.NDDataset(
        [[1.0 + 2.0j, 2.0 + 0j], [1.3 + 2.0j, 2.0 + 0.5j], [1.0 + 4.2j, 2.0 + 3j]]
    )
    assert nd.data.size == 6
    assert nd.size == 6
    assert nd.data.shape == (3, 2)
    assert nd.shape == (3, 2)


def test_nddataset_complex_operations():
    # Test complex-specific operations
    nd = scp.NDDataset([1.0, 2.0], units="meter")
    coordset = scp.CoordSet(x=scp.Coord([10, 20]))
    nd.set_coordset(coordset)
    assert not nd.is_complex
    ndj = nd * 1j
    assert ndj.is_complex
    assert ndj.data.size == 2


def test_nddataset_complex_slicing():
    na0 = np.array([1.0 + 2.0j, 2.0 + 0j, 1.3 + 2.0j])
    nd = scp.NDDataset(na0)
    assert nd.is_complex
    assert nd.shape == (3,)

    # Test slicing preserves complex nature
    nd1 = nd[1:]
    assert nd1.is_complex
    assert nd1.shape == (2,)


def test_nddataset_init_complex_1D_with_mask():
    # test with complex with mask and units
    np.random.seed(12345)
    d = np.random.random((5)) * np.exp(0.1j)
    d1 = scp.NDDataset(d, units=ur.Hz)  # with units
    d1[1] = scp.MASKED
    assert d1.shape == (5,)
    assert d1._data.shape == (5,)
    assert d1.size == 5
    assert d1.dtype == np.complex128
    assert d1.is_complex
    assert d1.mask.shape[-1] == 5
    assert d1[2].data == d[2]
    d1R = d1.real
    assert not d1R.is_complex
    assert d1R._data.shape == (5,)
    assert d1R._mask.shape == (5,)


if __name__ == "__main__":
    pytest.main([__file__])
