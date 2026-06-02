# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.utils.testing import (
    RandomSeedContext,
    assert_array_equal,
)


def test_nddataset_mask_valid():
    mask = np.random.random((10, 10)) > 0.5
    ds = scp.NDDataset(np.random.random((10, 10)), mask=mask)
    assert ds.is_masked
    assert ds.mask.shape == (10, 10)
    assert_array_equal(ds.mask, mask)


def test_nddataset_masked_array_input():
    a = np.random.randn(100)
    marr = np.ma.masked_where(a > 0, a)
    nd = scp.NDDataset(marr)
    # check that masks and data match
    assert_array_equal(nd.mask, marr.mask)
    assert_array_equal(nd.data, marr.data)
    # check that they are both by reference
    marr.mask[10] = ~marr.mask[10]
    marr.data[11] = 123456789
    assert_array_equal(nd.mask, marr.mask)
    assert_array_equal(nd.data, marr.data)


def test_nddataset_mask_array_input():
    marr = np.ma.array([1.0, 2.0, 5.0])  # Masked array with no masked entries
    nd = scp.NDDataset(marr)
    assert not nd.is_masked
    marr = np.ma.array([1.0, 2.0, 5.0], mask=[True, False, False])  # Masked array
    nd = scp.NDDataset(marr)
    assert nd.is_masked


def test_nddataset_unmasked_in_operation_with_masked_numpy_array():
    ndd = scp.NDDataset(np.array([1, 2, 3]))
    np_data = -np.ones_like(ndd)
    np_mask = np.array([True, False, True])
    np_arr_masked = np.ma.array(np_data, mask=np_mask)
    result1 = ndd * np_arr_masked
    assert result1.is_masked
    assert np.all(result1.mask == np_mask)
    # TODO: IndexError: in the future, 0-d boolean arrays will be
    #  interpreted as a valid boolean index
    # assert np.all(result1[~result1.mask].data == -ndd.data[~np_mask])
    result2 = np_arr_masked * ndd
    # Numpy masked  array return a masked array in this case
    # assert result2.is_masked
    assert np.all(
        result2.mask == np_mask
    )  # assert np.all(result2[  #  # ~result2.mask].data == -ndd.data[~np_mask])


@pytest.mark.parametrize("shape", [(10,), (5, 5), (3, 10, 10)])
def test_nddataset_mask_invalid_shape(shape):
    with pytest.raises(ValueError) as exc:
        with RandomSeedContext(789):
            scp.NDDataset(
                np.random.random((10, 10)), mask=np.random.random(shape) > 0.5
            )
    assert exc.value.args[0] == "mask {} and data (10, 10) shape mismatch!".format(
        shape
    )


@pytest.mark.parametrize(
    "mask_in", [np.array([True, False]), np.array([1, 0]), [True, False], [1, 0]]
)
def test_nddataset_mask_init_without_np_array(mask_in):
    ndd = scp.NDDataset(np.array([1, 1]), mask=mask_in)
    assert (ndd.mask == mask_in).all()


def test_nddataset_with_mask_acts_like_masked_array():
    # test for #2414
    input_mask = np.array([True, False, False])
    input_data = np.array([1.0, 2.0, 3.0])
    ndd_masked = scp.NDDataset(input_data.copy(), mask=input_mask.copy())
    #   ndd_masked = np.sqrt(ndd_masked)
    other = -np.ones_like(input_data)
    result1 = np.multiply(ndd_masked, other)
    result2 = ndd_masked * other
    result3 = other * ndd_masked
    result4 = other / ndd_masked
    # Test for both orders of multiplication
    for result in [result1, result2, result3, result4]:
        assert result.is_masked
        # Result mask should match input mask because other has no mask
        assert np.all(
            result.mask == input_mask
        )  # TODO:IndexError: in the   #  #  future, 0-d boolean arrays will be  #
        # interpreted  # as a  # valid  # boolean index  # assert np.all(result[~result.mask].data == -   #  #  #
        # input_data[~input_mask])


def test_nddataset_use_of_mask(dsm):
    nd = dsm.copy()
    # dsm already has a mask; verify more values can be masked
    original_mask_sum = nd.mask.sum()
    nd[950.0:1260.0] = scp.MASKED
    assert nd.is_masked
    # Additional mask entries were added
    assert nd.mask.sum() > original_mask_sum
    # Setting MASKED with values outside the coordinate range should be a no-op
    nd_noop = dsm.copy()
    mask_before = nd_noop.mask.sum()
    nd_noop[99999.0:100000.0] = scp.MASKED
    assert nd_noop.mask.sum() == mask_before


if __name__ == "__main__":
    pytest.main([__file__])
