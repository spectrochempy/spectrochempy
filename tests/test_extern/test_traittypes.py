import pytest
import numpy as np
from traitlets import HasTraits, TraitError, default

from spectrochempy.extern.traittypes import (
    Array,
    Empty,
)


# Test classes that use the trait types
class HasArray(HasTraits):
    arr = Array()
    arr_none = Array(allow_none=True)
    arr_dtype = Array(dtype=np.float64)


# Array trait tests
def test_array_default():
    """Test Array trait default values"""
    obj = HasArray()
    assert isinstance(obj.arr, np.ndarray)
    assert obj.arr.size == 1
    assert obj.arr == 0


def test_array_assignment():
    """Test Array trait assignment"""
    obj = HasArray()

    # Test valid assignments
    obj.arr = [1, 2, 3]
    assert np.array_equal(obj.arr, np.array([1, 2, 3]))

    obj.arr = np.array([4, 5, 6])
    assert np.array_equal(obj.arr, np.array([4, 5, 6]))

    # Test dtype enforcement
    obj.arr_dtype = [1, 2, 3]
    assert obj.arr_dtype.dtype == np.float64

    # Test None assignment
    with pytest.raises(TraitError):
        obj.arr = None

    obj.arr_none = None
    assert obj.arr_none is None


# Test validators
def test_array_validators():
    """Test trait validators"""

    class ValidatedArray(HasTraits):
        arr = Array().valid(lambda trait, value: value * 2)

    obj = ValidatedArray()
    obj.arr = np.array([1, 2, 3])
    assert np.array_equal(obj.arr, np.array([2, 4, 6]))


def test_empty_sentinel():
    """Test Empty sentinel behavior"""

    class HasEmptyArray(HasTraits):
        arr = Array(Empty)

    obj = HasEmptyArray()
    assert isinstance(obj.arr, np.ndarray)
    assert obj.arr.size == 1
    assert obj.arr == 0


def test_dynamic_default():
    """Test dynamic default values and data independence"""

    class HasDynamicArray(HasTraits):
        arr = Array()

        @default("arr")
        def _default_arr(self):
            return np.array([1, 2, 3])

    # Add print statements to debug initialization
    print("\nDebugging Array trait initialization:")
    obj1 = HasDynamicArray()
    print(f"obj1.arr initial value: {obj1.arr}")
    print(f"obj1._trait_values: {obj1._trait_values}")

    # Create two instances
    obj1 = HasDynamicArray()
    obj2 = HasDynamicArray()

    # Verify initial values are equal but separate
    assert obj1 is not obj2
    assert np.array_equal(obj1.arr, obj2.arr)
    assert obj1.arr is not obj2.arr  # Check they are different objects

    # Modify first instance
    obj1.arr[0] = 10

    # Verify obj2 remains unchanged
    assert obj1.arr[0] == 10
    assert obj2.arr[0] == 1
    assert not np.array_equal(obj1.arr, obj2.arr)


if __name__ == "__main__":
    pytest.main([__file__])
