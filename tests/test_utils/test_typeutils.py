import numpy as np
import pytest

from spectrochempy.utils.typeutils import is_array_like
from spectrochempy.utils.typeutils import is_datetime64
from spectrochempy.utils.typeutils import is_duck_array
from spectrochempy.utils.typeutils import is_iterable
from spectrochempy.utils.typeutils import is_number
from spectrochempy.utils.typeutils import is_numpy_array
from spectrochempy.utils.typeutils import is_quantity
from spectrochempy.utils.typeutils import is_sequence
from spectrochempy.utils.typeutils import is_sequence_with_quantity_elements
from spectrochempy.utils.typeutils import is_string


class MockQuantity:
    def __init__(self, magnitude, units):
        self._magnitude = magnitude
        self._units = units


class MockNDDataset:
    def __init__(self, data):
        self._data = data


def test_is_iterable():
    assert is_iterable([1, 2, 3])
    assert not is_iterable(123)


def test_is_number():
    assert is_number(123)
    assert not is_number("123")


def test_is_sequence():
    assert is_sequence([1, 2, 3])
    assert not is_sequence("123")


def test_is_string():
    assert is_string("123")
    assert not is_string(123)


def test_is_numpy_array():
    assert is_numpy_array(np.array([1, 2, 3]))
    assert not is_numpy_array([1, 2, 3])


def test_is_duck_array():
    class DuckArray:
        def __array_function__(self):
            pass

        shape = (3,)
        ndim = 1
        dtype = np.dtype("float64")

    assert is_duck_array(DuckArray())
    assert not is_duck_array(np.array([1, 2, 3]))


def test_is_quantity():
    assert is_quantity(MockQuantity(1, "m"))
    assert not is_quantity(123)


def test_is_sequence_with_quantity_elements():
    assert is_sequence_with_quantity_elements([MockQuantity(1, "m"), 2, 3])
    assert not is_sequence_with_quantity_elements([1, 2, 3])


def test_is_array_like():
    assert is_array_like(MockQuantity(1, "m"))
    assert not is_array_like(np.array([1, 2, 3]))


def test_is_datetime64():
    assert is_datetime64(np.datetime64("2021-01-01"))
    assert is_datetime64(MockNDDataset(np.array([np.datetime64("2021-01-01")])))
    assert not is_datetime64("2021-01-01")


if __name__ == "__main__":
    pytest.main()
