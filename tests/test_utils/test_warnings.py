import warnings

import pytest

from spectrochempy.utils.warnings import assert_produces_warning


def test_assert_produces_warning():
    with assert_produces_warning(UserWarning):
        warnings.warn("This is a user warning", UserWarning, stacklevel=2)

    with pytest.raises(AssertionError), assert_produces_warning(UserWarning):
        warnings.warn("This is a runtime warning", RuntimeWarning, stacklevel=2)

    with assert_produces_warning(False):
        pass

    with pytest.raises(AssertionError), assert_produces_warning(False):
        warnings.warn("This is a user warning", UserWarning, stacklevel=2)


def test_assert_produces_warning_with_match():
    with assert_produces_warning(UserWarning, match="user warning"):
        warnings.warn("This is a user warning", UserWarning, stacklevel=2)

    with pytest.raises(AssertionError):  # noqa: SIM117
        with assert_produces_warning(UserWarning, match="user warning"):
            warnings.warn("This is a different warning", UserWarning, stacklevel=2)


if __name__ == "__main__":
    pytest.main()
