# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""Small shared helpers for Result-related tests.

These helpers intentionally stay lightweight. They cover only the repeated
runtime Result contract checks that are common across many estimator test
files.
"""


def assert_result_basics(fitted, expected_type, estimator_name):
    """Assert the generic runtime Result contract for a fitted estimator."""
    result = fitted.result
    assert isinstance(result, expected_type)
    assert result.estimator == estimator_name
    assert fitted.result is not fitted.result, (
        f"{expected_type.__name__} is recreated on every access; "
        "change this assertion if caching is added later"
    )
    text = repr(result)
    assert estimator_name in text
    return result


def assert_result_raises_before_fit(estimator, exc_type):
    """Assert that `.result` is unavailable before fitting."""
    import pytest

    with pytest.raises(exc_type):
        _ = estimator.result


def assert_fit_returns_self(estimator, *fit_args):
    """Assert that fit() preserves the estimator-returning convention."""
    ret = estimator.fit(*fit_args)
    assert ret is estimator
