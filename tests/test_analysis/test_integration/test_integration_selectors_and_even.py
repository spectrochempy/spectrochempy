# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
r"""
Regressions for the dimension selectors and for the removed ``even`` keyword of
``NDDataset.trapezoid()`` and ``NDDataset.simpson()``.

Two defects are covered.

The shared wrapper removed the dimension selector from the keywords forwarded
to SciPy with a truthiness test, ``if kwargs.get("dim")``. The integer ``0`` is
false, so ``dim=0`` survived the test and reached SciPy, raising
``TypeError: trapezoid() got an unexpected keyword argument 'dim'``. The ``axis``
synonym, which :meth:`~spectrochempy.core.dataset.basearrays.ndarray.NDArray.get_axis`
also accepts, was never removed at all, and collided with the ``axis`` argument
the wrapper passes explicitly, raising ``TypeError: got multiple values for
keyword argument 'axis'``. The ``dims`` synonym leaked in the same way as
``dim``.

``even`` was documented with the strategies ``'avg'``, ``'first'`` and ``'last'``
and a default of ``'avg'``. That default has not been SciPy's since 1.11.0, which
introduced ``'simpson'`` as the default; ``even`` was deprecated in 1.11.0 and
removed in 1.14.0. Every documented value therefore raised
``TypeError: simpson() got an unexpected keyword argument 'even'`` without any
mention of the deprecation or of the way forward.

With N samples there are N-1 intervals, and the Simpson 1/3 rule needs an even
number of intervals. The awkward case is therefore an *even* number of samples,
which gives an odd number of intervals. ``even`` had no effect at all for an odd
number of samples. The contract under test is:

- ``dim``, ``dims`` and ``axis`` are all accepted and all removed before the
  SciPy call, so they are equivalent, including for the integer ``0``, for a
  negative axis, and for a multi-dimensional dataset;
- the existing precedence between the synonyms is untouched, and an
  unsupported tuple or list selector is still rejected;
- any presence of ``even`` is refused, whether the value is a former strategy,
  an unknown string, or ``None``. For ``simpson`` the message names the removal,
  the current strategy and the risk of a changed result; for ``trapezoid``,
  which never had the parameter, it says so instead of describing a SciPy
  removal that never applied to it;
- the refusal happens before any computation or copy, so the source is intact;
- the masked-slice policy of #1698 is preserved: a slice that used a masked
  point yields a masked result with a raw ``NaN``, and the values hidden under
  the mask contribute nothing.

These tests use real ``NDDataset`` objects and the public integration API. The
calculation is never mocked, and valid results are compared against an
independent reference.
"""

import numpy as np
import pytest

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset

# ======================================================================================
# FIXTURES
# ======================================================================================


@pytest.fixture
def dataset_2d():
    """2D dataset, first dimension 'y' of size 2, second dimension 'x' of size 3."""
    y = Coord(np.array([10.0, 20.0]), title="temperature", units="K")
    x = Coord(np.array([0.0, 1.0, 2.0]), title="time", units="s")
    return NDDataset(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        coordset=[y, x],
        units="absorbance",
    )


@pytest.fixture
def masked_dataset_2d():
    """2D dataset with the first point of the second row masked."""
    y = Coord(np.array([10.0, 20.0]), title="temperature", units="K")
    x = Coord(np.array([0.0, 1.0, 2.0]), title="time", units="s")
    data = np.ma.MaskedArray(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        mask=np.array([[False, False, False], [True, False, False]]),
    )
    return NDDataset(data, coordset=[y, x], units="absorbance")


@pytest.fixture
def dataset_4_samples():
    """1D dataset with an even number of samples, hence an odd number of intervals."""
    x = Coord(np.array([0.0, 1.0, 2.0, 3.0]), title="time", units="s")
    return NDDataset(np.array([0.0, 1.0, 4.0, 9.0]), coordset=[x], units="V")


@pytest.fixture
def dataset_3_samples():
    """1D dataset with an odd number of samples, hence an even number of intervals."""
    x = Coord(np.array([0.0, 1.0, 2.0]), title="time", units="s")
    return NDDataset(np.array([0.0, 1.0, 4.0]), coordset=[x], units="V")


# ======================================================================================
# DIMENSION SELECTORS
# ======================================================================================

# Every accepted spelling of the selector, all expected to be equivalent.
# For the 2D fixture the first dimension is "y" (index 0, also index -2) and the
# last is "x" (index 1, also index -1).
SELECTORS_FIRST_DIM = [
    {"dim": 0},
    {"dim": -2},
    {"dim": "y"},
    {"dims": 0},
    {"dims": "y"},
    {"axis": 0},
    {"axis": -2},
    {"axis": "y"},
]
SELECTORS_LAST_DIM = [
    {"dim": 1},
    {"dim": -1},
    {"dim": "x"},
    {"dims": 1},
    {"dims": "x"},
    {"axis": 1},
    {"axis": -1},
    {"axis": "x"},
]


@pytest.mark.parametrize("method_name", ["trapezoid", "simpson"])
@pytest.mark.parametrize("selector", SELECTORS_FIRST_DIM)
def test_selectors_are_equivalent_on_first_dimension(dataset_2d, method_name, selector):
    # dim=0 is the case the truthiness test broke. Every spelling must give the
    # same result as the dimension name.
    method = getattr(dataset_2d, method_name)
    reference = method(dim="y")
    result = method(**selector)

    assert result.shape == reference.shape
    assert result.dims == reference.dims == ["x"]
    np.testing.assert_allclose(result.data, reference.data)
    assert result.units == reference.units


@pytest.mark.parametrize("method_name", ["trapezoid", "simpson"])
@pytest.mark.parametrize("selector", SELECTORS_LAST_DIM)
def test_selectors_are_equivalent_on_last_dimension(dataset_2d, method_name, selector):
    # The last dimension and a negative axis are the relevant controls.
    method = getattr(dataset_2d, method_name)
    reference = method()
    result = method(**selector)

    assert result.shape == reference.shape
    assert result.dims == reference.dims == ["y"]
    np.testing.assert_allclose(result.data, reference.data)


@pytest.mark.parametrize("method_name", ["trapezoid", "simpson"])
def test_positional_selector_matches_keyword(dataset_2d, method_name):
    # The positional form is mapped onto 'dim' and must be unaffected.
    method = getattr(dataset_2d, method_name)
    np.testing.assert_allclose(method("y").data, method(dim="y").data)
    np.testing.assert_allclose(method("y").data, method(axis=0).data)


@pytest.mark.parametrize("method_name", ["trapezoid", "simpson"])
@pytest.mark.parametrize("selector", SELECTORS_FIRST_DIM)
def test_selectors_match_independent_reference(dataset_2d, method_name, selector):
    # Valid results are compared against a hand-computed trapezoidal reference,
    # integrated along the first dimension: row areas over y = [10, 20] K.
    # Two samples per dimension give a single interval, which is a plain
    # trapezoid for Simpson too, so one reference serves both methods.
    result = getattr(dataset_2d, method_name)(**selector)
    lower = 10.0 * np.array([1.0, 2.0, 3.0])
    upper = 10.0 * np.array([4.0, 5.0, 6.0])
    expected = 0.5 * (lower + upper)
    np.testing.assert_allclose(result.data, expected)


@pytest.mark.parametrize("method_name", ["trapezoid", "simpson"])
@pytest.mark.parametrize("selector", SELECTORS_FIRST_DIM)
def test_selectors_preserve_reduced_mask(masked_dataset_2d, method_name, selector):
    # The mask policy of #1698 must survive the selector correction: the first
    # row is complete, the second row used a masked point.
    result = getattr(masked_dataset_2d, method_name)(**selector)

    assert result.shape == (3,)
    assert result.mask.shape == result.data.shape
    # Reducing along the first dimension keeps one output per column of the
    # source, so the masked point at (row 1, column 0) makes the first output
    # unavailable, and the two remaining columns stay valid.
    np.testing.assert_array_equal(result.mask, [True, False, False])
    # The unavailable slice stores a raw NaN in .data, while the complete
    # slices keep their finite areas.
    assert np.isnan(result.data[0])
    assert np.isfinite(result.data[1:]).all()


@pytest.mark.parametrize("method_name", ["trapezoid", "simpson"])
@pytest.mark.parametrize("selector", SELECTORS_FIRST_DIM)
def test_selectors_preserve_coordinates_units_and_source(
    dataset_2d, method_name, selector
):
    method = getattr(dataset_2d, method_name)
    before_data = dataset_2d.data.copy()
    before_mask = dataset_2d.mask

    result = method(**selector)

    # The surviving coordinate, its values, its units and its title are
    # preserved, because the first dimension is the one being reduced.
    np.testing.assert_array_equal(result.x.data, [0.0, 1.0, 2.0])
    assert result.x.units == dataset_2d.x.units
    assert result.x.title == "time"
    # Units combine the data units with the integrated coordinate units, here K.
    assert result.units == dataset_2d.units * dataset_2d.y.units
    # The source is not mutated.
    np.testing.assert_array_equal(dataset_2d.data, before_data)
    assert dataset_2d.mask is before_mask


@pytest.mark.parametrize("method_name", ["trapezoid", "simpson"])
def test_dim_none_is_equivalent_to_no_selector(dataset_2d, method_name):
    # dim=None selects the default dimension, as omitting the selector does.
    method = getattr(dataset_2d, method_name)
    np.testing.assert_allclose(method(dim=None).data, method().data)


@pytest.mark.parametrize("method_name", ["trapezoid", "simpson"])
def test_unmasked_results_are_identical_across_selectors(dataset_2d, method_name):
    # A single unmasked integration must not depend on how the axis was named.
    method = getattr(dataset_2d, method_name)
    baseline = method(dim="y")
    for selector in SELECTORS_FIRST_DIM:
        other = method(**selector)
        np.testing.assert_array_equal(other.data, baseline.data)
        assert other.mask is baseline.mask or np.array_equal(other.mask, baseline.mask)


# ======================================================================================
# THE REMOVED 'even' KEYWORD
# ======================================================================================

# The former documented strategies, the former default that is no longer SciPy's
# default, an unknown value, and None, which used to mean "use the default".
EVEN_VALUES = ["avg", "first", "last", "simpson", None, "unknown-strategy", 0]


@pytest.mark.parametrize("method_name", ["trapezoid", "simpson"])
@pytest.mark.parametrize("even_value", EVEN_VALUES)
def test_even_is_refused(dataset_4_samples, method_name, even_value):
    with pytest.raises(TypeError) as excinfo:
        getattr(dataset_4_samples, method_name)(even=even_value)

    message = str(excinfo.value)
    assert "even" in message
    # The message must name the method that was called.
    assert method_name in message
    # The value must be echoed so a caller can see what was rejected.
    assert repr(even_value) in message

    if method_name == "simpson":
        # simpson() really did lose the parameter, so the message explains the
        # removal, the way forward, and the result risk.
        assert "1.14.0" in message
        assert "Omit the keyword" in message
        assert "may change the result" in message
    else:
        # trapezoid() never had it, so talking about a SciPy removal or about a
        # Simpson strategy would be misleading.
        assert "1.14.0" not in message
        assert "no strategy" in message
        assert "simpson()" in message


@pytest.mark.parametrize("method_name", ["trapezoid", "simpson"])
def test_even_is_refused_for_odd_number_of_samples(dataset_3_samples, method_name):
    # 'even' never had an effect for an odd number of samples, but its presence
    # is still refused rather than silently ignored, so that no caller relies on
    # a keyword that no longer exists.
    with pytest.raises(TypeError, match="even"):
        getattr(dataset_3_samples, method_name)(even="avg")


@pytest.mark.parametrize("method_name", ["trapezoid", "simpson"])
def test_even_refusal_happens_before_computation(dataset_4_samples, method_name):
    # The source must be intact, and no result may be produced.
    before_data = dataset_4_samples.data.copy()
    before_mask = dataset_4_samples.mask
    before_history = list(dataset_4_samples.history)

    with pytest.raises(TypeError, match="even"):
        getattr(dataset_4_samples, method_name)(even="avg")

    np.testing.assert_array_equal(dataset_4_samples.data, before_data)
    assert dataset_4_samples.mask is before_mask
    assert dataset_4_samples.history == before_history


@pytest.mark.parametrize("method_name", ["trapezoid", "simpson"])
def test_even_refusal_precedes_dimension_resolution(dataset_4_samples, method_name):
    # The refusal is deterministic and happens first, so a caller passing both
    # 'even' and an invalid selector always gets the same, actionable error.
    with pytest.raises(TypeError, match="even"):
        getattr(dataset_4_samples, method_name)(dim=("y", "x"), even="avg")


@pytest.mark.parametrize("method_name", ["trapezoid", "simpson"])
def test_call_without_even_works_for_even_and_odd_sample_counts(
    dataset_4_samples, dataset_3_samples, method_name
):
    # Omitting the keyword is the supported path, for both sample parities.
    even_n = getattr(dataset_4_samples, method_name)()
    assert even_n.shape == ()
    assert np.isfinite(even_n.masked_data)

    odd_n = getattr(dataset_3_samples, method_name)()
    assert odd_n.shape == ()
    # Three samples of t**2 on a unit grid over [0, 2]. The Simpson 1/3 rule is
    # exact for a cubic and gives 8/3; the trapezoidal rule gives 3.
    expected = 8.0 / 3.0 if method_name == "simpson" else 3.0
    np.testing.assert_allclose(odd_n.data, expected)


def test_four_samples_simpson_is_exact_where_trapezoid_is_not(dataset_4_samples):
    # Four samples of t**2 on a unit grid over [0, 3] give three intervals, an odd
    # number, which is the case the former 'even' keyword existed to steer. This is
    # exactly the case a caller pinning the old default even='avg' got wrong.
    #
    # The integral of t**2 over [0, 3] is 9. The current Simpson strategy is exact
    # for a cubic, so it returns 9; the trapezoidal rule returns 9.5, so the two
    # methods are demonstrably not interchangeable here.
    simpson_result = dataset_4_samples.simpson()
    trapezoid_result = dataset_4_samples.trapezoid()

    np.testing.assert_allclose(simpson_result.data, 9.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(trapezoid_result.data, 9.5, rtol=0, atol=1e-12)
    assert simpson_result.data != trapezoid_result.data
    # The accuracy is the property being asserted, not incidental finiteness.
    assert abs(simpson_result.data - 9.0) < abs(trapezoid_result.data - 9.0)


@pytest.mark.parametrize("even_value", EVEN_VALUES)
def test_even_refusal_preserves_the_mask_policy(masked_dataset_2d, even_value):
    # A refused call must not disturb the masked-slice contract, and the same
    # dataset without 'even' must still produce a masked NaN slice.
    with pytest.raises(TypeError, match="even"):
        masked_dataset_2d.simpson(dim=0, even=even_value)

    result = masked_dataset_2d.simpson(dim=0)
    np.testing.assert_array_equal(result.mask, [True, False, False])
    assert np.isnan(result.data[0])
    assert np.isfinite(result.data[1:]).all()


def test_even_refusal_does_not_mutate_masked_source(masked_dataset_2d):
    before_data = masked_dataset_2d.data.copy()
    # 'data' is the plain ndarray: its mask lives in 'mask', so a prior copy of
    # 'mask' is what actually has to survive. Reading the mask off 'data' would
    # compare an all-False array and assert nothing.
    before_mask = masked_dataset_2d.mask.copy()
    assert before_mask.any(), "fixture must really carry a masked point"

    with pytest.raises(TypeError, match="even"):
        masked_dataset_2d.simpson(dim=0, even="avg")

    np.testing.assert_array_equal(masked_dataset_2d.data, before_data)
    np.testing.assert_array_equal(masked_dataset_2d.mask, before_mask)
