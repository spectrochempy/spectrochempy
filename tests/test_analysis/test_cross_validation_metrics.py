# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for private cross-validation regression metric kernels."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

import spectrochempy as scp
from spectrochempy.analysis._cross_validation import _bias
from spectrochempy.analysis._cross_validation import _compute_regression_metrics
from spectrochempy.analysis._cross_validation import _evaluate_target_metrics
from spectrochempy.analysis._cross_validation import _mae
from spectrochempy.analysis._cross_validation import _rmse
from spectrochempy.utils.exceptions import SpectroChemPyError


def _targets(observed, predicted, *, units="mol/L"):
    samples = scp.Coord(
        np.arange(len(observed)),
        labels=[f"sample-{index}" for index in range(len(observed))],
        title="sample",
    )
    targets = scp.Coord(
        [10, 20], labels=["concentration", "temperature"], title="target"
    )
    observed = scp.NDDataset(
        np.asarray(observed),
        coordset=[samples, targets],
        dims=["y", "x"],
        units=units,
        title="observed",
    )
    predicted = scp.NDDataset(
        np.asarray(predicted),
        coordset=[samples.copy(), targets.copy()],
        dims=["y", "x"],
        units=units,
        title="predicted",
    )
    return observed, predicted


def test_known_errors_define_metric_values_units_and_residual_sign():
    observed, predicted = _targets(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        [[2.0, 8.0], [2.0, 21.0], [1.0, 33.0]],
    )

    result = _compute_regression_metrics(observed, predicted)

    assert_allclose(result.residuals.data, [[-1.0, 2.0], [0.0, -1.0], [2.0, -3.0]])
    assert_allclose(result.metric("rmse").values.data, np.sqrt([5.0 / 3.0, 14.0 / 3.0]))
    assert_allclose(result.metric("bias").values.data, [-1.0 / 3.0, 2.0 / 3.0])
    assert_allclose(result.metric("mae").values.data, [1.0, 2.0])
    assert_allclose(result.metric("r2").values.data, [-1.5, 0.93])
    assert result.metric("rmse").values.units == observed.units
    assert result.metric("bias").values.units == observed.units
    assert result.metric("mae").values.units == observed.units
    assert result.metric("r2").values.units is None
    assert result.residuals.units == observed.units
    assert result.residuals.coordset == observed.coordset
    assert result.metric("rmse").defined == (True, True)
    assert result.metric("rmse").reasons == (None, None)


def test_perfect_predictions_preserve_target_identity_and_inputs():
    observed, predicted = _targets(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
    )
    observed_before = observed.copy()
    predicted_before = predicted.copy()

    result = _compute_regression_metrics(observed, predicted)

    for name in ("rmse", "bias", "mae"):
        assert_allclose(result.metric(name).values.data, [0.0, 0.0])
    assert_allclose(result.metric("r2").values.data, [1.0, 1.0])
    assert result.metric("rmse").values.dims == ["x"]
    assert result.metric("rmse").values.coordset["x"] == observed.coordset["x"]
    assert result.n_valid.coordset["x"] == observed.coordset["x"]
    assert result.metric("rmse").values.coordset["x"].labels.tolist() == [
        "concentration",
        "temperature",
    ]
    assert_allclose(observed.data, observed_before.data)
    assert observed.coordset == observed_before.coordset
    assert_allclose(predicted.data, predicted_before.data)
    assert predicted.coordset == predicted_before.coordset


@pytest.mark.parametrize("invalid", ["masked", "nan", "positive-inf", "negative-inf"])
def test_invalid_prediction_only_invalidates_its_target(invalid):
    observed, predicted = _targets(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
    )
    if invalid == "masked":
        mask = np.zeros(predicted.shape, dtype=bool)
        mask[1, 0] = True
        predicted.mask = mask
    elif invalid == "nan":
        predicted.data[1, 0] = np.nan
    elif invalid == "positive-inf":
        predicted.data[1, 0] = np.inf
    else:
        predicted.data[1, 0] = -np.inf

    result = _compute_regression_metrics(observed, predicted)

    assert_allclose(result.n_valid.data, [2, 3])
    for name in ("rmse", "r2", "bias", "mae"):
        metric = result.metric(name)
        assert np.isnan(metric.values.data[0])
        assert metric.defined == (False, True)
        assert metric.reasons == ("invalid_prediction", None)
    if invalid == "masked":
        assert result.residuals.mask[1, 0]


@pytest.mark.parametrize("invalid", ["masked", "nan", "positive-inf", "negative-inf"])
def test_invalid_observations_are_rejected(invalid):
    observed, predicted = _targets(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
    )
    if invalid == "masked":
        mask = np.zeros(observed.shape, dtype=bool)
        mask[1, 0] = True
        observed.mask = mask
    elif invalid == "nan":
        observed.data[1, 0] = np.nan
    elif invalid == "positive-inf":
        observed.data[1, 0] = np.inf
    else:
        observed.data[1, 0] = -np.inf

    with pytest.raises(SpectroChemPyError, match="Observed targets"):
        _compute_regression_metrics(observed, predicted)


def test_r2_reports_single_pair_and_zero_variance_reasons():
    one_observed, one_predicted = _targets([[2.0, 5.0]], [[3.0, 4.0]])
    one = _compute_regression_metrics(one_observed, one_predicted)

    assert one.metric("r2").defined == (False, False)
    assert one.metric("r2").reasons == (
        "insufficient_valid_pairs",
        "insufficient_valid_pairs",
    )
    assert_allclose(one.metric("rmse").values.data, [1.0, 1.0])
    assert_allclose(one.metric("bias").values.data, [1.0, -1.0])
    assert_allclose(one.metric("mae").values.data, [1.0, 1.0])

    constant_observed, constant_predicted = _targets(
        [[2.0, 1.0], [2.0, 2.0], [2.0, 3.0]],
        [[2.0, 1.0], [3.0, 2.0], [1.0, 3.0]],
    )
    constant = _compute_regression_metrics(constant_observed, constant_predicted)

    assert constant.metric("r2").defined == (False, True)
    assert constant.metric("r2").reasons == ("zero_observed_variance", None)
    assert_allclose(constant.metric("r2").values.data[1], 1.0)


def test_compatible_prediction_units_are_converted_without_mutation():
    observed, _ = _targets(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
    )
    _, predicted = _targets(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        [[1000.0, 10000.0], [2000.0, 20000.0], [3000.0, 30000.0]],
        units="mmol/L",
    )
    predicted_before = predicted.copy()

    result = _compute_regression_metrics(observed, predicted)

    assert_allclose(result.metric("rmse").values.data, [0.0, 0.0])
    assert result.metric("rmse").values.units == observed.units
    assert_allclose(predicted.data, predicted_before.data)
    assert predicted.units == predicted_before.units


@pytest.mark.parametrize(
    "unit_case", ["incompatible", "missing-observed", "missing-predicted"]
)
def test_incoherent_target_units_are_rejected(unit_case):
    observed, predicted = _targets(
        [[1.0, 10.0], [2.0, 20.0]], [[1.0, 10.0], [2.0, 20.0]]
    )
    if unit_case == "incompatible":
        predicted.ito("s", force=True)
    elif unit_case == "missing-observed":
        observed, _ = _targets(
            [[1.0, 10.0], [2.0, 20.0]],
            [[1.0, 10.0], [2.0, 20.0]],
            units=None,
        )
    else:
        _, predicted = _targets(
            [[1.0, 10.0], [2.0, 20.0]],
            [[1.0, 10.0], [2.0, 20.0]],
            units=None,
        )

    with pytest.raises(SpectroChemPyError, match="units"):
        _compute_regression_metrics(observed, predicted)


@pytest.mark.parametrize(
    "mismatch", ["shape", "dims", "sample-order", "target-order", "missing-labels"]
)
def test_geometry_mismatches_are_rejected_without_broadcasting(mismatch):
    observed, predicted = _targets(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
    )
    if mismatch == "shape":
        predicted = predicted[:, :1]
    elif mismatch == "dims":
        predicted.dims = ["x", "y"]
    elif mismatch == "sample-order":
        predicted = predicted[[1, 0, 2]]
    elif mismatch == "target-order":
        predicted = predicted[:, [1, 0]]
    else:
        target = scp.Coord(predicted.coordset["x"].data, title="target")
        predicted.set_coordset(y=predicted.coordset["y"].copy(), x=target)

    with pytest.raises(SpectroChemPyError, match="shape|dimensions|coordinate"):
        _compute_regression_metrics(observed, predicted)


def test_transposed_and_univariate_targets_have_explicit_geometry():
    observed, predicted = _targets(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
    )

    transposed = _compute_regression_metrics(observed.T, predicted.T)

    assert transposed.residuals.shape == (2, 3)
    assert transposed.residuals.dims == ["x", "y"]
    assert transposed.metric("rmse").values.dims == ["x"]
    assert transposed.metric("rmse").values.coordset["x"] == observed.coordset["x"]

    univariate_observed = observed[:, 0].squeeze()
    univariate_predicted = predicted[:, 0].squeeze()
    univariate = _compute_regression_metrics(univariate_observed, univariate_predicted)

    assert univariate.metric("rmse").values.ndim == 0
    assert univariate.n_valid.ndim == 0
    assert float(univariate.metric("rmse").values.data) == 0.0
    assert int(univariate.n_valid.data) == 3


def test_missing_target_coordinates_remain_missing_on_metric_outputs():
    observed, predicted = _targets(
        [[1.0, 10.0], [2.0, 20.0]], [[1.0, 10.0], [2.0, 20.0]]
    )
    observed.set_coordset(y=observed.coordset["y"].copy(), x=None)
    predicted.set_coordset(y=predicted.coordset["y"].copy(), x=None)

    result = _compute_regression_metrics(observed, predicted)

    assert result.metric("rmse").values.coordset is None
    assert result.n_valid.coordset is None


def test_private_metric_helpers_are_not_exported():
    assert not hasattr(scp, "_compute_regression_metrics")
    assert not hasattr(scp, "_RegressionMetricResult")


def test_global_rmse_is_not_the_mean_of_unequal_fold_rmse_values():
    first_fold = _rmse(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    second_fold = _rmse(np.array([0.0]), np.array([3.0]))
    global_rmse = _rmse(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 3.0]))

    assert_allclose(global_rmse, np.sqrt(3.0))
    assert_allclose((first_fold + second_fold) / 2.0, 1.5)
    assert global_rmse != pytest.approx((first_fold + second_fold) / 2.0)


def test_integer_arithmetic_is_promoted_before_subtraction():
    limit = np.iinfo(np.int64)
    observed = np.array([limit.min + 10, limit.max - 10], dtype=np.int64)
    predicted = np.array([limit.min + 11, limit.max - 11], dtype=np.int64)

    assert_allclose(_rmse(observed, predicted), 1.0)
    assert_allclose(_bias(observed, predicted), 0.0)
    assert_allclose(_mae(observed, predicted), 1.0)


def test_large_offset_preserves_small_float64_differences():
    observed = np.array([1.0e16, 1.0e16 + 2.0, 1.0e16 + 4.0], dtype=np.float64)
    predicted = observed + 2.0

    assert_allclose(_rmse(observed, predicted), 2.0, rtol=0.0, atol=0.0)
    assert_allclose(_bias(observed, predicted), 2.0, rtol=0.0, atol=0.0)
    n_valid, values, defined, reasons = _evaluate_target_metrics(
        observed, predicted, metrics=("r2",)
    )
    assert n_valid == 3
    assert_allclose(values["r2"], -0.5, rtol=0.0, atol=0.0)
    assert defined["r2"] is True
    assert reasons["r2"] is None


def test_nonfinite_numeric_result_is_reported_not_presented_as_valid():
    largest = np.finfo(float).max
    observed, predicted = _targets(
        [[-largest, 0.0], [largest, 1.0]],
        [[largest, 0.0], [-largest, 1.0]],
        units=None,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = _compute_regression_metrics(observed, predicted)

    for name in ("rmse", "mae"):
        metric = result.metric(name)
        assert np.isnan(metric.values.data[0])
        assert metric.defined[0] is False
        assert metric.reasons[0] == "non_finite_result"
    assert result.metric("bias").defined[0] is True
    assert_allclose(result.metric("bias").values.data[0], 0.0)
    assert result.metric("r2").defined[0] is True
    assert_allclose(result.metric("r2").values.data[0], -3.0)


def test_empty_numeric_input_has_explicit_undefined_reasons_without_warnings():
    n_valid, values, defined, reasons = _evaluate_target_metrics(
        np.array([], dtype=float),
        np.array([], dtype=float),
        metrics=("rmse", "r2", "bias", "mae"),
    )

    assert n_valid == 0
    assert all(np.isnan(value) for value in values.values())
    assert defined == {"rmse": False, "r2": False, "bias": False, "mae": False}
    assert set(reasons.values()) == {"no_observations"}


def test_metric_selection_is_private_but_strict():
    observed, predicted = _targets([[1.0, 2.0], [2.0, 3.0]], [[1.0, 2.0], [2.0, 3.0]])

    result = _compute_regression_metrics(observed, predicted, metrics=("mae",))

    assert [metric.name for metric in result.metrics] == ["mae"]
    with pytest.raises(SpectroChemPyError, match="Unsupported"):
        _compute_regression_metrics(observed, predicted, metrics=("rmsecv",))
    with pytest.raises(SpectroChemPyError, match="non-empty"):
        _compute_regression_metrics(observed, predicted, metrics="rmse")
