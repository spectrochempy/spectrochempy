# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for the internal cross-validation dataset helpers."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.model_selection import KFold

import spectrochempy as scp
from spectrochempy.analysis._cross_validation import _prepare_fold_subsets
from spectrochempy.analysis._cross_validation import _resolve_sample_geometry
from spectrochempy.analysis._cross_validation import _restore_prediction_geometry
from spectrochempy.analysis._cross_validation import _validate_fold_positions
from spectrochempy.utils._estimator import clone_unfitted
from spectrochempy.utils.exceptions import SpectroChemPyError


@pytest.fixture()
def aligned_data():
    """Return deterministic spectra and aligned multivariate targets."""
    n_observations = 7
    latent = np.linspace(-1.5, 1.5, n_observations)
    X_values = np.column_stack(
        [latent, latent**2, np.sin(latent), np.cos(latent), latent**3]
    )
    y_values = np.column_stack([1.0 + 2.0 * latent, latent**2 - 0.5])
    samples = scp.Coord(
        np.arange(n_observations, dtype=float),
        labels=[f"sample-{index}" for index in range(n_observations)],
        title="sample",
        units="s",
    )
    features = scp.Coord.linspace(
        1000.0, 1400.0, X_values.shape[1], title="wavenumber", units="cm^-1"
    )
    targets = scp.Coord([0, 1], labels=["concentration", "temperature"], title="target")
    X = scp.NDDataset(
        X_values,
        coordset=[samples, features],
        dims=["y", "x"],
        units="absorbance",
        title="spectra",
    )
    X.mask = np.array(
        [
            [False, True, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ]
    )
    X.meta.instrument = "synthetic"
    y = scp.NDDataset(
        y_values,
        coordset=[samples.copy(), targets],
        dims=["y", "x"],
        units="mmol/L",
        title="response",
    )
    y.mask = np.array(
        [
            [False, False],
            [False, False],
            [False, False],
            [False, True],
            [False, False],
            [False, False],
            [False, False],
        ]
    )
    y.meta.method = "synthetic"
    return X, y


@pytest.mark.parametrize(
    ("transpose_X", "transpose_y", "expected_axes"),
    [
        (False, False, (0, 0)),
        (True, False, (1, 0)),
        (False, True, (0, 1)),
        (True, True, (1, 1)),
    ],
)
def test_sample_dimension_is_resolved_independently(
    aligned_data, transpose_X, transpose_y, expected_axes
):
    X, y = aligned_data
    X = X.T if transpose_X else X
    y = y.T if transpose_y else y

    geometry = _resolve_sample_geometry(X, y, sample_dim="y")

    assert (geometry.x_sample_axis, geometry.y_sample_axis) == expected_axes
    assert geometry.n_observations == 7
    assert geometry.sample_dim == "y"


def test_nondefault_sample_dimension_name_is_resolved_on_each_dataset():
    samples = scp.Coord(np.arange(5), labels=[f"s{index}" for index in range(5)])
    X = scp.NDDataset(
        np.arange(15.0).reshape(3, 5),
        coordset=[scp.Coord([10, 20, 30]), samples],
        dims=["z", "x"],
    )
    y = scp.NDDataset(
        np.arange(10.0).reshape(5, 2),
        coordset=[samples.copy(), scp.Coord([0, 1])],
        dims=["x", "z"],
    )

    geometry = _resolve_sample_geometry(X, y, sample_dim="x")

    assert (geometry.x_sample_axis, geometry.y_sample_axis) == (1, 0)


@pytest.mark.parametrize(
    ("missing_X", "missing_y"), [(True, True), (True, False), (False, True)]
)
def test_missing_observation_coordinates_use_positional_alignment(
    aligned_data, missing_X, missing_y
):
    X, y = aligned_data
    if missing_X:
        X.set_coordset(y=None, x=X.coordset["x"].copy())
    if missing_y:
        y.set_coordset(y=None, x=y.coordset["x"].copy())

    geometry = _resolve_sample_geometry(X, y)

    assert geometry.n_observations == 7


def test_physically_equivalent_observation_coordinate_units_are_accepted(aligned_data):
    X, y = aligned_data
    y_sample = scp.Coord(
        np.arange(7, dtype=float) * 1000.0,
        labels=[f"sample-{index}" for index in range(7)],
        units="ms",
    )
    y.set_coordset(y=y_sample, x=y.coordset["x"].copy())

    geometry = _resolve_sample_geometry(X, y)

    assert geometry.n_observations == 7
    assert y.coordset["y"].units == scp.ur.ms


@pytest.mark.parametrize(
    "mismatch", ["values", "order", "labels", "units", "missing-units"]
)
def test_observation_coordinate_mismatches_are_rejected(aligned_data, mismatch):
    X, y = aligned_data
    values = np.arange(7, dtype=float)
    labels = [f"sample-{index}" for index in range(7)]
    units = "s"
    if mismatch == "values":
        values[3] += 0.25
    elif mismatch == "order":
        values = values[::-1]
        labels = labels[::-1]
    elif mismatch == "labels":
        labels[3] = "different-sample"
    elif mismatch == "units":
        units = "m"
    else:
        units = None
    y_sample = scp.Coord(values, labels=labels, units=units)
    y.set_coordset(y=y_sample, x=y.coordset["x"].copy())

    with pytest.raises(SpectroChemPyError, match="Observation coordinate"):
        _resolve_sample_geometry(X, y)


@pytest.mark.parametrize(
    ("X_factory", "y_factory", "sample_dim", "message"),
    [
        (lambda X: X.data, lambda y: y, "y", "NDDataset"),
        (
            lambda X: scp.NDDataset(np.arange(7.0), dims=["y"]),
            lambda y: y,
            "y",
            "two-dimensional",
        ),
        (
            lambda X: X,
            lambda y: scp.NDDataset(np.ones((2, 2, 7)), dims=["z", "y", "x"]),
            "y",
            "one- or two-dimensional",
        ),
        (lambda X: X, lambda y: y, "z", "not present in X"),
        (lambda X: X, lambda y: y[:5], "y", "same number"),
    ],
)
def test_unsupported_dataset_geometries_are_rejected(
    aligned_data, X_factory, y_factory, sample_dim, message
):
    X, y = aligned_data

    with pytest.raises(SpectroChemPyError, match=message):
        _resolve_sample_geometry(X_factory(X), y_factory(y), sample_dim=sample_dim)


def test_ambiguous_sample_dimension_is_rejected(aligned_data):
    X, y = aligned_data
    X.dims = ["y", "y"]

    with pytest.raises(SpectroChemPyError, match="exactly one axis"):
        _resolve_sample_geometry(X, y)


def test_multiple_observation_coordinates_are_rejected_explicitly(aligned_data):
    X, y = aligned_data
    X.set_coordset(
        y=[X.coordset["y"].copy(), scp.Coord(np.arange(7), title="run")],
        x=X.coordset["x"].copy(),
    )

    with pytest.raises(SpectroChemPyError, match="multiple coordinates"):
        _resolve_sample_geometry(X, y)


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        (2, "one-dimensional"),
        ([[0, 1]], "one-dimensional"),
        ([], "must not be empty"),
        ([0.0, 1.0], "must be integers"),
        ([False, True], "must be integers"),
        ([-1, 1], "outside"),
        ([0, 7], "outside"),
        ([0, 0], "duplicates"),
    ],
)
def test_invalid_fold_positions_are_rejected(positions, message):
    with pytest.raises(SpectroChemPyError, match=message):
        _validate_fold_positions(positions, n_observations=7, name="Validation")


def test_overlapping_fold_positions_are_rejected(aligned_data):
    X, y = aligned_data

    with pytest.raises(SpectroChemPyError, match="disjoint"):
        _prepare_fold_subsets(X, y, [0, 1, 2], [2, 3])


@pytest.mark.parametrize(
    ("transpose_X", "transpose_y"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_fold_subsets_are_aligned_row_oriented_and_preserve_metadata(
    aligned_data, transpose_X, transpose_y
):
    X, y = aligned_data
    X = X.T if transpose_X else X
    y = y.T if transpose_y else y
    train = np.array([5, 0, 2, 1])
    validation = np.array([6])

    fold = _prepare_fold_subsets(X, y, train, validation)

    assert fold.X_train.shape == (4, 5)
    assert fold.X_validation.shape == (1, 5)
    assert fold.y_train.shape == (4, 2)
    assert fold.y_validation.shape == (1, 2)
    assert fold.X_train.dims[0] == "y"
    assert fold.y_train.dims[0] == "y"
    expected_X = np.take(X.data, train, axis=X.dims.index("y"))
    if transpose_X:
        expected_X = expected_X.T
    assert_allclose(fold.X_train.data, expected_X)
    expected_y = np.take(y.data, train, axis=y.dims.index("y"))
    if transpose_y:
        expected_y = expected_y.T
    assert_allclose(fold.y_train.data, expected_y)
    assert fold.X_train.coordset["y"].labels.tolist() == [
        "sample-5",
        "sample-0",
        "sample-2",
        "sample-1",
    ]
    assert fold.X_train.coordset["x"] == X.coordset["x"]
    assert fold.y_train.coordset["x"] == y.coordset["x"]
    assert fold.X_train.units == X.units
    assert fold.y_train.units == y.units
    assert fold.X_train.meta.instrument == "synthetic"
    assert fold.y_train.meta.method == "synthetic"
    assert fold.X_train.mask[1, 1]
    assert fold.target_geometry.template.shape == tuple(
        1 if dim == "y" else 2 for dim in y.dims
    )
    assert fold.target_geometry.template.dims == y.dims


def test_univariate_targets_and_singleton_dimensions_are_preserved(aligned_data):
    X, y = aligned_data
    y = y[:, :1]

    fold = _prepare_fold_subsets(X.T, y.T, [0, 1, 2, 3, 4, 5], [6])

    assert fold.X_validation.shape == (1, 5)
    assert fold.y_validation.shape == (1, 1)
    assert fold.target_geometry.template.shape == (1, 1)
    assert fold.target_geometry.template.dims == ["x", "y"]


def test_one_dimensional_targets_are_supported():
    samples = scp.Coord(np.arange(5), labels=[f"s{index}" for index in range(5)])
    X = scp.NDDataset(
        np.arange(15.0).reshape(5, 3),
        coordset=[samples, scp.Coord([10, 20, 30])],
        dims=["y", "x"],
    )
    y = scp.NDDataset(
        np.arange(5.0), coordset=[samples.copy()], dims=["y"], units="mol/L"
    )

    fold = _prepare_fold_subsets(X, y, [0, 1, 2, 3], [4])
    restored = _restore_prediction_geometry(
        fold.y_validation.copy(), fold.target_geometry
    )

    assert fold.y_validation.shape == (1,)
    assert restored.shape == (1,)
    assert restored.dims == ["y"]


def test_fold_subsets_are_isolated_from_sources_and_each_other(aligned_data):
    X, y = aligned_data
    original_X = X.copy()
    original_y = y.copy()
    first = _prepare_fold_subsets(X, y, [0, 1, 2, 3], [4, 5])
    second = _prepare_fold_subsets(X, y, [2, 3, 4, 5], [0, 1])
    second_X = second.X_train.copy()
    second_y = second.y_validation.copy()

    first.X_train.data[2, 0] = 999.0
    first.X_train.mask = np.ones(first.X_train.shape, dtype=bool)
    first.y_validation.data[0, 0] = -999.0
    first.target_geometry.template.data[0, 0] = -888.0

    assert_allclose(X.data, original_X.data)
    assert np.array_equal(X.mask, original_X.mask)
    assert X.dims == original_X.dims
    assert X.coordset == original_X.coordset
    assert X.units == original_X.units
    assert_allclose(y.data, original_y.data)
    assert np.array_equal(y.mask, original_y.mask)
    assert y.dims == original_y.dims
    assert y.coordset == original_y.coordset
    assert y.units == original_y.units
    assert_allclose(second.X_train.data, second_X.data)
    assert_allclose(second.y_validation.data, second_y.data)


def test_helpers_are_not_exported_from_public_namespace():
    assert not hasattr(scp, "_prepare_fold_subsets")
    assert not hasattr(scp, "_restore_prediction_geometry")


@pytest.mark.parametrize("transpose_y", [False, True])
def test_prediction_round_trip_restores_target_geometry(aligned_data, transpose_y):
    X, y = aligned_data
    y = y.T if transpose_y else y
    validation = np.array([5, 1, 3])
    fold = _prepare_fold_subsets(X, y, [0, 2, 4, 6], validation)
    prediction = fold.y_validation.copy()
    prediction.data = prediction.data + 10.0
    prediction.mask = np.array(
        [[False, True], [False, False], [True, False]], dtype=bool
    )
    prediction.ito("mol/L")

    restored = _restore_prediction_geometry(prediction, fold.target_geometry)

    expected = np.take(y.data, validation, axis=y.dims.index("y")) + 10.0
    assert_allclose(restored.data, expected)
    assert restored.shape == fold.target_geometry.template.shape
    assert restored.dims == y.dims
    assert restored.coordset["y"] == fold.target_geometry.template.coordset["y"]
    assert restored.coordset["x"] == fold.target_geometry.template.coordset["x"]
    assert restored.units == y.units
    expected_mask = prediction.mask.T if transpose_y else prediction.mask
    assert np.array_equal(restored.mask, expected_mask)
    assert restored.meta.method == "synthetic"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("shape", "shape"),
        ("dimensions", "dimensions"),
        ("observation", "coordinate values"),
        ("target", "coordinate values"),
        ("missing-target", "defined on both"),
        ("units", "units are not compatible"),
    ],
)
def test_incoherent_predictions_are_rejected(aligned_data, change, message):
    X, y = aligned_data
    fold = _prepare_fold_subsets(X, y, [0, 1, 2, 3], [4, 5, 6])
    prediction = fold.y_validation.copy()
    if change == "shape":
        prediction = prediction[:2]
    elif change == "dimensions":
        prediction.dims = ["x", "y"]
    elif change == "observation":
        observation = prediction.coordset["y"].copy()
        observation.data = observation.data + np.array([0.0, 0.5, 0.0])
        prediction.set_coordset(y=observation, x=prediction.coordset["x"].copy())
    elif change == "target":
        target = prediction.coordset["x"].copy()
        target.data = target.data + np.array([0, 1])
        prediction.set_coordset(y=prediction.coordset["y"].copy(), x=target)
    elif change == "missing-target":
        prediction.set_coordset(y=prediction.coordset["y"].copy(), x=None)
    else:
        prediction.ito("s", force=True)

    with pytest.raises(SpectroChemPyError, match=message):
        _restore_prediction_geometry(prediction, fold.target_geometry)


def test_manual_oof_loop_restores_original_order_and_geometry(aligned_data):
    X, y = aligned_data
    X.mask = False
    y.mask = False
    X = X.T
    y = y.T
    oof = y.copy()
    oof.data = np.full(y.shape, np.nan)
    fold_order = []
    fold_predictions = []
    folds = list(KFold(n_splits=3, shuffle=True, random_state=7).split(np.arange(7)))

    for train, validation in reversed(folds):
        fold = _prepare_fold_subsets(X, y, train, validation)
        estimator = clone_unfitted(scp.PLSRegression(n_components=1, scale=False))
        estimator.fit(fold.X_train, fold.y_train)
        row_prediction = estimator.predict(fold.X_validation)
        prediction = _restore_prediction_geometry(row_prediction, fold.target_geometry)
        oof.data[:, validation] = prediction.data
        fold_order.extend(validation.tolist())
        fold_predictions.append(row_prediction.data)

    predictions_in_fold_order = np.concatenate(fold_predictions)
    assert fold_order != list(range(7))
    assert_allclose(oof.data[:, fold_order].T, predictions_in_fold_order)
    assert np.all(np.isfinite(oof.data))
    assert oof.shape == y.shape
    assert oof.dims == y.dims
    assert oof.coordset["y"] == y.coordset["y"]
    assert oof.coordset["x"] == y.coordset["x"]
    assert oof.units == y.units
