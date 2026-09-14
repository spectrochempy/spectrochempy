"""Tests for the public Savitzky-Golay differentiation facade."""

from __future__ import annotations

import inspect
import re

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.lazyimport.root_symbols import is_reserved_root_symbol


def _make_quadratic_dataset(*, descending=False):
    x_values = np.linspace(1.0, 10.0, 51)
    if descending:
        x_values = x_values[::-1]
    x = scp.Coord(x_values, title="time", units="s")
    y = scp.Coord([0.0, 1.0, 2.0], title="sample")
    data = np.vstack(
        [
            3.0 * x_values**2,
            2.0 * x_values**2 + 1.0,
            -(x_values**2),
        ]
    )
    dataset = scp.NDDataset(
        data,
        coordset=[y, x],
        title="signal",
        units="V",
    )
    dataset.name = "quadratic spectra"
    return dataset


def _history_messages(dataset):
    return [entry.split("> ", 1)[1] for entry in dataset.history]


def _assert_equivalent(actual, expected):
    np.testing.assert_allclose(actual.data, expected.data)
    np.testing.assert_array_equal(actual.mask, expected.mask)
    assert actual.dims == expected.dims
    assert actual.coordset == expected.coordset
    assert actual.units == expected.units
    assert actual.title == expected.title
    assert actual.name == expected.name
    assert _history_messages(actual) == _history_messages(expected)


class TestDifferentiatePublicAPI:
    def test_function_and_dataset_method_are_available(self):
        dataset = _make_quadratic_dataset()

        function_result = scp.differentiate(dataset)
        method_result = dataset.differentiate()

        _assert_equivalent(function_result, method_result)
        assert is_reserved_root_symbol("differentiate")

    def test_signature_distinguishes_derivative_and_polynomial_orders(self):
        signature = inspect.signature(scp.differentiate)

        assert signature.parameters["derivative_order"].default == 1
        assert signature.parameters["polynomial_order"].default == 2
        assert signature.parameters["size"].default == 5
        assert "deriv" not in signature.parameters
        assert "order" not in signature.parameters
        assert "kwargs" not in signature.parameters


class TestDifferentiateEquivalence:
    @pytest.mark.parametrize("derivative_order", [1, 2])
    def test_matches_savgol_for_supported_derivative_orders(self, derivative_order):
        dataset = _make_quadratic_dataset()
        parameters = {
            "size": 7,
            "polynomial_order": 3,
            "delta": None,
            "dim": "x",
            "mode": "mirror",
            "cval": 2.0,
        }

        actual = scp.differentiate(
            dataset,
            derivative_order=derivative_order,
            **parameters,
        )
        expected = scp.savgol(
            dataset,
            deriv=derivative_order,
            size=parameters["size"],
            order=parameters["polynomial_order"],
            delta=parameters["delta"],
            dim=parameters["dim"],
            mode=parameters["mode"],
            cval=parameters["cval"],
        )

        _assert_equivalent(actual, expected)

    @pytest.mark.parametrize("descending", [False, True])
    def test_uses_signed_physical_coordinate_spacing(self, descending):
        dataset = _make_quadratic_dataset(descending=descending)

        result = scp.differentiate(
            dataset,
            derivative_order=1,
            size=7,
            polynomial_order=3,
        )

        x_values = dataset.coord("x").data
        np.testing.assert_allclose(result.data[0, 4:-4], 6.0 * x_values[4:-4])
        assert np.all(result.data[0, 4:-4] > 0.0)

    def test_explicit_dimension_matches_savgol(self):
        dataset = _make_quadratic_dataset()

        actual = scp.differentiate(
            dataset,
            derivative_order=1,
            size=3,
            polynomial_order=2,
            dim="y",
            delta=1.0,
        )
        expected = scp.savgol(
            dataset,
            deriv=1,
            size=3,
            order=2,
            dim="y",
            delta=1.0,
            mode="interp",
            cval=0.0,
        )

        _assert_equivalent(actual, expected)

    def test_preserves_savgol_metadata_and_mask_contract(self):
        dataset = _make_quadratic_dataset()
        dataset[1] = scp.MASKED
        source_data = dataset.data.copy()
        source_mask = dataset.mask.copy()
        source_history = dataset.history.copy()

        actual = scp.differentiate(
            dataset,
            derivative_order=2,
            size=7,
            polynomial_order=3,
        )
        expected = scp.savgol(dataset, deriv=2, size=7, order=3)

        _assert_equivalent(actual, expected)
        assert actual.units == dataset.units / dataset.coord("x").units ** 2
        assert actual.title == "signal (2nd derivative)"
        assert np.all(actual.mask[1])
        np.testing.assert_array_equal(dataset.data, source_data)
        np.testing.assert_array_equal(dataset.mask, source_mask)
        assert dataset.history == source_history


class TestDifferentiateValidation:
    @pytest.mark.parametrize("derivative_order", [0, -1, 1.5, True])
    def test_rejects_non_positive_or_non_integer_derivative_order(
        self, derivative_order
    ):
        dataset = _make_quadratic_dataset()

        with pytest.raises(
            ValueError,
            match="derivative_order must be a positive integer",
        ):
            scp.differentiate(dataset, derivative_order=derivative_order)

    def test_rejects_derivative_order_above_polynomial_order(self):
        dataset = _make_quadratic_dataset()

        with pytest.raises(
            ValueError,
            match="derivative_order must not exceed polynomial_order",
        ):
            scp.differentiate(
                dataset,
                derivative_order=3,
                polynomial_order=2,
            )

    @pytest.mark.parametrize(
        ("differentiate_kwargs", "savgol_kwargs"),
        [
            (
                {"size": 5, "polynomial_order": 5},
                {"size": 5, "order": 5, "deriv": 1},
            ),
            ({"size": 4}, {"size": 4, "deriv": 1}),
        ],
        ids=["incompatible-polynomial", "invalid-window"],
    )
    def test_reuses_savgol_errors(self, differentiate_kwargs, savgol_kwargs):
        dataset = _make_quadratic_dataset()

        with pytest.raises(Exception) as reference_error:  # noqa: B017, PT011
            scp.savgol(dataset, **savgol_kwargs)
        with pytest.raises(
            type(reference_error.value),
            match=re.escape(str(reference_error.value)),
        ):
            scp.differentiate(dataset, **differentiate_kwargs)

    def test_nonuniform_coordinate_reuses_savgol_warning_and_fallback(self):
        dataset = _make_quadratic_dataset()
        x_values = dataset.x.data.copy()
        x_values[20] += 0.05
        dataset.x = scp.Coord(x_values, title="time", units="s")

        with pytest.warns(UserWarning, match="not uniformly spaced"):
            actual = scp.differentiate(
                dataset,
                derivative_order=1,
                size=7,
                polynomial_order=3,
            )
        with pytest.warns(UserWarning, match="not uniformly spaced"):
            expected = scp.savgol(dataset, deriv=1, size=7, order=3)

        _assert_equivalent(actual, expected)
