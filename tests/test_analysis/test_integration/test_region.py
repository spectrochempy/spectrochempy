import json

import numpy as np
import pint
import pytest

import spectrochempy as scp


def _dataset(values, grid, **kwargs):
    return scp.NDDataset(values, coordset=[scp.Coord(grid)], **kwargs)


def test_clipped_trapezoid_and_uncertainty_use_source_sample_weights():
    source = _dataset([0, 1, 3], [0, 1, 3])
    source.annotate("acquired spectrum")
    result = source.integrate_region({"x": (0.5, 2)}, noise_std=2)

    assert isinstance(result, scp.RegionIntegrationResult)
    assert result.valid and result.reason is None
    assert result.value == 1.875
    assert result.uncertainty == pytest.approx(2 * np.sqrt(43 / 32))
    assert result.uncertainty_status == "available"
    assert result.unit is None
    entry = result.history_entries[-1]
    assert entry["operation"] == "integrate_region"
    assert entry["parameters"]["contributing_samples"] == 3
    assert entry["parameters"]["invalid_samples"] == 0
    json.dumps(entry["parameters"], allow_nan=False)
    entry["parameters"]["dims"].clear()
    assert result.history_entries[-1]["parameters"]["dims"] == ["x"]
    assert len(source.history_entries) == 1


@pytest.mark.parametrize("rank", [2, 3])
def test_tensor_rule_on_multilinear_signal_with_units(rank):
    x = scp.Coord([0, 1, 3], units="s")
    y = scp.Coord([4, 1, 0], units="Hz")
    z = scp.Coord([-1, 0, 2], units="m")
    coords = [y, x] if rank == 2 else [z, y, x]
    data = -(1 + y.data[:, None]) * (2 + x.data)
    bounds = {"x": (0.5, 2), "y": (0.5, 3)}
    expected = -(2 * 1.5 + (2**2 - 0.5**2) / 2) * (2.5 + (3**2 - 0.5**2) / 2)
    units = scp.ur.V * scp.ur.s * scp.ur.Hz
    if rank == 3:
        data = (2 + z.data[:, None, None]) * data
        bounds["z"] = (-0.5, 1)
        expected *= 2 * 1.5 + (1 - 0.5**2) / 2
        units *= scp.ur.m
    source = scp.NDDataset(data, coordset=coords, units="V")

    result = scp.integrate_region(source, bounds, noise_std=1)
    assert result.value == pytest.approx(expected)
    assert result.unit == units
    assert source.T.integrate_region(bounds, noise_std=1).uncertainty == pytest.approx(
        result.uncertainty
    )
    reversed_axes = source[tuple(slice(None, None, -1) for _ in range(rank))]
    assert reversed_axes.integrate_region(bounds).value == pytest.approx(expected)
    full = {
        dim: (min(coord.data), max(coord.data))
        for dim, coord in zip(source.dims, coords, strict=True)
    }
    repeated = source.sort(dim="y", descend=False)
    for dim in tuple(source.dims):
        repeated = repeated.trapezoid(dim=dim)
    assert source.integrate_region(full).value == pytest.approx(repeated.data.item())


@pytest.mark.parametrize(
    ("values", "mask"),
    [
        ([0, 1, 3, 4], [True, False, False, False]),
        ([np.nan, 1, 3, 4], False),
        ([np.inf, 1, 3, 4], False),
    ],
)
def test_invalid_interpolation_neighbor_outside_box_invalidates(values, mask):
    source = _dataset(values, [0, 1, 3, 4], mask=mask)
    invalid = source.integrate_region({"x": (0.5, 2)}, noise_std=1)
    assert (invalid.valid, invalid.value, invalid.reason) == (
        False,
        None,
        "incomplete_support",
    )
    assert invalid.uncertainty is None
    assert invalid.uncertainty_status == "incomplete_support"
    assert source.integrate_region({"x": (1, 2)}).value == pytest.approx(1.5)


def test_background_estimation_masks_and_disjoint_support():
    source = _dataset(
        [0, 1, 2, 7, 3, 5],
        [0, 1, 2, 3, 4, 5],
        mask=[False, False, False, True, False, False],
    )
    result = source.integrate_region({"x": (0, 1.5)}, noise_bounds={"x": (3, 5)})
    assert result.value == pytest.approx(1.125)
    assert result.uncertainty == pytest.approx(
        np.sqrt(2) * np.linalg.norm([0.5, 0.875, 0.125])
    )
    assert result.history_entries[-1]["parameters"]["noise"]["valid_samples"] == 2
    insufficient = source.integrate_region({"x": (0, 1.5)}, noise_bounds={"x": (3, 4)})
    assert insufficient.valid and insufficient.uncertainty is None
    assert insufficient.uncertainty_status == "insufficient_background"
    with pytest.raises(ValueError, match="overlaps"):
        source.integrate_region({"x": (0, 1.5)}, noise_bounds={"x": (1.6, 4)})


def test_large_integer_noise_is_recorded_without_overflow():
    source = _dataset([1, 1], [0, 1])
    result = source.integrate_region({"x": (0, 1)}, noise_std=10**50)
    assert result.uncertainty == pytest.approx(10**50 / np.sqrt(2))
    assert result.history_entries[-1]["parameters"]["noise"]["sigma"] == 10**50


@pytest.mark.parametrize(("level", "dtype"), [(10**15, np.float64), (2**60, np.int64)])
def test_background_std_preserves_noise_on_a_large_baseline(level, dtype):
    source = _dataset([1, 1, 0, level, level + 1], [0, 1, 2, 3, 4], dtype=dtype)
    result = source.integrate_region({"x": (0, 1)}, noise_bounds={"x": (3, 4)})
    assert result.uncertainty == pytest.approx(0.5, rel=1e-15)


def test_compatible_external_pint_bounds_and_noise():
    registry = pint.UnitRegistry(system="cgs")
    registry.formatter.default_format = "L"
    source = scp.NDDataset(
        [1, 1, 1], coordset=[scp.Coord([0, 1, 2], units="s")], units="V"
    )
    result = source.integrate_region(
        {"x": (registry.Quantity(500, "ms"), registry.Quantity(1500, "ms"))},
        noise_std=registry.Quantity(200, "mV"),
    )
    assert result.value == pytest.approx(1)
    assert result.unit == scp.ur.V * scp.ur.s
    assert result.uncertainty == pytest.approx(
        0.2 * np.linalg.norm([0.125, 0.75, 0.125])
    )
    assert source.integrate_region({"x": (0, 1)}, noise_std=0).uncertainty == 0
    with pytest.raises(ValueError, match="Invalid bounds"):
        source.integrate_region({"x": (0, registry.Quantity(1, "V"))})


@pytest.mark.parametrize("grid", [[0, 0, 1], [0, 2, 1], [0, np.nan, 2], [1]])
def test_rejects_invalid_coordinates(grid):
    source = _dataset(np.ones(len(grid)), grid)
    with pytest.raises(ValueError, match="monotone"):
        source.integrate_region({"x": (0, 1)})


@pytest.mark.parametrize(
    "bounds",
    [
        {"y": (0, 1)},
        {"x": (1, 0)},
        {"x": (-1, 1)},
        {"x": (1, 1)},
        {"x": (0, np.inf)},
        {"x": (False, 1)},
    ],
)
def test_rejects_malformed_bounds(bounds):
    with pytest.raises(ValueError):
        _dataset([1, 2], [0, 1]).integrate_region(bounds)


def test_rejects_missing_coordinates_complex_data_and_conflicting_noise():
    with pytest.raises(ValueError, match="requires coordinates"):
        scp.NDDataset([1, 2]).integrate_region({"x": (0, 1)})
    with pytest.raises(ValueError, match="real dataset"):
        _dataset([1j, 2j], [0, 1]).integrate_region({"x": (0, 1)})
    source = _dataset([1, 2], [0, 1])
    with pytest.raises(ValueError, match="at most one"):
        source.integrate_region({"x": (0, 1)}, noise_std=0, noise_bounds={"x": (0, 1)})
    with pytest.raises(ValueError, match="nonnegative"):
        source.integrate_region({"x": (0, 1)}, noise_std=-1)


@pytest.mark.parametrize("units", ["degC", "dB"])
def test_rejects_nonmultiplicative_units(units):
    source = scp.NDDataset([0, 1], coordset=[scp.Coord([0, 1], units=units)])
    with pytest.raises(ValueError, match="multiplicative"):
        source.integrate_region({"x": (0, 1)})


def test_integer_grid_keeps_unit_spacing_above_float_precision():
    origin = 2**53
    source = scp.NDDataset(
        [1, 2],
        coordset=[scp.Coord([origin, origin + 1], dtype=np.int64)],
    )
    assert source.integrate_region({"x": (origin, origin + 1)}).value == 1.5


def test_intersected_cells_define_support_even_if_a_weight_underflows():
    span = np.finfo(np.longdouble).max / 4
    coord = scp.Coord([0, span], dtype=np.longdouble)
    source = scp.NDDataset([0, span], dtype=np.longdouble, coordset=[coord])
    bounds = {"x": (0, 1e-100)}

    result = source.integrate_region(bounds)
    assert not result.valid and result.reason == "numeric_range"
    assert result.history_entries[-1]["parameters"]["contributing_samples"] == 2

    source.mask = [False, True]
    result = source.integrate_region(bounds)
    assert not result.valid and result.reason == "incomplete_support"


def test_reversing_an_axis_preserves_reduction_order_under_cancellation():
    source = _dataset([2e20, -1e20, 2], [0, 1, 2])
    bounds = {"x": (0, 2)}
    assert source.integrate_region(bounds).value == 1
    assert source[::-1].integrate_region(bounds).value == 1
