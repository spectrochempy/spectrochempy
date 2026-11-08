import json

import numpy as np
import pytest

import spectrochempy as scp


def _axis(model="gaussian", center=0, width=2):
    return {"model": model, "center": center, "width": width}


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_center_and_fwhm_for_separable_profiles(rank):
    template = scp.NDDataset(
        np.empty((3,) * rank),
        coordset=[scp.Coord([-1, 0, 1]) for _ in range(rank)],
    )
    axes = {
        dim: _axis("gaussian" if axis % 2 == 0 else "lorentzian")
        for axis, dim in enumerate(reversed(template.dims))
    }
    result = scp.simulate_peaks(template, [{"height": -4, "axes": axes}])

    expected = -4 * np.power(
        0.5, np.count_nonzero(np.indices(template.shape) != 1, axis=0)
    )
    np.testing.assert_allclose(result.data, expected, rtol=1e-14)
    assert result.dims == template.dims


def test_overlap_signed_heights_and_empty_sequence():
    template = scp.NDDataset([np.nan, np.inf, 17], coordset=[scp.Coord([-1, 0, 1])])
    positive = {"height": 3, "axes": {"x": _axis()}}
    negative = {"height": -1, "axes": {"x": _axis("lorentzian", center=1)}}

    combined = template.simulate_peaks([positive, negative])
    expected = 3 * np.array([0.5, 1, 0.5]) - np.array([0.2, 0.5, 1])
    np.testing.assert_allclose(combined.data, expected)
    np.testing.assert_array_equal(template.simulate_peaks([]).data, [0, 0, 0])


def test_irregular_reversed_coordinates_units_and_provenance():
    template = scp.NDDataset(
        np.full((3, 3), -99),
        units="V",
        mask=[[True, False, False], [False, False, False], [False, False, False]],
        coordset=[scp.Coord([5, 3, 2], units="s"), scp.Coord([1, 4, 6], units="Hz")],
    )
    template.annotate("measurement template")
    template.filename = "measurement.scp"
    source_history = template.history_entries
    peak = {
        "height": 2000 * scp.ur.mV,
        "axes": {
            "x": _axis(center=0.004 * scp.ur.kHz, width=4000 * scp.ur.mHz),
            "y": _axis("lorentzian", center=3000 * scp.ur.ms, width=2 * scp.ur.s),
        },
    }
    result = template.simulate_peaks([peak])

    assert result.units == template.units
    np.testing.assert_allclose(
        result.data, np.outer([0.2, 1, 0.5], [2**-2.25, 1, 0.5]) * 2
    )
    np.testing.assert_array_equal(result.mask, template.mask)
    np.testing.assert_array_equal(template.data, np.full((3, 3), -99))
    assert result.filename != template.filename
    assert result.history_entries[:-1] == source_history
    entry = result.history_entries[-1]
    assert entry["operation"] == "simulate_peaks"
    assert entry["parameters"]["peaks"] == [
        {
            "height": 2.0,
            "axes": {
                "y": {"model": "lorentzian", "center": 3.0, "width": 2.0},
                "x": {"model": "gaussian", "center": 4.0, "width": 4.0},
            },
        }
    ]
    json.dumps(entry["parameters"], allow_nan=False)
    assert peak["axes"]["x"]["center"].units == scp.ur.kHz
    result.clear_history()
    result.x.data[0] = -1
    result.mask[0, 0] = False
    assert template.history_entries == source_history
    assert template.x.data[0] == 1
    assert template.mask[0, 0]


def test_width_conversion_uses_temperature_intervals():
    template = scp.NDDataset([0, 0, 0], coordset=[scp.Coord([9, 10, 11], units="degC")])
    axis = _axis("lorentzian", scp.ur.Quantity(283.15, "K"), 3.6 * scp.ur.delta_degF)
    result = template.simulate_peaks([{"height": 1, "axes": {"x": axis}}])

    np.testing.assert_allclose(result.data, [0.5, 1, 0.5])


def test_off_grid_center_and_large_coordinate_origin():
    template = scp.NDDataset([0, 0], coordset=[scp.Coord([1e12, 1e12 + 1])])
    axis = _axis("lorentzian", center=1e12 + 0.5, width=1)
    result = template.simulate_peaks([{"height": 2, "axes": {"x": axis}}])

    np.testing.assert_allclose(result.data, [1, 1])


def test_scaled_distance_survives_subtraction_overflow():
    template = scp.NDDataset([0, 0], coordset=[scp.Coord([-1e308, 0])])
    axis = _axis("lorentzian", center=1e308, width=1e308)
    result = template.simulate_peaks([{"height": 1, "axes": {"x": axis}}])

    np.testing.assert_allclose(result.data, [1 / 17, 1 / 5])


def test_lorentzian_tail_remains_finite_beyond_square_overflow():
    template = scp.NDDataset([0, 0], coordset=[scp.Coord([0, 1e155])])
    peak = {"height": 1e308, "axes": {"x": _axis("lorentzian", width=1)}}
    result = template.simulate_peaks([peak])

    np.testing.assert_allclose(result.data, [1e308, 0.0025], rtol=1e-10)


def test_integer_grid_preserves_single_point_displacements_above_float_precision():
    origin = 2**53
    coord = scp.Coord([origin, origin + 1], dtype=np.int64)
    template = scp.NDDataset([0, 0], coordset=[coord])
    peak = {"height": 1, "axes": {"x": _axis("lorentzian", center=origin)}}

    np.testing.assert_allclose(template.simulate_peaks([peak]).data, [1, 0.5])


def test_integer_center_between_large_float_coordinates():
    origin = 2**53
    coord = scp.Coord([origin, origin + 2], dtype=np.float64)
    template = scp.NDDataset([0, 0], coordset=[coord])
    peak = {"height": 1, "axes": {"x": _axis("lorentzian", center=origin + 1)}}

    np.testing.assert_allclose(template.simulate_peaks([peak]).data, [0.5, 0.5])


@pytest.mark.skipif(
    np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant,
    reason="longdouble has no additional precision",
)
def test_extended_precision_centers_are_used_and_serialized():
    origin = np.longdouble(2**53)
    coord = scp.Coord([origin, origin + 1], dtype=np.longdouble)
    template = scp.NDDataset([0, 0], coordset=[coord])
    peak = {"height": 1, "axes": {"x": _axis("lorentzian", center=origin + 1)}}
    result = template.simulate_peaks([peak])

    np.testing.assert_allclose(result.data, [0.5, 1])
    parameters = result.history_entries[-1]["parameters"]
    center = parameters["peaks"][0]["axes"]["x"]["center"]
    assert np.longdouble(center) == origin + 1
    json.dumps(parameters, allow_nan=False)


@pytest.mark.parametrize(
    ("model", "position", "expected"),
    [("lorentzian", 1e170, 2.5e-33), ("gaussian", 20, 2.2490905336086767e-174)],
)
def test_height_can_scale_an_underflowing_profile_to_a_representable_value(
    model, position, expected
):
    template = scp.NDDataset([0, 0], coordset=[scp.Coord([0, position])])
    peak = {"height": 1e308, "axes": {"x": _axis(model, width=1)}}
    result = template.simulate_peaks([peak])

    assert result.data[0] == 1e308
    assert result.data[1] == pytest.approx(expected, rel=1e-10, abs=0)


def test_reference_coordinates_are_preserved():
    coords = scp.CoordSet(x=scp.Coord([0, 1], units="s"), y="x")
    template = scp.NDDataset([[0, 0], [0, 0]], coordset=coords)
    result = template.simulate_peaks(
        [{"height": 1, "axes": {"y": _axis(), "x": _axis()}}]
    )

    np.testing.assert_allclose(result.data, [[1, 0.5], [0.5, 0.25]])
    assert result.coordset.references == {"y": "x"}


@pytest.mark.parametrize(
    "peak",
    [
        {},
        {"height": True, "axes": {"x": _axis()}},
        {"height": 1j, "axes": {"x": _axis()}},
        {"height": np.inf, "axes": {"x": _axis()}},
        {"height": 1 * scp.ur.V, "axes": {"x": _axis()}},
        {"height": 1, "axes": {}},
        {"height": 1, "axes": {"y": _axis()}},
        {"height": 1, "axes": {"x": _axis("voigt")}},
        {"height": 1, "axes": {"x": _axis(center=1 * scp.ur.V)}},
        {"height": 1, "axes": {"x": _axis(center=np.nan)}},
        {"height": 1, "axes": {"x": _axis(width=0)}},
        {"height": 1, "axes": {"x": _axis(width=-1)}},
    ],
)
def test_rejects_invalid_peak_parameters(peak):
    template = scp.NDDataset([0, 0], coordset=[scp.Coord([0, 1], units="s")])
    with pytest.raises(ValueError):
        template.simulate_peaks([peak])


@pytest.mark.parametrize(
    "template",
    [
        scp.NDDataset(),
        scp.NDDataset([0, 0]),
        scp.NDDataset([0, 0], coordset=[scp.Coord([np.nan, 0])]),
        scp.NDDataset([0, 0], coordset=[scp.Coord(labels=["a", "b"])]),
    ],
)
def test_rejects_invalid_grids(template):
    with pytest.raises(ValueError):
        template.simulate_peaks([])
