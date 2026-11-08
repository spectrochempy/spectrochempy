import json

import numpy as np
import pint
import pytest

import spectrochempy as scp


def test_1d_strict_threshold_and_signed_intensities():
    source = scp.NDDataset([-1, 0, -2, 2, 3])
    labels, regions = scp.segment_regions(source, threshold=-1, dim="x")

    np.testing.assert_array_equal(labels.data, [0, 1, 0, 2, 2])
    assert labels.dtype.kind == "i"
    assert labels.units is None
    assert labels.coordset is None
    assert regions == (
        {"label": 1, "size": 1, "bounds": {"x": (1, 2)}},
        {"label": 2, "size": 2, "bounds": {"x": (3, 5)}},
    )


@pytest.mark.parametrize(("connectivity", "second_label"), [("face", 2), ("full", 1)])
def test_2d_diagonal_connectivity(connectivity, second_label):
    source = scp.NDDataset([[3, 0], [0, 2]])
    labels, regions = source.segment_regions(
        threshold=1, axis=(0, -1), connectivity=connectivity
    )

    np.testing.assert_array_equal(labels.data, [[1, 0], [0, second_label]])
    assert len(regions) == second_label
    assert sum(row["size"] for row in regions) == 2


def test_excluded_cells_are_zero_masked_and_do_not_bridge_regions():
    source = scp.NDDataset(
        [3, 8, 2, np.nan, 4, np.inf, 1, -np.inf, 5],
        mask=[False, True, False, False, False, False, False, False, False],
    )
    labels, regions = source.segment_regions(threshold=1, dim=("x",))

    np.testing.assert_array_equal(labels.data, [1, 0, 2, 0, 3, 0, 0, 0, 4])
    np.testing.assert_array_equal(
        labels.mask, [False, True, False, True, False, True, False, True, False]
    )
    assert [row["size"] for row in regions] == [1, 1, 1, 1]


def test_overlapping_boxes_do_not_define_membership():
    data = np.ones((5, 5))
    data[1:-1, 1:-1] = 0
    data[2, 2] = 2
    labels, regions = scp.NDDataset(data).segment_regions(threshold=0, dim=("y", "x"))

    assert regions == (
        {"label": 1, "size": 16, "bounds": {"y": (0, 5), "x": (0, 5)}},
        {"label": 2, "size": 1, "bounds": {"y": (2, 3), "x": (2, 3)}},
    )
    assert labels.data[2, 2] == 2
    assert labels.data[1, 2] == 0


def test_native_axes_and_labels_are_independent_of_selector_and_memory_order():
    data = np.array([[0, 0, 2], [2, 0, 0], [2, 0, 0]])
    coords = [
        scp.Coord([10, 7, 1], units="s"),
        scp.Coord([1, 3, 9], units="Hz"),
    ]
    source = scp.NDDataset(data, dims=["u", "v"], coordset=coords)
    reordered = scp.NDDataset(np.asfortranarray(data), dims=["u", "v"])
    labels, regions = source.segment_regions(threshold=1, dim=("u", "v"))
    alternate, alternate_regions = reordered.segment_regions(
        threshold=1, axis=(np.int64(-1), "u")
    )

    assert labels.dims == ["u", "v"]
    assert regions == alternate_regions
    np.testing.assert_array_equal(labels.data, [[0, 0, 1], [2, 0, 0], [2, 0, 0]])
    np.testing.assert_array_equal(labels.data, alternate.data)
    for name in source.dims:
        np.testing.assert_array_equal(labels.coord(name).data, source.coord(name).data)
        assert labels.coord(name).units == source.coord(name).units
    labels.coord("u").title = "derived"
    assert source.coord("u").title != "derived"
    parameters = alternate.history_entries[-1]["parameters"]
    assert parameters["requested"] == [-1, "u"]
    assert parameters["axes"] == [1, 0]
    assert parameters["dims"] == ["v", "u"]


def test_coordinate_references_and_labels_are_preserved():
    coords = scp.CoordSet(x=scp.Coord([1, 3], labels=["a", "b"], units="s"), y="x")
    source = scp.NDDataset([[1, 0], [0, 1]], coordset=coords)
    labels, _ = source.segment_regions(threshold=0, dim=("y", "x"))

    assert labels.coordset.references == {"y": "x"}
    np.testing.assert_array_equal(labels.x.labels, ["a", "b"])
    labels.x.data[0] = 2
    assert source.x.data[0] == 1


def test_unit_threshold_history_and_source_are_preserved():
    source = scp.NDDataset([1, 2, 0], units="V")
    source.annotate("source history")
    source.filename = "source.scp"
    history = source.history_entries
    labels, regions = source.segment_regions(threshold=1000 * scp.ur.mV, dim=0)

    np.testing.assert_array_equal(labels.data, [0, 1, 0])
    assert len(regions) == 1
    assert labels.units is None
    assert labels.filename != source.filename
    assert labels.history_entries[:-1] == history
    entry = labels.history_entries[-1]
    assert entry["operation"] == "segment_regions"
    parameters = entry["parameters"]
    assert parameters["threshold"] == {
        "requested": {"value": 1000, "units": str(scp.ur.mV)},
        "effective": {"value": 1, "units": str(scp.ur.V)},
    }
    assert parameters["comparison"] == ">"
    assert parameters["connectivity"] == "face"
    assert parameters["invalid_policy"] == "exclude_masked_nonfinite"
    assert parameters["n_regions"] == 1
    json.dumps(parameters, allow_nan=False)

    labels.clear_history()
    labels.data[1] = 4
    assert source.history_entries == history
    np.testing.assert_array_equal(source.data, [1, 2, 0])


@pytest.mark.parametrize("threshold_type", [int, float])
def test_integer_intensity_comparison_retains_native_precision(threshold_type):
    limit = 2**53
    source = scp.NDDataset([limit, limit + 1], dtype=np.int64)
    labels, _ = source.segment_regions(threshold=threshold_type(limit), dim=0)

    np.testing.assert_array_equal(labels.data, [0, 1])


@pytest.mark.parametrize(
    ("dtype", "precision"),
    [(np.float16, 11), (np.float32, 24), (np.float64, 53)],
)
@pytest.mark.parametrize("sign", [1, -1])
def test_integer_threshold_does_not_round_up_on_floating_grids(dtype, precision, sign):
    origin = 2**precision
    low, high = sorted((sign * (origin + 2), sign * (origin + 4)))
    source = scp.NDDataset([low, high], dtype=dtype)
    labels, _ = source.segment_regions(threshold=low + 1, dim=0)

    np.testing.assert_array_equal(labels.data, [0, 1])


@pytest.mark.parametrize("unit_format", ["", "L", "H"])
def test_external_pint_registry_threshold_is_converted_in_base_units(unit_format):
    registry = pint.UnitRegistry()
    registry.formatter.default_format = unit_format
    registry.define("custom_voltage = 2 * volt")
    source = scp.NDDataset([1, 2, 3], units="V")
    labels, _ = source.segment_regions(threshold=1 * registry.custom_voltage, dim=0)

    np.testing.assert_array_equal(labels.data, [0, 0, 1])
    assert labels.history_entries[-1]["parameters"]["threshold"] == {
        "requested": {"value": 1, "units": str(registry.custom_voltage)},
        "effective": {"value": 2, "units": str(scp.ur.V)},
    }
    with pytest.raises(ValueError, match="must match"):
        source.segment_regions(threshold=1 * registry.second, dim=0)


@pytest.mark.parametrize(
    ("units", "values", "dtype"),
    [("degC", [0.2, 1.2], np.float64), ("V", [2**53 + 1, 2**53 + 2], np.int64)],
)
def test_equivalent_external_pint_unit_preserves_exact_threshold(units, values, dtype):
    registry = pint.UnitRegistry()
    source = scp.NDDataset(values, units=units, dtype=dtype)
    threshold = registry.Quantity(values[0], units)
    labels, _ = source.segment_regions(threshold=threshold, dim=0)

    np.testing.assert_array_equal(labels.data, [0, 1])
    effective = labels.history_entries[-1]["parameters"]["threshold"]["effective"]
    assert effective["value"] == values[0]


def test_numpy_timedelta_is_not_an_axis_or_threshold():
    source = scp.NDDataset([0, 2])
    duration = np.timedelta64(0, "ns")
    with pytest.raises(TypeError, match="name or integer"):
        source.segment_regions(threshold=1, axis=duration)
    with pytest.raises(ValueError, match="finite real scalar"):
        source.segment_regions(threshold=duration, axis=0)


@pytest.mark.parametrize("axis_type", [np.int8, np.int16, np.uint16])
def test_all_numpy_integer_axis_types_are_accepted(axis_type):
    source = scp.NDDataset([0, 2])
    labels, _ = source.segment_regions(threshold=1, axis=axis_type(0))

    np.testing.assert_array_equal(labels.data, [0, 1])
    assert labels.history_entries[-1]["parameters"]["requested"] == [0]


def test_float32_intensities_are_compared_to_full_precision_threshold():
    source = scp.NDDataset([1, 0], dtype=np.float32)
    labels, _ = source.segment_regions(threshold=1 - 2**-30, dim=0)

    np.testing.assert_array_equal(labels.data, [1, 0])


@pytest.mark.parametrize(
    ("data", "masked"), [([1, 0], False), ([np.nan, np.inf], True)]
)
def test_empty_segmentation_preserves_grid_and_exclusion(data, masked):
    labels, regions = scp.NDDataset(data).segment_regions(threshold=1, dim=0)

    np.testing.assert_array_equal(labels.data, [0, 0])
    np.testing.assert_array_equal(np.broadcast_to(labels.mask, (2,)), [masked, masked])
    assert regions == ()
    assert labels.history_entries[-1]["parameters"]["n_regions"] == 0


@pytest.mark.parametrize(
    "selection",
    [
        {},
        {"dim": 0},
        {"axis": (0, 0)},
        {"dim": ("y", 0)},
        {"axis": (0, -3)},
        {"dim": ("y", True)},
        {"dim": ("y", None)},
        {"dim": ("y", "z")},
        {"dim": ("y", "x"), "axis": (0, 1)},
    ],
)
def test_invalid_selectors(selection):
    with pytest.raises((TypeError, ValueError)):
        scp.NDDataset([[1, 2], [3, 4]]).segment_regions(threshold=0, **selection)


@pytest.mark.parametrize("threshold", [True, [1], np.array(1), np.nan, np.inf, 1j, "1"])
def test_invalid_thresholds(threshold):
    with pytest.raises(ValueError, match="finite real scalar"):
        scp.NDDataset([1, 2]).segment_regions(threshold=threshold, dim=0)


def test_incompatible_and_missing_threshold_units():
    with pytest.raises(ValueError, match="must match"):
        scp.NDDataset([1, 2], units="V").segment_regions(threshold=1 * scp.ur.s, dim=0)
    with pytest.raises(ValueError, match="requires dataset units"):
        scp.NDDataset([1, 2]).segment_regions(threshold=1 * scp.ur.V, dim=0)


@pytest.mark.parametrize(
    "source",
    [
        scp.NDDataset(),
        scp.NDDataset(np.empty((0, 2))),
        scp.NDDataset(np.zeros((2, 2, 2))),
        scp.NDDataset([1j, 2j]),
        scp.NDDataset([True, False], dtype=bool),
    ],
)
def test_unsupported_spectra(source):
    with pytest.raises(ValueError):
        source.segment_regions(threshold=0, dim=0)


def test_unsupported_connectivity():
    with pytest.raises(ValueError, match="connectivity"):
        scp.NDDataset([1, 2]).segment_regions(threshold=0, dim=0, connectivity=8)
