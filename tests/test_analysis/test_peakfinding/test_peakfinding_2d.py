import copy
import json

import numpy as np
import pytest

from spectrochempy.analysis.peakfinding.peakfinding import PeakFindingResult
from spectrochempy.analysis.peakfinding.peakfinding import PeakTable
from spectrochempy.analysis.peakfinding.peakfinding import find_peaks
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import ur


def _dataset(data, *, y=None, x=None, units="absorbance", mask=None, dtype=None):
    data = np.asarray(data, dtype=dtype)
    coords = []
    for dim, values in (("y", y), ("x", x)):
        if values is not None:
            coords.append(Coord(values, title=dim))
        else:
            coords.append(Coord(np.arange(data.shape[len(coords)]), title=dim))
    return NDDataset(data, coordset=coords, units=units, mask=mask)


def _two_peaks(*, y_units="ppm", x_units="ppm", reverse_y=False):
    y_values = np.linspace(10.0, 0.0, 31) if reverse_y else np.linspace(0.0, 10.0, 31)
    x_values = np.linspace(0.0, 6.0, 25)
    y_grid, x_grid = np.meshgrid(y_values, x_values, indexing="ij")
    data = np.exp(
        -((y_grid - y_values[7]) ** 2 / 0.5 + (x_grid - x_values[6]) ** 2 / 0.4)
    )
    data += 0.8 * np.exp(
        -((y_grid - y_values[23]) ** 2 / 0.7 + (x_grid - x_values[18]) ** 2 / 0.6)
    )
    return NDDataset(
        data,
        coordset=[
            Coord(y_values, title="y", units=y_units),
            Coord(x_values, title="x", units=x_units),
        ],
        units="absorbance",
    )


def test_2d_gaussian_peaks_have_structured_physical_results():
    dataset = _two_peaks()
    before = dataset.data.copy()
    mask_before = np.array(dataset.mask, copy=True)
    history_before = dataset.history_entries

    result = dataset.find_peaks(
        dims=("y", "x"),
        height=0.2,
        distance={"y": 1.0 * ur.ppm, "x": 1.0 * ur.ppm},
        as_result=True,
    )

    assert isinstance(result, PeakFindingResult)
    assert isinstance(result.peaks, PeakTable)
    assert result.table is result.peaks
    assert result.analysis_dims == ("y", "x")
    assert result.table.columns == (
        "index",
        "y_index",
        "x_index",
        "y_position",
        "x_position",
        "height",
    )
    rows = result.to_dict()
    assert [row["y_index"] for row in rows] == [7, 23]
    assert [row["x_index"] for row in rows] == [6, 18]
    assert [row["height"].magnitude for row in rows] == pytest.approx([1.0, 0.8])
    assert all(row["y_position"].units == ur.ppm for row in rows)
    assert all(row["x_position"].units == ur.ppm for row in rows)
    assert all(row["height"].units == ur.absorbance for row in rows)
    assert dataset.history_entries == history_before
    np.testing.assert_array_equal(dataset.data, before)
    np.testing.assert_array_equal(dataset.mask, mask_before)


def test_2d_default_return_and_empty_result_have_stable_schema():
    dataset = _two_peaks()
    table, properties = find_peaks(dataset, dims=("y", "x"), height=0.2)
    assert isinstance(table, PeakTable)
    assert len(table) == 2
    assert properties == {"peak_heights": pytest.approx([1.0, 0.8])}

    empty = _dataset(np.zeros((5, 7)))
    empty_table, empty_properties = find_peaks(empty, dims=("y", "x"), height=0.1)
    assert len(empty_table) == 0
    assert empty_table.to_dict() == []
    assert empty_table.columns == table.columns
    assert empty_properties == {}

    empty_result = find_peaks(empty, dims=("y", "x"), as_result=True)
    assert isinstance(empty_result, PeakFindingResult)
    assert isinstance(empty_result.peaks, PeakTable)
    assert empty_result.to_dict() == []
    assert empty_result.properties == {}
    assert empty_result.table.columns == table.columns
    assert empty_result.history_entries[-1]["parameters"]["n_peaks"] == 0


def test_2d_default_height_finds_only_strictly_positive_maxima():
    data = np.zeros((7, 7))
    data[2, 2] = 2
    data[4, 4] = -1
    table, _ = find_peaks(_dataset(data), dims=("y", "x"), height=None)
    assert [(row["y_index"], row["x_index"]) for row in table.to_dict()] == [(2, 2)]


def test_2d_distance_none_zero_and_rectangular_suppression():
    data = np.zeros((12, 12))
    data[3, 3] = 10
    data[5, 5] = 9
    data[5, 10] = 8
    dataset = _dataset(data)

    no_suppression, _ = find_peaks(dataset, dims=("y", "x"), distance=None)
    assert [(r["y_index"], r["x_index"]) for r in no_suppression.to_dict()] == [
        (3, 3),
        (5, 5),
        (5, 10),
    ]

    suppressed, _ = find_peaks(dataset, dims=("y", "x"), distance={"y": 3, "x": 3})
    assert [(r["y_index"], r["x_index"]) for r in suppressed.to_dict()] == [
        (3, 3),
        (5, 10),
    ]

    zero_axis, _ = find_peaks(dataset, dims=("y", "x"), distance={"y": 0, "x": 3})
    assert [(r["y_index"], r["x_index"]) for r in zero_axis.to_dict()] == [
        (3, 3),
        (5, 10),
    ]


def test_2d_distance_boundary_is_inclusive_and_ties_are_lexicographic():
    data = np.zeros((9, 9))
    data[2, 2] = 5
    data[2, 5] = 5
    dataset = _dataset(data)
    table, _ = find_peaks(dataset, dims=("y", "x"), distance=(1, 3))
    assert [(row["y_index"], row["x_index"]) for row in table.to_dict()] == [
        (2, 2),
        (2, 5),
    ]

    close = np.zeros((9, 9))
    close[2, 2] = 5
    close[2, 4] = 5
    table, _ = find_peaks(_dataset(close), dims=("y", "x"), distance=(1, 3))
    assert [(row["y_index"], row["x_index"]) for row in table.to_dict()] == [(2, 2)]


def test_2d_plateau_returns_lexicographic_representative_and_boundary_maximum():
    data = np.zeros((8, 8))
    data[0, 2:4] = 4
    data[5:7, 5:7] = 3
    table, _ = find_peaks(_dataset(data), dims=("y", "x"), height=1)
    assert [(row["y_index"], row["x_index"]) for row in table.to_dict()] == [
        (0, 2),
        (5, 5),
    ]


def test_2d_masks_nan_and_infinities_are_excluded_without_mutation():
    data = np.zeros((9, 9), dtype=np.float32)
    data[2, 2] = 5
    data[2, 6] = 4
    data[6, 6] = np.nan
    data[0, 0] = np.inf
    data[0, 1] = -np.inf
    mask = np.zeros_like(data, dtype=bool)
    mask[4, 4] = True
    dataset = _dataset(data, mask=mask, dtype=np.float32)
    original = dataset.data.copy()
    original_mask = np.array(dataset.mask, copy=True)
    original_dtype = dataset.dtype

    result, _ = find_peaks(dataset, dims=("y", "x"), height=1)

    assert [(row["y_index"], row["x_index"]) for row in result.to_dict()] == [
        (2, 2),
        (2, 6),
    ]
    assert dataset.dtype == original_dtype
    np.testing.assert_array_equal(dataset.data, original)
    np.testing.assert_array_equal(dataset.mask, original_mask)


def test_2d_masked_plateau_is_split_into_valid_components():
    data = np.zeros((8, 8))
    data[3:5, 2:6] = 4
    mask = np.zeros_like(data, dtype=bool)
    mask[3:5, 3] = True
    table, _ = find_peaks(_dataset(data, mask=mask), dims=("y", "x"), height=1)
    assert [(row["y_index"], row["x_index"]) for row in table.to_dict()] == [
        (3, 2),
        (3, 4),
    ]


def test_2d_coordinates_preserve_source_axes_when_dims_are_reversed():
    dataset = _two_peaks(reverse_y=True)
    normal = find_peaks(dataset, dims=("y", "x"), height=0.2)[0].to_dict()
    reversed_dims = find_peaks(dataset, dims=("x", "y"), height=0.2)[0].to_dict()

    assert [(row["y_index"], row["x_index"]) for row in normal] == [
        (7, 6),
        (23, 18),
    ]
    assert [(row["x_index"], row["y_index"]) for row in reversed_dims] == [
        (6, 7),
        (18, 23),
    ]
    assert reversed_dims[0]["y_position"].magnitude == pytest.approx(
        dataset.coord("y").values[7].magnitude
    )


def test_2d_physical_distances_support_different_units_and_irregular_axes():
    y_values = np.array([0.0, 0.2, 0.7, 1.4, 2.4, 3.7, 5.3, 7.2, 9.4])
    x_values = np.array([0.0, 0.5, 1.2, 2.0, 3.0, 4.2, 5.5, 7.0, 8.7])
    data = np.zeros((9, 9))
    data[2, 2] = 5
    data[5, 5] = 4
    dataset = NDDataset(
        data,
        coordset=[
            Coord(y_values, title="y", units="cm^-1"),
            Coord(x_values, title="x", units="nm"),
        ],
        units="absorbance",
    )
    table, _ = find_peaks(
        dataset,
        dims=("y", "x"),
        distance={"y": 2.0 * ur("cm^-1"), "x": 2.0 * ur.nm},
    )
    assert [(row["y_index"], row["x_index"]) for row in table.to_dict()] == [
        (2, 2),
        (5, 5),
    ]
    assert table.to_dict()[0]["y_position"].units == ur("cm^-1")
    assert table.to_dict()[0]["x_position"].units == ur.nm


def test_2d_irregular_physical_distance_uses_coordinate_differences_directly():
    y_values = np.array([0.0, 0.1, 0.2, 4.0, 4.1, 4.2, 9.0])
    x_values = np.arange(7.0)
    data = np.zeros((7, 7))
    data[1, 1] = 5
    data[4, 1] = 4
    dataset = NDDataset(
        data,
        coordset=[
            Coord(y_values, title="y", units="ppm"),
            Coord(x_values, title="x", units="nm"),
        ],
    )
    table, _ = find_peaks(
        dataset,
        dims=("y", "x"),
        distance={"y": 2.0 * ur.ppm, "x": 0 * ur.nm},
    )
    assert [(row["y_index"], row["x_index"]) for row in table.to_dict()] == [
        (1, 1),
        (4, 1),
    ]


def test_2d_height_units_are_converted_and_validated():
    data = np.zeros((7, 7))
    data[3, 3] = 0.5
    dataset = _dataset(data, units="absorbance")
    table, _ = find_peaks(dataset, dims=("y", "x"), height=0.4 * ur.absorbance)
    assert len(table) == 1
    with pytest.raises(ValueError, match="Height units"):
        find_peaks(dataset, dims=("y", "x"), height=0.4 * ur.seconds)
    with pytest.raises(ValueError, match="requires dataset units"):
        find_peaks(
            _dataset(data, units=None), dims=("y", "x"), height=0.4 * ur.absorbance
        )


@pytest.mark.parametrize(
    "distance",
    [
        2,
        -1,
        np.nan,
        np.inf,
        {"y": 1},
        {"y": 1, "x": 2, "z": 3},
        {"y": 1, "x": 1 * ur.ppm},
        (1, 2, 3),
    ],
)
def test_2d_invalid_distances_are_rejected(distance):
    with pytest.raises(ValueError):
        find_peaks(_dataset(np.zeros((5, 5))), dims=("y", "x"), distance=distance)


@pytest.mark.parametrize("height", [0, -1, np.nan, np.inf, True, "0.5"])
def test_2d_invalid_heights_are_rejected(height):
    with pytest.raises(ValueError):
        find_peaks(_dataset(np.zeros((5, 5))), dims=("y", "x"), height=height)


@pytest.mark.parametrize(
    ("dims", "error"),
    [
        (("x",), ValueError),
        (("x", "x"), ValueError),
        (("x", 1), ValueError),
        (("missing", "x"), ValueError),
        ((0, 2), ValueError),
        ("xy", TypeError),
        (("x", None), TypeError),
        ((0, 1.5), TypeError),
        ((True, 1), TypeError),
    ],
)
def test_2d_invalid_dims_are_rejected(dims, error):
    with pytest.raises(error):
        find_peaks(_dataset(np.zeros((5, 5))), dims=dims)


def test_2d_nonmonotonic_and_unitless_physical_coordinates_are_rejected():
    data = np.zeros((5, 5))
    nonmonotonic = NDDataset(
        data,
        coordset=[Coord([0, 1, 2, 1, 4], title="y"), Coord(np.arange(5), title="x")],
    )
    with pytest.raises(ValueError, match="strictly monotonic"):
        find_peaks(nonmonotonic, dims=("y", "x"))

    unitless = _dataset(data)
    with pytest.raises(ValueError, match="requires coordinate units"):
        find_peaks(
            unitless,
            dims=("y", "x"),
            distance={"y": 1 * ur.ppm, "x": 1 * ur.ppm},
        )


def test_2d_invalid_data_types_are_rejected_but_1d_complex_is_unchanged():
    complex_dataset = _dataset(np.ones((5, 5), dtype=complex))
    with pytest.raises(ValueError, match="real numeric"):
        find_peaks(complex_dataset, dims=("y", "x"))

    one_d = NDDataset(np.array([0 + 0j, 1 + 2j, 0 + 0j]))
    peaks, _ = find_peaks(one_d)
    assert len(peaks) == 1
    assert peaks.data[0] == pytest.approx(1.0)


def test_2d_unsupported_parameters_and_dataset_dimensionality_are_explicit():
    dataset = _dataset(np.zeros((5, 5)))
    with pytest.raises(NotImplementedError, match="threshold"):
        find_peaks(dataset, dims=("y", "x"), threshold=0.1)
    with pytest.raises(NotImplementedError, match="rel_height"):
        find_peaks(dataset, dims=("y", "x"), rel_height=0.4)
    with pytest.raises(NotImplementedError, match="interpolate"):
        find_peaks(dataset, dims=("y", "x"), window_length=5)
    with pytest.raises(ValueError, match="one or two dimensions"):
        find_peaks(NDDataset(np.zeros((3, 4, 5))), dims=("x", "y", "z"))


def test_explicit_1d_dims_keep_historical_engine_and_invalid_dims_fail():
    dataset = NDDataset([0.0, 1.0, 0.0], coordset=[Coord([0.0, 1.0, 2.0], title="x")])
    old_result = find_peaks(dataset)
    explicit_result = find_peaks(dataset, dims=("x",))
    np.testing.assert_array_equal(old_result[0].data, explicit_result[0].data)
    assert old_result[1] == explicit_result[1]
    with pytest.raises(ValueError):
        find_peaks(dataset, dims=("y",))


def test_2d_history_is_reproducible_detached_and_export_methods_work(tmp_path):
    dataset = _two_peaks()
    source_history = dataset.history_entries
    result = find_peaks(
        dataset,
        dims=("x", "y"),
        height=0.2,
        distance=None,
        as_result=True,
    )
    table = result.table
    entry = table.history_entries[-1]
    assert entry["operation"] == "find_peaks_2d"
    assert entry["parameters"]["axes"] == [1, 0]
    assert entry["parameters"]["dims_requested"] == ["x", "y"]
    assert entry["parameters"]["dims"] == ["x", "y"]
    assert entry["parameters"]["use_coord"] is True
    assert entry["parameters"]["distance"] == {
        "mode": "none",
        "requested": None,
        "effective": None,
    }
    assert entry["parameters"]["n_peaks"] == 2
    json.dumps(entry["parameters"])
    assert "find_peaks (2D)" in table.history[-1]
    original = copy.deepcopy(table.history_entries)
    detached = table.history_entries
    detached[-1]["parameters"]["dims"].append("bad")
    assert table.history_entries == original
    assert dataset.history_entries == source_history

    assert table.head(1).columns == table.columns
    assert table.top(1, by="height").columns == table.columns
    path = table.to_csv(tmp_path / "peaks.csv")
    assert path.read_text(encoding="utf-8").splitlines()[0] == (
        "index,x_index,y_index,x_position,y_position,height"
    )
    result_path = result.to_csv(tmp_path / "result.csv")
    assert result_path.read_text(encoding="utf-8").startswith("index,x_index,y_index")


def test_2d_unit_aware_mapping_rejects_incompatible_units_and_missing_coords():
    dataset = _two_peaks()
    with pytest.raises(ValueError, match="incompatible coordinate units"):
        find_peaks(
            dataset,
            dims=("y", "x"),
            distance={"y": 1 * ur.seconds, "x": 1 * ur.ppm},
        )

    no_coord_data = np.zeros((5, 5))
    no_coord_data[2, 3] = 1
    no_coords = NDDataset(no_coord_data)
    with pytest.raises(ValueError, match="requires coordinates"):
        find_peaks(no_coords, dims=("y", "x"), use_coord=True)

    point_table, _ = find_peaks(
        no_coords, dims=("y", "x"), use_coord=False, distance=(1, 1)
    )
    assert point_table.to_dict()[0]["y_position"] == 2
    assert point_table.to_dict()[0]["x_position"] == 3


def test_2d_threshold_and_unused_legacy_parameters_are_rejected():
    dataset = _dataset(np.eye(5))
    for parameter in ("prominence", "width", "wlen", "plateau_size"):
        with pytest.raises(NotImplementedError, match=parameter):
            find_peaks(dataset, dims=("y", "x"), **{parameter: 1})

    with pytest.raises(ValueError, match="height"):
        find_peaks(dataset, dims=("y", "x"), height=(0.1, 1.0))


def test_2d_index_dimensions_and_distance_sequence_follow_requested_order():
    dataset = _two_peaks()
    by_names = find_peaks(dataset, dims=("y", "x"), height=0.2)[0].to_dict()
    by_indices = find_peaks(dataset, dims=(0, 1), height=0.2)[0].to_dict()
    assert by_names == by_indices
    reversed_result = find_peaks(dataset, dims=(1, 0), height=0.2, distance=(0, 0))[
        0
    ].to_dict()
    assert [(row["x_index"], row["y_index"]) for row in reversed_result] == [
        (6, 7),
        (18, 23),
    ]


def test_2d_history_records_index_dims_and_point_coordinates_mode():
    dataset = _two_peaks()
    result = find_peaks(
        dataset,
        dims=(1, 0),
        use_coord=False,
        distance=(2, 3),
        as_result=True,
    )

    parameters = result.history_entries[-1]["parameters"]
    assert parameters["dims_requested"] == [1, 0]
    assert parameters["dims"] == ["x", "y"]
    assert parameters["axes"] == [1, 0]
    assert parameters["use_coord"] is False
    assert parameters["distance"]["mode"] == "points"
