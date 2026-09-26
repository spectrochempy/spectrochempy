# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Focused contract tests for the local structured-history prototype."""

from __future__ import annotations

import json
import zipfile
from datetime import UTC
from datetime import datetime

import numpy as np
import pytest
import xarray as xr

import spectrochempy as scp
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.nddataset import _prepare_xarray_dataset_for_netcdf
from spectrochempy.utils.exceptions import SpectroChemPyError
from spectrochempy.utils.jsonutils import json_loads


def _dataset(name="left"):
    return NDDataset(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        coordset=[
            Coord([10.0, 20.0], title="rows"),
            Coord([100.0, 200.0], title="columns"),
        ],
        mask=np.array([[False, True], [False, False]]),
        name=name,
    )


def test_annotation_has_readable_and_detached_structured_views():
    dataset = _dataset()
    recorded_at = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)

    dataset.annotate("sample loaded", date=recorded_at)
    entries = dataset.history_entries
    entries[0]["message"] = "changed elsewhere"

    assert len(dataset.history) == 1
    assert dataset.history[0].endswith("> Sample loaded")
    assert dataset.history_entries == [
        {
            "date": recorded_at,
            "operation": None,
            "parameters": {},
            "message": "sample loaded",
        }
    ]


def test_replace_and_clear_are_explicit_and_list_assignment_keeps_all_entries():
    dataset = _dataset()
    dataset.annotate("discarded")

    dataset.history = ["first replacement", "second replacement"]

    assert [entry["message"] for entry in dataset.history_entries] == [
        "first replacement",
        "second replacement",
    ]
    dataset.clear_history()
    assert dataset.history == []


def test_transpose_records_parameters_only_after_success():
    dataset = _dataset()

    transposed = dataset.transpose("x", "y")

    assert np.array_equal(transposed.data, dataset.data.T)
    assert np.array_equal(transposed.mask, dataset.mask.T)
    assert transposed.units == dataset.units
    assert np.array_equal(transposed.coord("x").data, dataset.coord("x").data)
    assert np.array_equal(transposed.coord("y").data, dataset.coord("y").data)
    entry = transposed.history_entries[-1]
    assert entry["operation"] == "transpose"
    assert entry["parameters"] == {
        "requested_dims": ["x", "y"],
        "result_dims": ["x", "y"],
        "inplace": False,
    }
    assert entry["message"] == "Data transposed between dims: ('x', 'y')"

    before = dataset.history_entries
    with pytest.raises(ValueError):
        dataset.transpose("not-a-dimension")
    assert dataset.history_entries == before


def test_scalar_addition_preserves_science_and_records_detached_sources():
    dataset = _dataset()

    result = dataset + 2

    valid = ~dataset.mask
    assert np.array_equal(result.data[valid], (dataset.data + 2)[valid])
    assert np.array_equal(result.mask, dataset.mask)
    assert result.units == dataset.units
    assert tuple(result.dims) == tuple(dataset.dims)
    assert np.array_equal(result.coord("x").data, dataset.coord("x").data)
    assert np.array_equal(result.coord("y").data, dataset.coord("y").data)
    entry = result.history_entries[-1]
    assert entry["operation"] == "add"
    assert entry["message"] == "Binary operation add with `2` has been performed"
    assert entry["parameters"]["sources"] == [
        {
            "role": "left",
            "kind": "NDDataset",
            "name": "left",
            "title": None,
            "shape": [2, 2],
        },
        {"role": "right", "kind": "scalar", "value": 2},
    ]


def test_dataset_addition_names_both_sources_without_merging_chronologies():
    left = _dataset("left")
    right = _dataset("right")
    left.annotate("left preparation")
    right.annotate("right preparation")

    result = left + right

    valid = ~left.mask
    assert np.array_equal(result.data[valid], (left.data + right.data)[valid])
    assert [entry["message"] for entry in result.history_entries[:-1]] == [
        "left preparation"
    ]
    assert "right preparation" not in [
        entry["message"] for entry in result.history_entries
    ]
    sources = result.history_entries[-1]["parameters"]["sources"]
    assert [(source["role"], source["name"]) for source in sources] == [
        ("left", "left"),
        ("right", "right"),
    ]


def test_failed_addition_does_not_change_source_history():
    dataset = _dataset()
    before = dataset.history_entries

    with pytest.raises((TypeError, ValueError)):
        dataset + object()

    assert dataset.history_entries == before


def test_slice_preserves_science_and_describes_requested_coordinates():
    dataset = _dataset()
    dataset.units = "m"
    dataset.annotate("prepared")
    before = dataset.history_entries

    result = dataset[:1, 100.0:200.0]

    assert result.shape == (1, 2)
    assert np.array_equal(result.data, dataset.data[:1])
    assert np.array_equal(result.mask, dataset.mask[:1])
    assert result.units == dataset.units
    assert np.array_equal(result.coord("x").data, dataset.coord("x").data)
    assert np.array_equal(result.coord("y").data, dataset.coord("y").data[:1])
    assert dataset.history_entries == before
    assert result.history_entries[:-1] == before
    entry = result.history_entries[-1]
    assert entry["operation"] == "slice"
    assert entry["message"] == (
        "Slice extracted: y indices [:1], x coordinates [100.0:200.0]"
    )
    assert entry["parameters"] == {
        "source_dims": ["y", "x"],
        "requested": [
            {
                "dimension": "y",
                "kind": "index_slice",
                "start": None,
                "stop": 1,
                "step": None,
            },
            {
                "dimension": "x",
                "kind": "coordinate_slice",
                "start": 100.0,
                "stop": 200.0,
                "step": None,
            },
        ],
        "inplace": False,
    }


def test_slice_parameters_are_detached_and_large_index_lists_are_not_stored():
    dataset = NDDataset(np.arange(20), name="series")
    requested = list(range(17))

    result = dataset[requested]
    requested[0] = 19

    selector = result.history_entries[-1]["parameters"]["requested"][0]
    assert selector == {
        "dimension": "x",
        "kind": "indices",
        "shape": [17],
        "values": "not stored",
    }


def test_single_row_slice_uses_the_dimension_routed_by_indexing():
    dataset = NDDataset(
        np.arange(6).reshape(1, 6),
        coordset=[Coord([0.0]), Coord([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])],
    )

    result = dataset[1:3]

    assert np.array_equal(result.data, dataset.data[:, 1:3])
    selector = result.history_entries[-1]["parameters"]["requested"][0]
    assert selector["dimension"] == "x"
    assert result.history_entries[-1]["message"] == "Slice extracted: x indices [1:3]"


def test_float_fancy_selection_is_described_as_coordinate_values():
    dataset = NDDataset(
        np.arange(6),
        coordset=[Coord([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])],
    )

    result = dataset[[10.0, 30.0]]

    assert np.array_equal(result.data, dataset.data[[1, 3]])
    selector = result.history_entries[-1]["parameters"]["requested"][0]
    assert selector == {
        "dimension": "x",
        "kind": "coordinate_values",
        "shape": [2],
        "values": [10.0, 30.0],
    }
    assert result.history_entries[-1]["message"] == (
        "Slice extracted: x coordinates [10.0, 30.0]"
    )


def test_open_ended_slice_keeps_its_trailing_colon():
    dataset = NDDataset(np.arange(6))

    result = dataset[2:]

    assert result.history_entries[-1]["message"] == "Slice extracted: x indices [2:]"


def test_failed_slice_does_not_change_source_history():
    dataset = _dataset()
    before = dataset.history_entries

    with pytest.raises((IndexError, TypeError)):
        dataset[object()]

    assert dataset.history_entries == before


def test_scalar_subtraction_preserves_science_and_source_history():
    dataset = _dataset()
    dataset.units = "m"
    dataset.annotate("prepared")
    before = dataset.history_entries

    result = dataset - 2

    valid = ~dataset.mask
    assert np.array_equal(result.data[valid], (dataset.data - 2)[valid])
    assert np.array_equal(result.mask, dataset.mask)
    assert result.units == dataset.units
    assert tuple(result.dims) == tuple(dataset.dims)
    assert np.array_equal(result.coord("x").data, dataset.coord("x").data)
    assert np.array_equal(result.coord("y").data, dataset.coord("y").data)
    assert dataset.history_entries == before
    assert result.history_entries[:-1] == before
    entry = result.history_entries[-1]
    assert entry["operation"] == "subtract"
    assert entry["message"] == "Subtracted `2` from `left`"
    assert entry["parameters"]["sources"] == [
        {
            "role": "left",
            "kind": "NDDataset",
            "name": "left",
            "title": None,
            "shape": [2, 2],
        },
        {"role": "right", "kind": "scalar", "value": 2},
    ]


def test_reflected_subtraction_records_mathematical_operand_order():
    dataset = _dataset()

    result = 10 - dataset

    valid = ~dataset.mask
    assert np.array_equal(result.data[valid], (10 - dataset.data)[valid])
    entry = result.history_entries[-1]
    assert entry["operation"] == "subtract"
    assert entry["message"] == "Subtracted `left` from `10`"
    assert entry["parameters"]["sources"] == [
        {"role": "left", "kind": "scalar", "value": 10},
        {
            "role": "right",
            "kind": "NDDataset",
            "name": "left",
            "title": None,
            "shape": [2, 2],
        },
    ]


def test_dataset_subtraction_does_not_merge_or_retain_right_source():
    left = _dataset("left")
    right = _dataset("right")
    left.annotate("left preparation")
    right.annotate("right preparation")

    result = left - right
    right.name = "renamed later"

    assert [entry["message"] for entry in result.history_entries[:-1]] == [
        "left preparation"
    ]
    assert "right preparation" not in [
        entry["message"] for entry in result.history_entries
    ]
    sources = result.history_entries[-1]["parameters"]["sources"]
    assert [(source["role"], source["name"]) for source in sources] == [
        ("left", "left"),
        ("right", "right"),
    ]


def test_failed_subtraction_does_not_change_source_history():
    dataset = _dataset()
    before = dataset.history_entries

    with pytest.raises((TypeError, ValueError)):
        dataset - object()

    assert dataset.history_entries == before


def test_mean_named_dimension_preserves_science_and_records_exact_entry():
    dataset = _dataset()
    dataset.units = "m"
    dataset.annotate("prepared")
    before = dataset.history_entries

    result = dataset.mean(dim="y")
    expected = np.ma.mean(dataset.masked_data, axis=0)

    np.testing.assert_allclose(result.data, expected.data)
    assert np.array_equal(
        np.broadcast_to(result.mask, result.shape), np.ma.getmaskarray(expected)
    )
    assert result.units == dataset.units
    assert list(result.dims) == ["x"]
    assert np.array_equal(result.coord("x").data, dataset.coord("x").data)
    assert dataset.history_entries == before
    assert result.history_entries[:-1] == before
    entry = result.history_entries[-1]
    assert entry["operation"] == "mean"
    assert entry["parameters"] == {
        "requested_dims": ["y"],
        "resolved_dims": ["y"],
        "all_dimensions": False,
        "keepdims": False,
    }
    assert entry["message"] == "Mean computed along y"


@pytest.mark.parametrize("form", ["instance", "class", "api", "numpy"])
def test_mean_public_forms_append_one_entry_with_resolved_numeric_axis(form):
    dataset = _dataset()
    dataset.annotate("prepared")

    if form == "instance":
        result = dataset.mean(dim=0)
    elif form == "class":
        result = NDDataset.mean(dataset, dim=0)
    elif form == "api":
        result = scp.mean(dataset, dim=0)
    else:
        result = np.mean(dataset, axis=0)

    assert len(result.history_entries) == len(dataset.history_entries) + 1
    mean_entries = [
        entry for entry in result.history_entries if entry["operation"] == "mean"
    ]
    assert len(mean_entries) == 1
    assert mean_entries[0]["parameters"] == {
        "requested_dims": [0],
        "resolved_dims": ["y"],
        "all_dimensions": False,
        "keepdims": False,
    }
    assert mean_entries[0]["message"] == "Mean computed along y"


def test_mean_default_scalar_preserves_source_without_artificial_history():
    dataset = _dataset()
    dataset.annotate("prepared")
    before = dataset.history_entries

    result = dataset.mean()

    assert not isinstance(result, NDDataset)
    assert np.isclose(result, np.ma.mean(dataset.masked_data))
    assert dataset.history_entries == before


def test_mean_keepdims_records_all_dimensions_explicitly():
    dataset = _dataset()

    result = dataset.mean(keepdims=True)

    assert result.shape == (1, 1)
    assert list(result.dims) == ["y", "x"]
    entry = result.history_entries[-1]
    assert entry["operation"] == "mean"
    assert entry["parameters"] == {
        "requested_dims": None,
        "resolved_dims": ["y", "x"],
        "all_dimensions": True,
        "keepdims": True,
    }
    assert entry["message"] == "Mean computed over all dimensions"


def test_mean_multiple_dimensions_records_request_and_resolution():
    dataset = _dataset()

    result = dataset.mean(dim=("x", "y"), keepdims=True)

    entry = result.history_entries[-1]
    assert entry["parameters"] == {
        "requested_dims": ["x", "y"],
        "resolved_dims": ["x", "y"],
        "all_dimensions": True,
        "keepdims": True,
    }
    assert entry["message"] == "Mean computed over all dimensions"


def test_mean_multiple_dimensions_uses_readable_partial_reduction_message():
    dataset = NDDataset(np.arange(24).reshape(2, 3, 4))

    result = dataset.mean(dim=("z", "x"))

    assert result.shape == (3,)
    assert list(result.dims) == ["y"]
    entry = result.history_entries[-1]
    assert entry["parameters"] == {
        "requested_dims": ["z", "x"],
        "resolved_dims": ["z", "x"],
        "all_dimensions": False,
        "keepdims": False,
    }
    assert entry["message"] == "Mean computed along z and x"


def test_failed_mean_does_not_change_source_history():
    dataset = _dataset()
    dataset.annotate("prepared")
    before = dataset.history_entries

    with pytest.raises(ValueError, match="not recognized"):
        dataset.mean(dim="z")

    assert dataset.history_entries == before


def test_mean_entry_survives_copy_and_native_roundtrip():
    result = _dataset().mean(dim="y")

    copied = result.copy()
    rebuilt = NDDataset.loads(json_loads(result.dumps()))

    assert copied.history_entries == result.history_entries
    assert rebuilt.history_entries == result.history_entries
    assert rebuilt.history == result.history


@pytest.mark.parametrize("deep", [False, True])
def test_copy_history_is_independent_even_for_shallow_copy(deep):
    dataset = _dataset()
    dataset._append_history_entry(
        operation="example",
        parameters={"nested": {"values": [1, 2]}},
        message="example",
    )

    detached = dataset.history_entries
    detached[0]["parameters"]["nested"]["values"].append(3)
    copied = dataset.copy(deep=deep)
    copied._history[0]["parameters"]["nested"]["values"].append(4)

    assert dataset.history_entries[0]["parameters"]["nested"]["values"] == [1, 2]


def test_inplace_addition_and_subtraction_remain_text_only():
    dataset = _dataset()

    dataset += 2
    addition = dataset.history_entries[-1]
    dataset -= 1
    subtraction = dataset.history_entries[-1]

    assert addition["operation"] is None
    assert addition["parameters"] == {}
    assert "iadd" in addition["message"]
    assert subtraction["operation"] is None
    assert subtraction["parameters"] == {}
    assert "isub" in subtraction["message"]


def test_failed_inplace_operation_does_not_append_success_entry():
    dataset = _dataset()
    before = dataset.history_entries

    with pytest.raises((TypeError, ValueError)):
        dataset += object()

    assert dataset.history_entries == before


def test_unsupported_parameter_is_described_without_retaining_value():
    dataset = _dataset()
    array = np.ones((2, 3))

    dataset._append_history_entry(
        operation="example",
        parameters={"payload": array},
        message="example",
    )

    assert dataset.history_entries[-1]["parameters"]["payload"] == {
        "description": "ndarray(shape=(2, 3), dtype=float64)"
    }


def test_native_roundtrip_preserves_structured_entries():
    dataset = (_dataset()[:1] - 2).T

    rebuilt = NDDataset.loads(json_loads(dataset.dumps()))

    assert rebuilt.history_entries == dataset.history_entries
    assert rebuilt.history == dataset.history


def test_native_loader_accepts_legacy_rendered_history_without_inventing_structure():
    legacy = "2024-01-02 03:04:05+00:00> Legacy processing message"

    rebuilt = NDDataset.loads({"history": [legacy]})

    assert rebuilt.history_entries == [
        {
            "date": datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
            "operation": None,
            "parameters": {},
            "message": "Legacy processing message",
        }
    ]
    assert rebuilt.history[0].endswith("> Legacy processing message")


def test_native_loader_accepts_version_2_with_textual_history(tmp_path):
    dataset = _dataset()
    dataset.annotate(
        "legacy native entry",
        date=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    filename = dataset.save_as(tmp_path / "legacy-history-v2", confirm=False)
    with zipfile.ZipFile(filename, "r") as archive:
        member = archive.namelist()[0]
        payload = json.loads(archive.read(member).decode("utf-8"))
    assert payload["__format__"] == "scp"
    assert payload["__version__"] == 3
    payload["__version__"] = 2
    payload["history"] = dataset.history
    with zipfile.ZipFile(filename, "w") as archive:
        archive.writestr(member, json.dumps(payload))

    rebuilt = NDDataset.load(filename)

    assert rebuilt.history == dataset.history
    assert rebuilt.history_entries[0]["operation"] is None
    assert rebuilt.history_entries[0]["parameters"] == {}


def test_xarray_loader_accepts_version_1_with_textual_history():
    dataset = _dataset("spectra")
    dataset.annotate(
        "legacy portable entry",
        date=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    portable = dataset.to_xarray()
    portable.attrs["scpy_version"] = 1
    portable.attrs["scpy_history"] = dataset.history

    rebuilt = NDDataset.from_xarray(portable)

    assert rebuilt.history == dataset.history
    assert rebuilt.history_entries[0]["operation"] is None
    assert rebuilt.history_entries[0]["parameters"] == {}


def test_netcdf_loader_accepts_version_1_with_textual_history(tmp_path):
    dataset = _dataset("spectra")
    dataset.annotate(
        "legacy NetCDF entry",
        date=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    portable = dataset.to_xarray()
    portable.attrs["scpy_version"] = 1
    portable.attrs["scpy_history"] = dataset.history
    filename = tmp_path / "legacy-history-v1.nc"
    _prepare_xarray_dataset_for_netcdf(portable).to_netcdf(filename, engine="scipy")

    rebuilt = NDDataset.from_netcdf(filename)

    assert rebuilt.history == dataset.history
    assert rebuilt.history_entries[0]["operation"] is None
    assert rebuilt.history_entries[0]["parameters"] == {}


def test_xarray_loader_rejects_unknown_format_version():
    portable = _dataset("spectra").to_xarray()
    portable.attrs["scpy_version"] = 999

    with pytest.raises(
        SpectroChemPyError, match="Unsupported NDDataset xarray format version"
    ):
        NDDataset.from_xarray(portable)


def test_xarray_and_netcdf_roundtrips_preserve_structured_entries(tmp_path):
    dataset = (_dataset("spectra") + 2).T
    dataset.annotate(
        "precision retained",
        date=datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=UTC),
    )

    portable = dataset.to_xarray()
    assert portable.attrs["scpy_format"] == "nddataset-xarray"
    assert portable.attrs["scpy_version"] == 2
    from_xarray = NDDataset.from_xarray(portable)
    filename = tmp_path / "history.nc"
    dataset.to_netcdf(filename)
    from_netcdf = NDDataset.from_netcdf(filename)
    with xr.open_dataset(filename) as stored:
        assert stored.attrs["scpy_format"] == "nddataset-xarray"
        assert stored.attrs["scpy_version"] == 2

    assert from_xarray.history_entries == dataset.history_entries
    assert from_netcdf.history_entries == dataset.history_entries
