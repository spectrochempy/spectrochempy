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


@pytest.mark.parametrize("deep", [False, True])
def test_copy_history_is_independent_even_for_shallow_copy(deep):
    dataset = _dataset()
    dataset._append_history_entry(
        operation="example",
        parameters={"nested": {"values": [1, 2]}},
        message="example",
    )

    copied = dataset.copy(deep=deep)
    copied._history[0]["parameters"]["nested"]["values"].append(3)

    assert dataset.history_entries[0]["parameters"]["nested"]["values"] == [1, 2]


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
    dataset = (_dataset() + 2).T

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

    from_xarray = NDDataset.from_xarray(dataset.to_xarray())
    filename = tmp_path / "history.nc"
    dataset.to_netcdf(filename)
    from_netcdf = NDDataset.from_netcdf(filename)

    assert from_xarray.history_entries == dataset.history_entries
    assert from_netcdf.history_entries == dataset.history_entries
