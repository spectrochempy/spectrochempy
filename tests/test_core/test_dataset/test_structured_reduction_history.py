"""Structured history coverage for numerical NDDataset reductions."""

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset


@pytest.fixture
def reduction_dataset():
    data = np.arange(24.0).reshape(2, 3, 4)
    mask = np.zeros_like(data, dtype=bool)
    mask[0, 1, 2] = True
    dataset = NDDataset(
        np.ma.MaskedArray(data, mask=mask),
        dims=["z", "y", "x"],
        units="m",
        name="source",
        title="measurement",
    )
    dataset.set_coordset(
        z=Coord([10.0, 20.0], units="s", title="delay"),
        y=Coord([1.0, 2.0, 3.0], units="K", title="temperature"),
        x=Coord([100.0, 200.0, 300.0, 400.0], units="cm^-1", title="wavenumber"),
    )
    dataset.meta.project = "history coverage"
    dataset.history = "prepared"
    return dataset


def _assert_source_unchanged(dataset, snapshot):
    assert np.array_equal(dataset.data, snapshot.data)
    assert np.array_equal(dataset.mask, snapshot.mask)
    assert dataset.dims == snapshot.dims
    assert dataset.units == snapshot.units
    assert dataset.title == snapshot.title
    assert dataset.name == snapshot.name
    assert dataset.meta == snapshot.meta
    assert dataset.history_entries == snapshot.history_entries
    for dim in dataset.dims:
        assert np.array_equal(dataset.coord(dim).data, snapshot.coord(dim).data)
        assert dataset.coord(dim).units == snapshot.coord(dim).units
        assert dataset.coord(dim).title == snapshot.coord(dim).title


@pytest.mark.parametrize(
    ("operation", "label", "extra"),
    [
        ("mean", "Mean", {}),
        ("sum", "Sum", {}),
        ("std", "Standard deviation", {"ddof": 1}),
        ("var", "Variance", {"ddof": 1}),
    ],
)
def test_numeric_reductions_record_one_structured_entry_and_preserve_science(
    reduction_dataset,
    operation,
    label,
    extra,
):
    source = reduction_dataset
    snapshot = source.copy()

    result = getattr(source, operation)(
        dim=("z", "x"),
        dtype=np.float64,
        **extra,
    )
    expected = getattr(np.ma, operation)(
        source.masked_data,
        axis=(0, 2),
        dtype=np.float64,
        **extra,
    )

    assert isinstance(result, NDDataset)
    assert result.dims == ["y"]
    assert np.allclose(result.data, expected.data)
    assert np.array_equal(
        np.ma.getmaskarray(result.masked_data), np.ma.getmaskarray(expected)
    )
    assert np.array_equal(result.y.data, source.y.data)
    assert result.y.units == source.y.units
    assert result.y.title == source.y.title
    assert result.title == source.title
    assert result.name == source.name
    assert result.meta == source.meta
    assert result.units == (source.units**2 if operation == "var" else source.units)

    assert result.history_entries[:-1] == source.history_entries
    entry = result.history_entries[-1]
    assert entry["operation"] == operation
    expected_parameters = {
        "requested_dims": ["z", "x"],
        "resolved_dims": ["z", "x"],
        "all_dimensions": False,
        "keepdims": False,
        "dtype": "float64",
    }
    if operation in {"std", "var"}:
        expected_parameters["ddof"] = 1
    assert entry["parameters"] == expected_parameters
    assert entry["message"] == f"{label} computed along z and x"
    assert len(result.history_entries) == len(source.history_entries) + 1
    assert "Dataset resulting from application" not in result.history[-1]
    _assert_source_unchanged(source, snapshot)


@pytest.mark.parametrize("operation", ["mean", "sum", "std", "var"])
def test_public_reduction_function_does_not_duplicate_history(
    reduction_dataset,
    operation,
):
    kwargs = {"ddof": 1} if operation in {"std", "var"} else {}

    result = getattr(scp, operation)(reduction_dataset, dim="x", **kwargs)

    matching = [
        entry for entry in result.history_entries if entry["operation"] == operation
    ]
    assert len(matching) == 1
    assert len(result.history_entries) == len(reduction_dataset.history_entries) + 1


@pytest.mark.parametrize("operation", ["mean", "sum", "std", "var"])
def test_full_keepdims_reduction_records_all_dimensions(
    reduction_dataset,
    operation,
):
    kwargs = {"ddof": 1} if operation in {"std", "var"} else {}

    result = getattr(reduction_dataset, operation)(keepdims=True, **kwargs)

    assert isinstance(result, NDDataset)
    assert result.shape == (1, 1, 1)
    entry = result.history_entries[-1]
    assert entry["operation"] == operation
    assert entry["parameters"]["requested_dims"] is None
    assert entry["parameters"]["resolved_dims"] == ["z", "y", "x"]
    assert entry["parameters"]["all_dimensions"] is True
    assert entry["parameters"]["keepdims"] is True
    assert entry["message"].endswith("computed over all dimensions")


@pytest.mark.parametrize("operation", ["mean", "sum", "std", "var"])
def test_scalar_and_failed_reductions_do_not_change_source_history(
    reduction_dataset,
    operation,
):
    before = reduction_dataset.history_entries
    kwargs = {"ddof": 1} if operation in {"std", "var"} else {}

    scalar = getattr(reduction_dataset, operation)(**kwargs)

    assert not isinstance(scalar, NDDataset)
    assert reduction_dataset.history_entries == before
    with pytest.raises(ValueError):
        getattr(reduction_dataset, operation)(dim="not-a-dimension", **kwargs)
    assert reduction_dataset.history_entries == before


def test_reduction_history_parameters_are_detached(reduction_dataset):
    result = reduction_dataset.sum(dim=("z", "x"))
    entries = result.history_entries

    entries[-1]["parameters"]["resolved_dims"].append("mutated")

    assert result.history_entries[-1]["parameters"]["resolved_dims"] == ["z", "x"]
