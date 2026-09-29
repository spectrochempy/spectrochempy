"""History regressions for the public procedural SNV wrapper."""

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy import Coord
from spectrochempy import NDDataset
from spectrochempy import SNVTransformer


@pytest.fixture
def history_dataset():
    """Return a masked, metadata-rich source with nested structured history."""
    dataset = NDDataset(
        np.ma.MaskedArray(
            np.array(
                [
                    [2.0, 5.0, 9.0, 3.0],
                    [7.0, 1.0, 4.0, 6.0],
                    [3.0, 12.0, 2.0, 14.0],
                ]
            ),
            mask=np.array(
                [
                    [False, True, False, False],
                    [False, False, False, False],
                    [False, False, True, False],
                ]
            ),
        ),
        dims=["y", "x"],
        coordset=[
            Coord([10.0, 20.0, 30.0], units="s", title="time"),
            Coord([1000.0, 1100.0, 1200.0, 1300.0], units="cm^-1", title="shift"),
        ],
        units="absorbance",
        name="snv_source",
        title="spectra",
    )
    dataset.description = "history-rich SNV source"
    dataset.meta.operator = "tester"
    dataset.annotate("prepared for SNV")
    # The fractional discrete shift records nested scientific/requested
    # parameter mappings. SNV must preserve them exactly and independently.
    return dataset.roll(pts=1.9, dim="x")


def _assert_state(dataset, snapshot):
    np.testing.assert_array_equal(dataset.data, snapshot.data)
    np.testing.assert_array_equal(dataset.mask, snapshot.mask)
    assert dataset.dims == snapshot.dims
    assert dataset.coordset == snapshot.coordset
    assert dataset.units == snapshot.units
    assert dataset.name == snapshot.name
    assert dataset.title == snapshot.title
    assert dataset.description == snapshot.description
    assert dataset.meta == snapshot.meta
    assert dataset.history_entries == snapshot.history_entries


def _assert_scientific_result(result, expected):
    np.testing.assert_allclose(result.data, expected.data)
    np.testing.assert_array_equal(result.mask, expected.mask)
    assert result.dims == expected.dims
    assert result.coordset == expected.coordset
    assert result.units == expected.units
    assert result.title == expected.title
    assert result.description == expected.description
    assert result.meta == expected.meta


@pytest.mark.parametrize("inplace", [False, True])
def test_public_snv_adds_one_transformer_entry_in_both_modes(history_dataset, inplace):
    source = history_dataset
    snapshot = source.copy()
    prior = source.history_entries
    expected = SNVTransformer().fit_transform(source)
    assert np.any(source.mask)

    result = source.snv(inplace=inplace)

    assert (result is source) is inplace
    assert result.history_entries[:-1] == prior
    assert len(result.history_entries) == len(prior) + 1
    assert result.history_entries[-1]["operation"] is None
    assert result.history_entries[-1]["parameters"] == {}
    assert result.history_entries[-1]["message"] == "SNVTransformer applied"
    _assert_scientific_result(result, expected)

    if inplace:
        assert result.name == snapshot.name
    else:
        _assert_state(source, snapshot)


@pytest.mark.parametrize("inplace", [False, True])
def test_public_snv_history_parameters_are_detached(history_dataset, inplace):
    source = history_dataset
    prior = source.history_entries

    result = source.snv(inplace=inplace)
    detached = result.history_entries
    detached[1]["parameters"]["scientific_parameters"]["pts"] = 999
    detached[-1]["parameters"]["invented"] = True

    assert result.history_entries[:-1] == prior
    assert result.history_entries[-1]["parameters"] == {}


@pytest.mark.parametrize("inplace", [False, True])
def test_two_public_snv_calls_append_once_each(history_dataset, inplace):
    source = history_dataset
    prior = source.history_entries

    first = source.snv(inplace=inplace)
    second = first.snv(inplace=inplace)

    assert second.history_entries[:-2] == prior
    assert [entry["message"] for entry in second.history_entries[-2:]] == [
        "SNVTransformer applied",
        "SNVTransformer applied",
    ]


def test_public_snv_failure_does_not_mutate_source():
    source = NDDataset(
        np.array([["a", "b"], ["c", "d"]]),
        dims=["y", "x"],
        coordset=[Coord([0.0, 1.0]), Coord([10.0, 20.0])],
        name="invalid_snv_source",
    )
    source.meta.reason = "non-numeric regression"
    source.annotate("must survive failed SNV")
    snapshot = source.copy()

    with pytest.raises(TypeError, match="resolved dtypes|ufunc"):
        source.snv(inplace=True)

    _assert_state(source, snapshot)


def test_public_snv_history_survives_scp_roundtrip(history_dataset, tmp_path):
    result = history_dataset.snv()

    filename = result.write(tmp_path / "snv-history.scp", overwrite=True)
    rebuilt = scp.read(filename)

    assert rebuilt.history_entries == result.history_entries
    assert rebuilt.history == result.history
