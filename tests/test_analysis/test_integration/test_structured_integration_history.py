"""Structured history regressions for public numerical integration."""

import numpy as np
import pytest
import scipy.integrate

import spectrochempy as scp
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset

METHODS = ("trapezoid", "simpson")


@pytest.fixture
def history_dataset():
    """Non-square 3D source with text and nested structured history."""
    dataset = NDDataset(
        np.arange(24.0).reshape(2, 3, 4),
        dims=["z", "y", "x"],
        coordset=[
            Coord([10.0, 20.0], units="s", title="delay"),
            Coord([1.0, 2.0, 4.0], units="K", title="temperature"),
            Coord([100.0, 200.0, 400.0, 800.0], units="cm^-1", title="wavenumber"),
        ],
        units="absorbance",
        name="integration_source",
        title="signal",
    )
    dataset.description = "history-rich integration source"
    dataset.meta.axis_labels = ["delay", "temperature", "wavenumber"]
    dataset.annotate("prepared for integration")
    # A fractional request gives the prior structured entry nested effective
    # and requested parameter mappings. Integration must preserve it exactly.
    return dataset.roll(pts=1.9, dim="x")


def _assert_source_unchanged(source, snapshot):
    np.testing.assert_array_equal(source.data, snapshot.data)
    np.testing.assert_array_equal(source.mask, snapshot.mask)
    assert source.dims == snapshot.dims
    assert source.units == snapshot.units
    assert source.name == snapshot.name
    assert source.title == snapshot.title
    assert source.description == snapshot.description
    assert source.meta == snapshot.meta
    assert source.history_entries == snapshot.history_entries
    for dim in source.dims:
        np.testing.assert_array_equal(source.coord(dim).data, snapshot.coord(dim).data)
        assert source.coord(dim).units == snapshot.coord(dim).units
        assert source.coord(dim).title == snapshot.coord(dim).title


def _independent_reference(source, method_name, axis):
    method = getattr(scipy.integrate, method_name)
    result = method(
        np.asarray(source.data),
        x=np.asarray(source.coord(source.dims[axis]).data),
        axis=axis,
    )
    # SpectroChemPy preserves its displayed-coordinate orientation convention:
    # an internally increasing coordinate marked as reversed changes the sign.
    if source.coord(source.dims[axis]).reversed:
        result *= -1
    return result


@pytest.mark.parametrize("method_name", METHODS)
def test_preserves_prior_entries_and_appends_one_structured_entry(
    history_dataset, method_name
):
    source = history_dataset
    snapshot = source.copy()
    prior = source.history_entries

    result = getattr(source, method_name)(dim="y")

    assert result.history_entries[:-1] == prior
    assert len(result.history_entries) == len(prior) + 1
    entry = result.history_entries[-1]
    assert entry["operation"] == method_name
    assert entry["parameters"] == {
        "requested_dim": "y",
        "resolved_dim": "y",
        "resolved_axis": 1,
    }
    assert entry["message"] == (
        f"Dataset resulting from application of `{method_name}` method"
    )
    np.testing.assert_allclose(
        result.data,
        _independent_reference(source, method_name, axis=1),
    )
    assert result.dims == ["z", "x"]
    assert result.units == source.units * source.y.units
    assert result.meta == source.meta
    _assert_source_unchanged(source, snapshot)

    detached = result.history_entries
    detached[1]["parameters"]["scientific_parameters"]["pts"] = 999
    detached[-1]["parameters"]["resolved_dim"] = "changed"
    assert result.history_entries == [*prior, entry]


SELECTOR_CASES = [
    ((), {}, None, "x", 2),
    (("y",), {}, "y", "y", 1),
    ((), {"dim": "z"}, "z", "z", 0),
    ((), {"dim": 0}, 0, "z", 0),
    ((), {"dim": -1}, -1, "x", 2),
    ((), {"dims": 0}, 0, "z", 0),
    ((), {"axis": -2}, -2, "y", 1),
    ((), {"dims": "y", "dim": "z", "axis": 0}, "y", "y", 1),
    ((), {"dim": None, "axis": 0}, None, "x", 2),
]


@pytest.mark.parametrize("method_name", METHODS)
@pytest.mark.parametrize(
    ("args", "kwargs", "requested", "resolved_dim", "resolved_axis"),
    SELECTOR_CASES,
)
def test_records_requested_and_source_resolved_dimension(
    history_dataset,
    method_name,
    args,
    kwargs,
    requested,
    resolved_dim,
    resolved_axis,
):
    result = getattr(history_dataset, method_name)(*args, **kwargs)

    assert result.history_entries[-1]["parameters"] == {
        "requested_dim": requested,
        "resolved_dim": resolved_dim,
        "resolved_axis": resolved_axis,
    }
    np.testing.assert_allclose(
        result.data,
        _independent_reference(history_dataset, method_name, resolved_axis),
    )


@pytest.mark.parametrize("method_name", METHODS)
def test_dx_is_forwarded_but_not_recorded_as_determining(history_dataset, method_name):
    method = getattr(history_dataset, method_name)

    default = method(dim="x")
    explicit = method(dim="x", dx=12345.0)

    np.testing.assert_array_equal(explicit.data, default.data)
    assert (
        explicit.history_entries[-1]["parameters"]
        == default.history_entries[-1]["parameters"]
    )
    assert "dx" not in explicit.history_entries[-1]["parameters"]


@pytest.mark.parametrize("method_name", METHODS)
def test_one_dimensional_result_remains_scalar_dataset(method_name):
    source = NDDataset(
        np.array([1.0, 2.0, 4.0]),
        coordset=[Coord([0.0, 1.0, 3.0], units="s", title="time")],
        units="V",
    )
    source.annotate("one-dimensional source")
    prior = source.history_entries

    result = getattr(source, method_name)()

    assert isinstance(result, NDDataset)
    assert result.shape == ()
    assert result.dims == []
    assert result.history_entries[:-1] == prior
    assert result.history_entries[-1]["parameters"] == {
        "requested_dim": None,
        "resolved_dim": "x",
        "resolved_axis": 0,
    }


def test_successive_integrations_keep_the_complete_chronology(history_dataset):
    first = history_dataset.trapezoid(dim="x")
    second = first.simpson(dim="y")

    assert second.history_entries[:-2] == history_dataset.history_entries
    assert [entry["operation"] for entry in second.history_entries[-2:]] == [
        "trapezoid",
        "simpson",
    ]
    assert second.history_entries[-2]["parameters"]["resolved_axis"] == 2
    # The second axis is resolved in the already reduced source, whose dims are z, y.
    assert second.history_entries[-1]["parameters"]["resolved_axis"] == 1
    assert second.dims == ["z"]


@pytest.mark.parametrize("method_name", METHODS)
@pytest.mark.parametrize("masked", [False, True])
def test_mask_contract_and_history_continuity(method_name, masked):
    mask = np.array([[False, False, False], [False, masked, False]])
    source = NDDataset(
        np.ma.MaskedArray(
            np.array([[1.0, 2.0, 3.0], [4.0, 200.0, 6.0]]),
            mask=mask,
        ),
        coordset=[Coord([10.0, 20.0], title="y"), Coord([0.0, 1.0, 2.0], title="x")],
    )
    source.annotate("masked-source" if masked else "unmasked-source")

    result = getattr(source, method_name)(dim="x")

    assert result.history_entries[:-1] == source.history_entries
    assert result.history_entries[-1]["operation"] == method_name
    published_mask = np.asarray(result.mask)
    if masked:
        assert bool(published_mask[1])
        assert np.isnan(np.asarray(result.data)[1])
    else:
        assert published_mask.shape == ()
        assert not bool(published_mask)
        assert np.isfinite(np.asarray(result.data)).all()


@pytest.mark.parametrize("method_name", METHODS)
@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"even": "avg"}, TypeError, "even"),
        ({"dim": "unknown"}, ValueError, "not recognized"),
        ({"dim": "y", "unknown_option": True}, TypeError, "unexpected keyword"),
    ],
)
def test_refusals_leave_source_and_history_unchanged(
    history_dataset, method_name, kwargs, exception, message
):
    source = history_dataset
    snapshot = source.copy()

    with pytest.raises(exception, match=message):
        getattr(source, method_name)(**kwargs)

    _assert_source_unchanged(source, snapshot)


def test_structured_integration_history_survives_scp_roundtrip(
    history_dataset, tmp_path
):
    result = history_dataset.simpson(dim="y")

    filename = result.write(tmp_path / "structured-integration.scp", overwrite=True)
    rebuilt = scp.read(filename)

    assert rebuilt.history_entries == result.history_entries
    assert rebuilt.history == result.history
