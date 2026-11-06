"""Plugins can veto FFT preparation before the caller is modified."""

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.plugins.manager import plugin_manager
from spectrochempy.utils.testing import assert_dataset_equal


def _dataset(inverse):
    data = np.arange(48).reshape(8, 6) + 1j * np.arange(48, 96).reshape(8, 6)
    units = "Hz" if inverse else "s"
    dataset = scp.NDDataset(
        data,
        coordset=[scp.Coord.arange(8, units=units), scp.Coord.arange(6, units=units)],
    )
    dataset.meta.isfreq = [inverse, inverse]
    dataset.meta.td = list(dataset.shape)
    dataset.meta.marker = {"values": [1, 2, 3]}
    return dataset


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("entry", ["method", "functional"])
@pytest.mark.parametrize("size", [None, 2, 16])
@pytest.mark.parametrize("masked", [False, True])
def test_plugin_veto_preserves_inplace_input(monkeypatch, inverse, entry, size, masked):
    dataset = _dataset(inverse)
    if masked:
        dataset[0, 0] = scp.MASKED
    before = dataset.copy()
    calls = []

    def validate(original, *, dim, inv, **kwargs):
        assert original is dataset
        assert dim == "y"
        assert inv is inverse
        assert kwargs["inplace"] is True
        assert kwargs["size"] == size
        assert kwargs["sizeff"] == 2
        assert_dataset_equal(original, before)
        assert original.meta == before.meta
        calls.append(dim)
        raise RuntimeError("fixture veto")

    monkeypatch.setattr(
        plugin_manager.registry,
        "get_handler",
        lambda name: validate if name == "fft.validate_input" else None,
    )
    name = "ifft" if inverse else "fft"
    with pytest.raises(RuntimeError, match="fixture veto"):
        if entry == "method":
            getattr(dataset, name)(axis=0, inplace=True, size=size, sizeff=2)
        else:
            getattr(scp, name)(dataset, axis=0, inplace=True, size=size, sizeff=2)
    assert calls == ["y"]
    assert_dataset_equal(dataset, before)
    assert dataset.dims == before.dims
    assert dataset.meta == before.meta
    assert dataset.history == before.history


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("registered", [False, True])
def test_absent_or_accepting_validator_preserves_numeric_behavior(
    monkeypatch, inverse, inplace, registered
):
    dataset = _dataset(inverse)
    before = dataset.copy()
    calls = []

    def validate(original, *, dim, inv, **kwargs):
        assert original is dataset
        assert dim == "y"
        assert inv is inverse
        calls.append(dim)

    monkeypatch.setattr(
        plugin_manager.registry,
        "get_handler",
        lambda name: validate if registered and name == "fft.validate_input" else None,
    )
    if inverse:
        expected = np.fft.ifft(np.fft.ifftshift(before.data, axes=0), axis=0)
        result = scp.ifft(dataset, dim="y", inplace=inplace)
    else:
        expected = np.fft.fftshift(np.fft.fft(before.data, axis=0), axes=0)
        result = scp.fft(dataset, dim="y", inplace=inplace)
    np.testing.assert_allclose(result.data, expected)
    assert result.dims == before.dims
    assert calls == (["y"] if registered else [])
    if inplace:
        assert result is dataset
    else:
        assert_dataset_equal(dataset, before)
