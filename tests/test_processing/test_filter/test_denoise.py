# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""History regression tests for denoising operations."""

import numpy as np

import spectrochempy as scp


def test_despike_appends_history_to_returned_dataset():
    data = np.tile(np.linspace(0.0, 1.0, 21), (2, 1))
    data[:, 10] += 5.0
    dataset = scp.NDDataset(data, name="spiked spectra")
    dataset.history = "source history"
    source_history = dataset.history_entries

    result = dataset.despike(size=5, delta=2, method="katsumoto")

    assert result is not dataset
    assert dataset.history_entries == source_history
    assert result.history_entries[0] == source_history[0]
    assert result.history_entries[-1]["operation"] is None
    message = result.history_entries[-1]["message"]
    assert message.startswith("Removed spikes using katsumoto")
    assert "size=5" in message
    assert "delta=2" in message


def test_despike_neighbors_preserves_persistent_bands_and_masks():
    """Detect a transient along acquisitions and retain spectral structure."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 0.05, size=(31, 5))
    data[:, 2] += 100
    data[15, 2] += 10
    data[20, 1] += 10
    dataset = scp.NDDataset(data, units="a.u.")
    dataset.x = scp.Coord(np.arange(5) + 450, units="cm^-1")
    dataset.y = scp.Coord(np.arange(31), units="s")
    mask = np.zeros(data.shape, dtype=bool)
    mask[19, 1] = True
    dataset.mask = mask

    result = dataset.despike(method="neighbors", dim="y", delta=8)

    expected = data.copy()
    expected[15, 2] = (data[14, 2] + data[16, 2]) / 2
    np.testing.assert_array_equal(result.data, expected)
    np.testing.assert_array_equal(dataset.data, data)
    np.testing.assert_array_equal(result.mask, dataset.mask)
    assert result.x == dataset.x
    assert result.y == dataset.y
    assert result.units == dataset.units
    assert "dim=y, delta=8" in result.history_entries[-1]["message"]


def test_neighbors_removes_spike_in_flat_trace():
    dataset = scp.NDDataset([100, 100, 100, 100, 1000, 100, 100, 100, 100])
    result = dataset.despike(method="neighbors")
    np.testing.assert_array_equal(result.data, np.full(9, 100))
