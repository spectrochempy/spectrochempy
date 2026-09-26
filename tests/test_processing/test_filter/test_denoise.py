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
