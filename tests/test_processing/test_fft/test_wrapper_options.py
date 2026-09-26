# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import numpy as np
import pytest

from spectrochempy import Coord
from spectrochempy import NDDataset
from spectrochempy import ur


def _apodization_dataset():
    return NDDataset(
        np.arange(1.0, 16.0).reshape(3, 5),
        coordset=[Coord(np.arange(3.0)), Coord(np.arange(5.0))],
    )


def _phasing_dataset():
    dataset = NDDataset(
        (np.arange(1.0, 16.0).reshape(3, 5) + 1.0j).astype(complex),
        coordset=[
            Coord(np.arange(3.0), units="Hz"),
            Coord(np.arange(5.0), units="Hz"),
        ],
    )
    dataset.meta.phased = [True, True]
    dataset.meta.phc0 = [0 * ur.degree, 0 * ur.degree]
    dataset.meta.phc1 = [0 * ur.degree, 0 * ur.degree]
    dataset.meta.pivot = [0 * ur.Hz, 0 * ur.Hz]
    dataset.meta.exptc = [0 / ur.Hz, 0 / ur.Hz]
    return dataset


@pytest.mark.parametrize(("name", "alpha"), [("hamming", 0.54), ("hann", 0.5)])
def test_hamming_wrappers_forward_dimension_and_preserve_source(name, alpha):
    source = _apodization_dataset()
    source_data = source.data.copy()

    result = getattr(source, name)(dim="y")
    expected = source.general_hamming(alpha=alpha, dim="y")

    np.testing.assert_allclose(result.data, expected.data)
    np.testing.assert_array_equal(source.data, source_data)


def test_hann_forwards_axis_alias():
    source = _apodization_dataset()

    result = source.hann(axis=0)
    expected = source.general_hamming(alpha=0.5, axis=0)

    np.testing.assert_allclose(result.data, expected.data)


@pytest.mark.parametrize(("name", "alpha"), [("hamming", 0.54), ("hann", 0.5)])
def test_hamming_wrappers_forward_inplace_and_retapod(name, alpha):
    source = _apodization_dataset()
    original = source.data.copy()

    result = getattr(source, name)(dim="y", inplace=True)

    assert result is source
    assert not np.array_equal(source.data, original)

    dataset = _apodization_dataset()
    result, window = getattr(dataset, name)(dim="y", retapod=True)
    expected, expected_window = dataset.general_hamming(
        alpha=alpha, dim="y", retapod=True
    )
    np.testing.assert_allclose(result.data, expected.data)
    np.testing.assert_allclose(window.data, expected_window.data)


def test_pk_exp_forwards_dimension_and_preserves_source():
    source = _phasing_dataset()
    source_data = source.data.copy()

    result = source.pk_exp(phc0=30, exptc=1, dim="y")
    expected = source.pk(phc0=30, phc1=0, exptc=1, dim="y")

    np.testing.assert_allclose(result.data, expected.data)
    np.testing.assert_array_equal(source.data, source_data)


def test_pk_exp_forwards_axis_alias_and_inplace():
    axis_source = _phasing_dataset()
    axis_result = axis_source.pk_exp(phc0=30, exptc=1, axis=0)
    axis_expected = axis_source.pk(phc0=30, phc1=0, exptc=1, axis=0)
    np.testing.assert_allclose(axis_result.data, axis_expected.data)

    source = _phasing_dataset()
    original = source.data.copy()
    result = source.pk_exp(phc0=30, exptc=1, dim="y", inplace=True)

    assert result is source
    assert not np.array_equal(source.data, original)


@pytest.mark.parametrize(
    ("name", "alpha", "options"),
    [
        ("hamming", 0.54, {"inv": True}),
        ("hann", 0.5, {"rev": True}),
    ],
)
def test_hamming_wrappers_forward_window_options(name, alpha, options):
    source = _apodization_dataset()

    result = getattr(source, name)(**options)
    expected = source.general_hamming(alpha=alpha, **options)

    np.testing.assert_allclose(result.data, expected.data)


def test_pk_exp_inplace_preserves_buffer_isolation_and_single_history_entry():
    source = _phasing_dataset()
    source.annotate("Synthetic source")
    previous_history = source.history_entries
    data_alias = source.data
    original = data_alias.copy()
    data_alias.flags.writeable = False

    result = source.pk_exp(phc0=30, exptc=1, inplace=True)

    assert result is source
    np.testing.assert_array_equal(data_alias, original)
    assert not np.shares_memory(result.data, data_alias)
    assert result.data.flags.writeable
    assert result.history_entries[:-1] == previous_history
    assert len(result.history_entries) == len(previous_history) + 1
    assert result.history_entries[-1]["message"].startswith(
        "Applied pk phasing on dimension"
    )
