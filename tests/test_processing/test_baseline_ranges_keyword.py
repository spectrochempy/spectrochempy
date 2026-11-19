"""Regression tests for the public ``ranges`` keyword of baseline wrappers."""

import numpy as np
import pytest
from traitlets import TraitError

import spectrochempy as scp


def _curved_dataset():
    coord = scp.Coord(
        np.linspace(4000.0, 1000.0, 120),
        title="Wavenumber",
        units="cm^-1",
    )
    x = coord.data
    distance = 4000.0 - x
    baseline = 0.5 + 3e-7 * distance**2 - 2e-4 * distance
    peak = np.exp(-0.5 * ((x - 1700.0) / 35.0) ** 2)
    return scp.NDDataset(
        baseline + peak,
        coordset=[coord],
        name="curved",
        title="Curved baseline",
        units="absorbance",
    )


@pytest.mark.parametrize("function", ["basc", "get_baseline"])
def test_keyword_ranges_match_positional(function):
    dataset = _curved_dataset()
    ranges = [[3600.0, 3400.0], [1400.0, 1100.0]]

    positional = getattr(scp, function)(dataset, *ranges, order=2)
    keyword = getattr(scp, function)(dataset, ranges=ranges, order=2)

    np.testing.assert_allclose(keyword.data, positional.data)


@pytest.mark.parametrize("function", ["basc", "get_baseline"])
def test_keyword_single_range_matches_positional(function):
    dataset = _curved_dataset()

    positional = getattr(scp, function)(dataset, [3600.0, 3400.0], order=2)
    keyword = getattr(scp, function)(dataset, ranges=[3600.0, 3400.0], order=2)

    np.testing.assert_allclose(keyword.data, positional.data)


def test_basc_empty_keyword_ranges_matches_default():
    dataset = _curved_dataset()

    np.testing.assert_allclose(dataset.basc(ranges=[]).data, dataset.basc().data)


@pytest.mark.parametrize("function", ["basc", "get_baseline"])
def test_keyword_ranges_none_is_rejected(function):
    with pytest.raises(TraitError):
        getattr(scp, function)(_curved_dataset(), ranges=None)


@pytest.mark.parametrize("function", ["basc", "get_baseline"])
def test_duplicate_ranges_specification_is_rejected(function):
    dataset = _curved_dataset()

    with pytest.raises(TypeError, match="either positionally or with the 'ranges'"):
        getattr(scp, function)(
            dataset,
            [3600.0, 3400.0],
            ranges=[[1400.0, 1100.0]],
        )


@pytest.mark.parametrize("function", ["basc", "get_baseline"])
@pytest.mark.parametrize("keyword", ["lambda_", "dim", "fit", "_fitted"])
@pytest.mark.parametrize("entrypoint", ["function", "method"])
def test_baseline_rejects_unknown_keywords(function, keyword, entrypoint):
    dataset = _curved_dataset()
    before = dataset.data.copy()
    call = (
        getattr(scp, function)
        if entrypoint == "function"
        else getattr(dataset, function)
    )
    args = (dataset,) if entrypoint == "function" else ()
    with pytest.raises(TypeError, match=keyword):
        call(*args, model="asls", **{keyword: 1e2})
    np.testing.assert_array_equal(dataset.data, before)


def test_baseline_accepts_documented_breakpoints():
    dataset = _curved_dataset()
    processor = scp.Baseline(model="detrend", order=1)
    processor.breakpoints = [60]
    processor.fit(dataset)
    result = dataset.get_baseline(model="detrend", order=1, breakpoints=[60])
    np.testing.assert_allclose(result.data, processor.baseline.data)
