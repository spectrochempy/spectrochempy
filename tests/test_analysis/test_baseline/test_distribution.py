import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from traitlets import TraitError

import spectrochempy as scp
from spectrochempy.processing.baselineprocessing._distribution import (
    _distribution_baseline,
)
from spectrochempy.utils.exceptions import NotFittedError


def spectrum(descending=False, rows=1):
    points = np.arange(4096)
    coordinate = np.linspace(0.0, 10.0, points.size)
    x = np.linspace(-1.0, 1.0, points.size)
    background = 2.0 + 0.2 * x + 0.1 * x * x
    peak = 5.0 * np.exp(-0.5 * ((points - 2048) / 6.0) ** 2)
    noise = np.random.default_rng(31).normal(0.0, 0.01, points.size)
    data = background + peak + noise
    if rows > 1:
        data = np.stack([data, -2.0 * data + 13.0])
    if descending:
        coordinate = coordinate[::-1]
        data = data[..., ::-1]
        background, peak, noise = (v[::-1] for v in (background, peak, noise))
    coords = [scp.Coord(coordinate, units="ppm")]
    if rows > 1:
        coords.insert(0, scp.Coord(np.arange(rows), units="s"))
    source = scp.NDDataset(data, coordset=coords, units="absorbance")
    source.meta.acquisition = {"sample": "unchanged"}
    return source, background, peak, noise


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("rows", [1, 2])
def test_distribution_public_model_preserves_context_and_signed_peaks(descending, rows):
    source, background, peak, noise = spectrum(descending, rows)
    original = source.data.copy()
    processor = scp.Baseline(model="distribution", distribution_width=20)
    processor.fit(source)
    estimated = processor.baseline
    corrected = processor.corrected
    for result in (estimated, corrected):
        assert result.shape == source.shape
        assert result.dims == source.dims
        assert result.units == source.units
        assert_array_equal(result.x.data, source.x.data)
    assert corrected.meta.acquisition["sample"] == "unchanged"
    assert_array_equal(source.data, original)
    assert source.history_entries == []
    first_baseline = estimated.data[0] if rows > 1 else estimated.data
    first_corrected = corrected.data[0] if rows > 1 else corrected.data
    assert np.sqrt(np.mean((first_baseline - background) ** 2)) < 0.003
    region = peak > 0.00001
    assert np.sum((first_corrected - noise)[region]) == pytest.approx(
        np.sum(peak[region]), rel=0.005
    )
    if rows > 1:
        assert_allclose(estimated.data[1], -2.0 * estimated.data[0] + 13.0, atol=2e-12)
        assert_allclose(corrected.data[1], -2.0 * corrected.data[0], atol=2e-12)
    entry = corrected.history_entries[-1]
    assert entry["operation"] == "baseline_correction"
    assert entry["parameters"]["model"] == "distribution"
    assert entry["parameters"]["distribution_width"] == 20
    assert entry["parameters"]["experimental"] is True


@pytest.mark.parametrize("length", [1, 2, 8, 128])
@pytest.mark.parametrize("level", [0.0, -3.0, 7.0, 1e300])
def test_distribution_constant_trace(length, level):
    source = scp.NDDataset(
        np.full(length, level), coordset=[scp.Coord(np.arange(length))]
    )
    processor = scp.Baseline(model="distribution")
    processor.fit(source)
    assert_array_equal(processor.baseline.data, source.data)
    assert_array_equal(processor.corrected.data, np.zeros(length))


@pytest.mark.parametrize("width", [0, -1, True, 1.5])
def test_distribution_rejects_invalid_half_width(width):
    with pytest.raises((ValueError, TraitError)):
        scp.Baseline(model="distribution", distribution_width=width)


@pytest.mark.parametrize(
    "options", [{"multivariate": True}, {"lls": True}, {"breakpoints": [100]}]
)
def test_distribution_rejects_unsupported_transformations(options):
    source, *_ = spectrum()
    with pytest.raises(ValueError, match="does not support"):
        scp.Baseline(model="distribution", **options).fit(source)


@pytest.mark.parametrize("invalid", ["complex", "nan", "masked", "irregular"])
def test_distribution_rejection_is_nonmutating_and_invalidates_refit(invalid):
    source, *_ = spectrum()
    processor = scp.Baseline(model="distribution")
    processor.fit(source)
    bad = source.copy()
    if invalid == "complex":
        bad = scp.NDDataset(
            source.data.astype(complex) + 1j, coordset=[source.x.copy()]
        )
    elif invalid == "nan":
        bad.data[100] = np.nan
    elif invalid == "masked":
        bad.mask = np.arange(bad.size) < 10
    else:
        coordinates = source.x.data.copy()
        coordinates[100] += 0.0001
        bad.set_coordset(scp.Coord(coordinates))
    original = bad.data.copy()
    mask = np.array(bad.mask, copy=True)
    with pytest.raises(ValueError, match="distribution baseline"):
        processor.fit(bad)
    with pytest.raises(NotFittedError):
        _ = processor.baseline
    assert_array_equal(bad.data, original)
    assert_array_equal(bad.mask, mask)


def test_distribution_is_available_through_existing_dataset_helpers():
    source, *_ = spectrum()
    baseline = source.get_baseline(model="distribution", distribution_width=20)
    corrected = source.basc(model="distribution", distribution_width=20)
    assert_allclose(corrected.data, source.data - baseline.data)
    assert scp.Baseline.class_traits()["model"].default_value == "polynomial"


def test_broad_weak_peak_is_a_documented_limit_not_a_quantitative_guarantee():
    points = np.arange(4096)
    x = np.linspace(-1.0, 1.0, points.size)
    background = 2.0 + 0.2 * x + 0.1 * x * x
    peak = 0.15 * np.exp(-0.5 * ((points - 2048) / 100.0) ** 2)
    noise = np.random.default_rng(31).normal(0.0, 0.01, points.size)
    data = background + peak + noise
    baseline = _distribution_baseline(data, 20)
    region = abs(points - 2048) <= 500
    recovered = np.sum((data - baseline - noise)[region])
    assert abs(recovered) < 0.1 * np.sum(peak)


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_distribution_promotes_integer_input_for_fractional_baselines(dtype):
    source = scp.NDDataset(
        np.array([1, 2, 3, 5], dtype=dtype),
        coordset=[scp.Coord(np.arange(4))],
        dtype=dtype,
    )
    processor = scp.Baseline(model="distribution")
    processor.fit(source)
    assert_allclose(processor.baseline.data, np.full(4, 2.75))
    assert_allclose(processor.corrected.data, source.data - 2.75)
    assert source.dtype == dtype
