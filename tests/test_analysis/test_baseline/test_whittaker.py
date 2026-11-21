"""Numerical references from pybaselines 1.2.1; no runtime/test dependency on it."""

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from traitlets import TraitError

import spectrochempy as scp
from spectrochempy.processing.baselineprocessing._whittaker import _whittaker_baseline
from spectrochempy.utils.exceptions import NotFittedError

REFERENCE = json.loads(Path(__file__).with_name("whittaker_reference.json").read_text())


def spectrum(descending=False, rows=1):
    case = REFERENCE["cases"][0]
    x = np.array(REFERENCE["coordinates"])
    y = np.array(case["data"])
    if rows > 1:
        y = np.stack([y, 2 * y + 5])
    if descending:
        x, y = x[::-1], y[..., ::-1]
    coords = [scp.Coord(x, units="cm^-1")]
    if rows > 1:
        coords.insert(0, scp.Coord([0.0, 1.0], units="s"))
    source = scp.NDDataset(y, coordset=coords, units="absorbance")
    source.name = "source"
    source.description = "acquisition context"
    source.author = "spectroscopist"
    source.meta.sample = "unchanged"
    source.history = "acquired"
    return source


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_matches_independent_reference(case):
    source = scp.NDDataset(case["data"], coordset=[scp.Coord(REFERENCE["coordinates"])])
    processor = scp.Baseline(model=case["model"], **case["params"])
    expected = (
        nullcontext()
        if case["converged"]
        else pytest.warns(RuntimeWarning, match="convergence")
    )
    with expected:
        processor.fit(source)
    assert_allclose(processor.baseline.data, case["baseline"], rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize("model", ["arpls", "aspls"])
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("descending", [False, True])
def test_public_contract_and_history(model, rows, descending):
    source = spectrum(descending, rows)
    original = source.data.copy()
    history = source.history_entries
    processor = scp.Baseline(model=model, lamb=1e4, max_iter=100)
    processor.fit(source)
    corrected = processor.corrected
    for output in [processor.baseline, corrected]:
        assert output.shape == source.shape
        assert output.dims == source.dims
        assert output.units == source.units
        assert_array_equal(output.x.data, source.x.data)
        if rows > 1:
            assert_array_equal(output.y.data, source.y.data)
    assert corrected.name == source.name
    assert corrected.description == source.description
    assert corrected.author == source.author
    assert corrected.meta.sample == "unchanged"
    assert len(corrected.history_entries) == len(history) + 1
    entry = corrected.history_entries[-1]
    assert entry["operation"] == "baseline_correction"
    assert entry["parameters"]["model"] == model
    assert entry["parameters"]["lamb"] == 1e4
    assert entry["parameters"]["tol"] == 1e-3
    assert entry["parameters"]["max_iter"] == 100
    if model == "aspls":
        assert entry["parameters"]["aspls_asymmetry"] == 0.5
    assert_array_equal(source.data, original)
    assert source.history_entries == history
    assert_allclose(corrected.data, source.data - processor.baseline.data)


@pytest.mark.parametrize("model", ["arpls", "aspls"])
@pytest.mark.parametrize("level", [0.0, -3.0, 7.0, 1e300, 1e-300])
def test_constant_and_affine_nullspace(model, level):
    for y in [np.full(16, level), level * np.linspace(1, 2, 16)]:
        baseline, converged = _whittaker_baseline(y, model, 1e5, 1e-3, 50)
        assert_array_equal(baseline, y)
        assert converged


@pytest.mark.parametrize("model", ["arpls", "aspls"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("observation_descending", [False, True])
def test_transposed_dimensions_use_last_axis(model, descending, observation_descending):
    source = spectrum(descending, rows=2)
    observation = scp.Coord([2.0, 1.0] if observation_descending else [1.0, 2.0])
    ordinary = scp.NDDataset(
        source.data, coordset=[observation.copy(), source.x.copy()]
    )
    transposed = scp.NDDataset(
        source.data.T, coordset=[source.x.copy(), observation.copy()]
    ).T
    options = {"model": model, "lamb": 1e4, "max_iter": 100}
    expected = ordinary.get_baseline(**options)
    processor = scp.Baseline(**options)
    processor.fit(transposed)
    assert transposed.dims == ["x", "y"]
    assert processor.baseline.dims == transposed.dims
    assert_allclose(processor.baseline.data, expected.data, rtol=1e-10, atol=1e-10)
    assert_allclose(processor.corrected.data, transposed.data - expected.data)
    assert_array_equal(processor.baseline.x.data, transposed.x.data)
    assert_array_equal(processor.baseline.y.data, transposed.y.data)


@pytest.mark.parametrize("model", ["arpls", "aspls"])
def test_three_point_penalty_and_iteration_limit(model):
    y = np.array([1.0, 3.0, 2.0])
    difference = np.array([[1.0, -2.0, 1.0]])
    expected = np.linalg.solve(np.eye(3) + 10 * difference.T @ difference, y)
    actual, _ = _whittaker_baseline(y, model, 10, 1e-3, 1)
    assert_allclose(actual, expected)


@pytest.mark.parametrize("model", ["arpls", "aspls"])
@pytest.mark.parametrize(
    "invalid", ["complex", "nan", "inf", "masked", "irregular", "short", "no_coord"]
)
def test_rejected_refit_is_nonmutating_and_invalidates_results(model, invalid):
    source = spectrum()
    processor = scp.Baseline(model=model, lamb=1e4, max_iter=100)
    processor.fit(source)
    bad = source.copy()
    if invalid == "complex":
        bad = scp.NDDataset(
            source.data.astype(complex) + 1j, coordset=[source.x.copy()]
        )
    elif invalid in {"nan", "inf"}:
        bad.data[12] = np.nan if invalid == "nan" else np.inf
    elif invalid == "masked":
        bad.mask = np.arange(bad.size) < 10
    elif invalid == "irregular":
        x = bad.x.data.copy()
        x[12] += 0.001
        bad.set_coordset(scp.Coord(x))
    elif invalid == "short":
        bad = bad[:2]
    else:
        bad = scp.NDDataset(source.data)
    original, mask = bad.data.copy(), np.array(bad.mask, copy=True)
    with pytest.raises(ValueError, match=f"{model} baseline"):
        processor.fit(bad)
    with pytest.raises(NotFittedError):
        _ = processor.baseline
    with pytest.raises(NotFittedError):
        _ = processor.corrected
    assert_array_equal(bad.data, original)
    assert_array_equal(bad.mask, mask)


@pytest.mark.parametrize("model", ["arpls", "aspls"])
@pytest.mark.parametrize(
    "params",
    [
        {"lamb": 0},
        {"lamb": -1},
        {"lamb": np.nan},
        {"lamb": np.inf},
        {"tol": 0},
        {"tol": 1},
        {"tol": np.nan},
        {"max_iter": 0},
        {"multivariate": True},
        {"lls": True},
        {"breakpoints": [30]},
    ],
)
def test_invalid_configuration(model, params):
    with pytest.raises((ValueError, TraitError)):
        scp.Baseline(model=model, **params).fit(spectrum())


@pytest.mark.parametrize("coefficient", [0, -1, np.nan, np.inf])
def test_invalid_aspls_coefficient(coefficient):
    with pytest.raises(ValueError, match="aspls_asymmetry"):
        scp.Baseline(model="aspls", aspls_asymmetry=coefficient).fit(spectrum())


@pytest.mark.parametrize("model", ["arpls", "aspls"])
def test_existing_helpers_and_default_preserved(model):
    source = spectrum()
    options = {"model": model, "lamb": 1e4, "max_iter": 100}
    baseline = source.get_baseline(**options)
    corrected = source.basc(**options)
    assert_allclose(corrected.data, source.data - baseline.data)
    assert scp.Baseline.class_traits()["model"].default_value == "polynomial"


@pytest.mark.parametrize("model", ["arpls", "aspls"])
def test_recovery_of_known_narrow_peaks(model):
    x = np.linspace(0, 1, 1024)
    background = 2 + 0.3 * x
    peaks = 3 * np.exp(-0.5 * ((x - 0.3) / 0.01) ** 2)
    peaks += 2 * np.exp(-0.5 * ((x - 0.7) / 0.015) ** 2)
    y = background + peaks + np.random.default_rng(17).normal(0, 0.01, x.size)
    source = scp.NDDataset(y, coordset=[scp.Coord(200 + 1600 * x)])
    processor = scp.Baseline(model=model, lamb=1e6, max_iter=200)
    processor.fit(source)
    assert np.sqrt(np.mean((processor.baseline.data - background) ** 2)) < 0.01
    area = np.trapezoid(processor.corrected.data, x)
    assert area == pytest.approx(np.trapezoid(peaks, x), rel=0.05)


@pytest.mark.parametrize("model", ["arpls", "aspls"])
def test_history_uses_fitted_settings_and_refit_replaces_them(model):
    source = spectrum()
    processor = scp.Baseline(model=model, lamb=1e4, max_iter=100)
    processor.fit(source)
    fitted = processor.baseline.data.copy()
    processor.lamb = 1e6
    processor.tol = 0.01
    processor.max_iter = 1
    processor.aspls_asymmetry = 2
    entry = processor.corrected.history_entries[-1]
    assert_array_equal(processor.baseline.data, fitted)
    assert entry["parameters"]["lamb"] == 1e4
    assert entry["parameters"]["tol"] == 1e-3
    assert entry["parameters"]["max_iter"] == 100
    if model == "aspls":
        assert entry["parameters"]["aspls_asymmetry"] == 0.5
    entry["parameters"]["lamb"] = -1
    assert processor.corrected.history_entries[-1]["parameters"]["lamb"] == 1e4
    with pytest.warns(RuntimeWarning, match="convergence"):
        processor.fit(source)
    parameters = processor.corrected.history_entries[-1]["parameters"]
    assert parameters["lamb"] == 1e6
    assert parameters["tol"] == 0.01
    assert parameters["max_iter"] == 1
    if model == "aspls":
        assert parameters["aspls_asymmetry"] == 2


@pytest.mark.parametrize("model", ["arpls", "aspls"])
def test_fitted_model_provenance_survives_model_switch_and_old_masked_state(model):
    original = spectrum()
    processor = scp.Baseline(model="asls", lamb=1e4)
    processor.fit(original)
    source = original + 5
    processor.model = model
    processor.max_iter = 100
    processor.fit(source)
    corrected = processor.corrected.data.copy()
    processor.model = "asls"
    result = processor.corrected
    assert_allclose(result.data, corrected)
    assert result.name == source.name
    entry = result.history_entries[-1]
    assert entry["parameters"]["model"] == model
    assert f"using {model} model" in entry["message"]
