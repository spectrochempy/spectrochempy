# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""
Tests for the curve-fitting 1-D models.

Uses deterministic grids throughout.  No plotting, no real-data dependencies.
"""

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.core.units import ur
from spectrochempy.utils.testing import assert_approx_equal


# (name, args, centre_value)
# ``centre_value`` is the value at x=pos obtained with
#   x = Coord.linspace(0, 0.999, 1000, units="m")
#   kwargs = {ampl: 1*g, width: 100*mm, ratio: 0.5, asym: 2, pos: 0.5, c_2: 1.0}
# The actual value is ``array[0.5].value``, then multiplied by 100 for all
# models except sigmoid.
_MODELS = [
    pytest.param(
        "gaussianmodel", ["ampl", "pos", "width"], 0.9394292818892936, id="gaussian"
    ),
    pytest.param(
        "lorentzianmodel", ["ampl", "pos", "width"], 0.6366197723675814, id="lorentzian"
    ),
    pytest.param(
        "voigtmodel", ["ampl", "pos", "width", "ratio"], 0.8982186579508358, id="voigt"
    ),
    pytest.param(
        "asymmetricvoigtmodel",
        ["ampl", "pos", "width", "ratio", "asym"],
        0.8982186579508358,
        id="asymmetric_voigt",
    ),
    pytest.param(
        "polynomialbaseline",
        ["ampl"] + [f"c_{i}" for i in range(2, 11)],
        0.0,
        id="polynomialbaseline",
    ),
    pytest.param("sigmoidmodel", ["ampl", "pos", "asym"], 50, id="sigmoid"),
]

_HELPERS = [
    pytest.param(
        "polynomial",
        dict(offset=1.0, slope=0.1, ampl=0.5, c_2=0.2),
        id="polynomial",
    ),
    pytest.param("gaussian", dict(ampl=1.0, pos=0.5, width=0.1), id="gaussian"),
    pytest.param("lorentzian", dict(ampl=1.0, pos=0.5, width=0.1), id="lorentzian"),
    pytest.param("voigt", dict(ampl=1.0, pos=0.5, width=0.1, ratio=0.5), id="voigt"),
    pytest.param(
        "asymmetricvoigt",
        dict(ampl=1.0, pos=0.5, width=0.1, ratio=0.5, asym=0.2),
        id="asymmetric_voigt",
    ),
    pytest.param("sigmoid", dict(ampl=1.0, pos=0.5, asym=2.0), id="sigmoid"),
]


def _full_kwargs():
    """Return the kwargs dict used for centre-value regression tests."""
    return dict(
        ampl=1.0 * ur("g"),
        width=100.0 * ur("mm"),
        ratio=0.5,
        asym=2.0,
        pos=0.5,
        c_2=1.0,
    )


def _filter_kwargs(kwargs, args):
    """Keep only keys that appear in *args*."""
    return {k: v for k, v in kwargs.items() if k in args}


# ---------------------------------------------------------------------------
# Construction metadata
# ---------------------------------------------------------------------------


class TestModelConstruction:
    """Model instantiation and static metadata."""

    @pytest.mark.parametrize("name, args, _", _MODELS)
    def test_args(self, name, args, _):
        """args attribute should match the expected parameter list."""
        assert getattr(scp, name)().args == args

    @pytest.mark.parametrize("name, _, __", _MODELS)
    def test_type(self, name, _, __):
        """All models should be labelled 1-D."""
        assert getattr(scp, name)().type == "1D"


# ---------------------------------------------------------------------------
# Evaluation — plain ndarray input (no units)
# ---------------------------------------------------------------------------


class TestEvaluationPlain:
    """Evaluation with a plain numpy array as x."""

    @pytest.mark.parametrize("name, args, _", _MODELS)
    def test_output(self, name, args, _):
        """Output should be an ndarray of the correct shape with finite values."""
        model = getattr(scp, name)()
        x = np.linspace(0, 1, 100)
        kwargs = _filter_kwargs(
            dict(ampl=1.0, width=0.1, ratio=0.5, asym=2.0, pos=0.5, c_2=1.0),
            args,
        )
        result = model.f(x, **kwargs)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100,)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("name, args, _", _MODELS)
    def test_amplitude_linearity(self, name, args, _):
        """Doubling ampl should double the output (sigmoid excluded — ampl appears in exponent too)."""
        if name == "sigmoidmodel":
            pytest.skip("sigmoid is not linear in ampl")
        model = getattr(scp, name)()
        x = np.linspace(0, 1, 100)
        kw = _filter_kwargs(
            dict(ampl=1.0, width=0.1, ratio=0.5, asym=2.0, pos=0.5, c_2=1.0),
            args,
        )
        r1 = model.f(x, **kw)
        kw["ampl"] = 2.0
        r2 = model.f(x, **kw)
        assert np.allclose(r2, 2.0 * r1, rtol=1e-12)


# ---------------------------------------------------------------------------
# Evaluation — inputs with physical units
# ---------------------------------------------------------------------------


class TestEvaluationUnits:
    """Unit handling when x or amplitude carry physical units."""

    @pytest.mark.parametrize("name, args, _", _MODELS)
    def test_x_units(self, name, args, _):
        """x with length units → ndarray output (no units if ampl is scalar)."""
        model = getattr(scp, name)()
        x = np.linspace(0, 1, 100) * ur("m")
        kwargs = _filter_kwargs(
            dict(ampl=1.0, width=0.1, ratio=0.5, asym=2.0, pos=0.5, c_2=1.0),
            args,
        )
        result = model.f(x, **kwargs)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100,)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("name, args, _", _MODELS)
    def test_ampl_units_propagate(self, name, args, _):
        """Amplitude with units should propagate to the output."""
        model = getattr(scp, name)()
        x = np.linspace(0, 1, 100) * ur("m")
        kwargs = _filter_kwargs(
            dict(ampl=1.0 * ur("g"), width=0.1, ratio=0.5, asym=2.0, pos=0.5, c_2=1.0),
            args,
        )
        result = model.f(x, **kwargs)
        assert hasattr(result, "units")
        assert result.units == ur("g")
        assert np.all(np.isfinite(result.magnitude))

    @pytest.mark.parametrize("name, args, _", _MODELS)
    def test_unit_conversion(self, name, args, _):
        """Width in compatible units (mm → m) should be auto-converted."""
        model = getattr(scp, name)()
        x = np.linspace(0, 1, 100) * ur("m")
        kwargs = _filter_kwargs(
            dict(
                ampl=1.0 * ur("g"),
                width=100.0 * ur("mm"),
                ratio=0.5,
                asym=2.0,
                pos=0.5,
                c_2=1.0,
            ),
            args,
        )
        result = model.f(x, **kwargs)
        assert hasattr(result, "units")
        assert result.units == ur("g")
        assert np.all(np.isfinite(result.magnitude))


# ---------------------------------------------------------------------------
# Evaluation — Coord input produces NDDataset
# ---------------------------------------------------------------------------


class TestEvaluationCoord:
    """Behaviour when x is a Coord object."""

    @pytest.mark.parametrize("name, args, _", _MODELS)
    def test_returns_nddataset(self, name, args, _):
        """Coord input should return an NDDataset."""
        model = getattr(scp, name)()
        x = scp.Coord.linspace(0, 1, 100)
        kwargs = _filter_kwargs(
            dict(ampl=1.0 * ur("g"), width=0.1, ratio=0.5, asym=2.0, pos=0.5, c_2=1.0),
            args,
        )
        result = model.f(x, **kwargs)
        assert isinstance(result, scp.NDDataset)
        assert result.shape == (100,)
        assert np.all(np.isfinite(result.data))
        assert result.units == ur("g")

    @pytest.mark.parametrize("name, args, _", _MODELS)
    def test_coord_with_units(self, name, args, _):
        """Coord with units should produce NDDataset with correct metadata."""
        model = getattr(scp, name)()
        x = scp.Coord.linspace(0, 0.999, 1000, units="m", title="distance")
        kwargs = _filter_kwargs(_full_kwargs(), args)
        result = model.f(x, **kwargs)
        assert isinstance(result, scp.NDDataset)
        assert result.shape == (1000,)
        assert np.all(np.isfinite(result.data))
        assert result.units == ur("g")


# ---------------------------------------------------------------------------
# Numerical centre values  (regression – preserves original expected numbers)
# ---------------------------------------------------------------------------


class TestNumericalValues:
    """Each model should produce the expected centre value for a fixed setup."""

    @pytest.mark.parametrize("name, args, expected", _MODELS)
    def test_center_value(self, name, args, expected):
        """Value at x=pos should match the known reference."""
        model = getattr(scp, name)()
        x = scp.Coord.linspace(0, 0.999, 1000, units="m", title="distance")
        kwargs = _filter_kwargs(_full_kwargs(), args)
        result = model.f(x, **kwargs)
        actual = result[0.5].value * 100
        assert_approx_equal(actual.m, expected, significant=4)


class TestConvenienceHelpers:
    """Top-level shape helpers should expose the built-in 1D models."""

    @pytest.mark.parametrize("name, kwargs", _HELPERS)
    def test_helper_returns_nddataset_for_coord(self, name, kwargs):
        x = scp.Coord.linspace(0, 1, 100, units="m")

        result = getattr(scp, name)(x, **kwargs)

        assert isinstance(result, scp.NDDataset)
        assert result.shape == (100,)
        assert result.dims == ["x"]
        assert result.name == name
        assert result.title == "intensity"
        assert result.x.units == ur("m")
        assert np.all(np.isfinite(result.data))

    @pytest.mark.parametrize("name, kwargs", _HELPERS)
    def test_helper_returns_ndarray_for_numpy_input(self, name, kwargs):
        x = np.linspace(0, 1, 100)

        result = getattr(scp, name)(x, **kwargs)

        assert isinstance(result, np.ndarray)
        assert result.shape == (100,)
        assert np.all(np.isfinite(result))


class TestNormalizedFalse:
    """normalized=False should make the peak amplitude equal to *ampl*."""

    @pytest.mark.parametrize(
        "helper,extra_kw",
        [
            pytest.param("gaussian", {}, id="gaussian"),
            pytest.param("lorentzian", {}, id="lorentzian"),
            pytest.param("voigt", {"ratio": 0.5}, id="voigt"),
        ],
    )
    def test_peak_equals_ampl(self, helper, extra_kw):
        x = np.linspace(-5, 5, 2000)
        result = getattr(scp, helper)(
            x, ampl=3.0, pos=0.0, width=1.0, normalized=False, **extra_kw
        )
        assert np.isclose(result.max(), 3.0, rtol=1e-2)

    @pytest.mark.parametrize(
        "helper,extra_kw",
        [
            pytest.param("gaussian", {}, id="gaussian"),
            pytest.param("lorentzian", {}, id="lorentzian"),
            pytest.param("voigt", {"ratio": 0.5}, id="voigt"),
        ],
    )
    def test_peak_at_pos(self, helper, extra_kw):
        x = np.linspace(-5, 5, 2000)
        result = getattr(scp, helper)(
            x, ampl=2.0, pos=1.5, width=1.0, normalized=False, **extra_kw
        )
        idx = np.argmin(np.abs(x - 1.5))
        assert np.isclose(result[idx], 2.0, rtol=1e-2)

    @pytest.mark.parametrize(
        "helper,extra_kw",
        [
            pytest.param("gaussian", {}, id="gaussian"),
            pytest.param("lorentzian", {}, id="lorentzian"),
            pytest.param("voigt", {"ratio": 0.5}, id="voigt"),
        ],
    )
    def test_normalized_true_is_default(self, helper, extra_kw):
        x = np.linspace(-5, 5, 1000)
        default = getattr(scp, helper)(x, ampl=1.0, pos=0.0, width=1.0, **extra_kw)
        explicit = getattr(scp, helper)(
            x, ampl=1.0, pos=0.0, width=1.0, normalized=True, **extra_kw
        )
        np.testing.assert_allclose(default, explicit, rtol=1e-12)

    def test_normalized_false_coord_input(self):
        x = scp.Coord.linspace(-5, 5, 1000, units="m")
        result = scp.gaussian(x, ampl=1.0, pos=0.0, width=1.0, normalized=False)
        assert isinstance(result, scp.NDDataset)
        assert np.isclose(result.data.max(), 1.0, rtol=1e-3)

    def test_normalized_false_amplitude_linearity(self):
        x = np.linspace(-5, 5, 1000)
        r1 = scp.gaussian(x, ampl=1.0, pos=0.0, width=1.0, normalized=False)
        r2 = scp.gaussian(x, ampl=2.5, pos=0.0, width=1.0, normalized=False)
        np.testing.assert_allclose(r2, 2.5 * r1, rtol=1e-12)


class TestBaselineHelper:
    """The baseline helper should provide ergonomic synthetic baselines."""

    def test_baseline_helper_matches_polynomial_model_without_linear_terms(self):
        x = scp.Coord.linspace(0, 1, 100, units="m")

        helper = scp.polynomial(x, ampl=2.0, c_2=0.5)
        model = scp.polynomialbaseline().f(x, ampl=2.0, c_2=0.5)

        np.testing.assert_allclose(helper.data, model.data, rtol=1e-12)

    def test_baseline_helper_adds_offset_and_slope(self):
        x = np.linspace(-1, 1, 101)

        baseline = scp.polynomial(x, offset=2.0, slope=3.0)

        assert np.isclose(baseline[50], 2.0, rtol=1e-12)
        assert baseline[-1] > baseline[0]
