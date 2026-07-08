# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: N815
"""
Tests for the public ``constraints=`` API on ``MCRALS``.

These tests verify that:

- The ``constraints=`` parameter accepts ``None``, an empty list, or a
  sequence of public ``Constraint`` instances.
- Invalid inputs raise appropriate ``TypeError`` / ``ValueError``.
- Mixed-API usage (legacy constraint traitlets + ``constraints=``) raises
  ``ValueError``.
- The internal constraint objects and the numerical ALS output are identical
  for equivalent legacy and public configurations.
"""

import numpy as np
import pytest

from spectrochempy.analysis.decomposition._legacy_constraint_converter import (
    legacy_to_constraints,
)
from spectrochempy.analysis.decomposition.mcrals import MCRALS
from spectrochempy.analysis.decomposition.mcrals_constraints import Closure
from spectrochempy.analysis.decomposition.mcrals_constraints import ModelProfile
from spectrochempy.analysis.decomposition.mcrals_constraints import Monotonic
from spectrochempy.analysis.decomposition.mcrals_constraints import NonNegative
from spectrochempy.analysis.decomposition.mcrals_constraints import ReferenceProfile
from spectrochempy.analysis.decomposition.mcrals_constraints import Unimodal

# ---------------------------------------------------------------------------------------
# Fixtures: deterministic synthetic PR4 dataset
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pr4_data():
    rng = np.random.RandomState(0)
    n_obs, n_wl = 8, 12
    C_true = np.zeros((n_obs, 2))
    C_true[:, 0] = np.array([0.1, 0.3, 0.8, 1.0, 0.7, 0.3, 0.1, 0.05])
    C_true[:, 1] = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 0.9])
    St_true = np.zeros((2, n_wl))
    St_true[0] = np.array(
        [0.1, 0.3, 0.9, 1.0, 0.8, 0.4, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01]
    )
    St_true[1] = np.array(
        [0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 0.9, 1.0, 0.8, 0.5, 0.2, 0.05]
    )
    X = C_true @ St_true + 1.0e-6 * rng.randn(n_obs, n_wl)
    C0 = np.abs(C_true + 0.05 * np.array([[1.0, -0.5]] * n_obs))
    return X, C0


# ---------------------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------------------


class TestConstraintsParameterValidation:
    """``constraints=`` parameter must accept valid inputs and reject invalid ones."""

    def test_default_is_none(self):
        """Omitting constraints should result in ``_constraints = None``."""
        mcr = MCRALS()
        assert mcr._constraints is None

    def test_explicit_none(self):
        mcr = MCRALS(constraints=None)
        assert mcr._constraints is None

    def test_empty_list(self):
        mcr = MCRALS(constraints=[])
        assert mcr._constraints == []

    def test_empty_tuple(self):
        mcr = MCRALS(constraints=())
        assert mcr._constraints == []

    def test_single_constraint(self):
        mcr = MCRALS(constraints=[NonNegative("C")])
        assert len(mcr._constraints) == 1
        assert isinstance(mcr._constraints[0], NonNegative)

    def test_multiple_constraints(self):
        mcr = MCRALS(constraints=[NonNegative("C"), Unimodal("C"), NonNegative("St")])
        assert len(mcr._constraints) == 3

    def test_all_constraint_types(self):
        """
        All public constraint types are accepted (even those without an
        internal engine counterpart yet).
        """
        constraints = [
            NonNegative("C"),
            Unimodal("C"),
            Monotonic("C", "increasing"),
            Closure("C"),
            ReferenceProfile("C", component=0, data=[1.0, 2.0]),
        ]
        mcr = MCRALS(constraints=constraints)
        assert len(mcr._constraints) == 5

    def test_invalid_type_string(self):
        with pytest.raises(TypeError, match="must be a Constraint instance"):
            MCRALS(constraints=["not_a_constraint"])

    def test_invalid_type_int(self):
        with pytest.raises(TypeError, match="must be a Constraint instance"):
            MCRALS(constraints=[42])

    def test_invalid_type_mixed(self):
        with pytest.raises(TypeError, match="must be a Constraint instance"):
            MCRALS(constraints=[NonNegative("C"), "bad"])

    def test_invalid_sequence_type(self):
        with pytest.raises(TypeError, match="must be a list or tuple"):
            MCRALS(constraints="not_a_sequence")


# ---------------------------------------------------------------------------------------
# Mixed-API guard
# ---------------------------------------------------------------------------------------


class TestMixedApiGuard:
    """Mixing legacy constraint traitlets with ``constraints=`` must raise."""

    def test_mixed_nonneg_conc(self):
        with pytest.raises(ValueError, match="cannot be used together"):
            MCRALS(nonnegConc="all", constraints=[NonNegative("C")])

    def test_mixed_unimod_conc(self):
        with pytest.raises(ValueError, match="cannot be used together"):
            MCRALS(unimodConc="all", constraints=[Unimodal("C")])

    def test_mixed_hard_conc(self):
        with pytest.raises(ValueError, match="cannot be used together"):
            MCRALS(hardConc=[0], constraints=[NonNegative("C")])

    def test_mixed_closure(self):
        with pytest.raises(ValueError, match="cannot be used together"):
            MCRALS(closureConc="all", constraints=[Closure("C")])

    def test_mixed_non_legacy_trait_is_ok(self):
        """
        Non-constraint traitlets (tol, max_iter, ...) can be combined
        with ``constraints=``.
        """
        mcr = MCRALS(constraints=[NonNegative("C")], tol=1.0e-9, max_iter=50)
        assert mcr._constraints is not None
        assert mcr.tol == 1.0e-9
        assert mcr.max_iter == 50


# ---------------------------------------------------------------------------------------
# Numerical equivalence: legacy API vs new API
# ---------------------------------------------------------------------------------------


class TestNumericalEquivalence:
    """
    For equivalent configurations, the legacy and new APIs must produce
    identical numerical results.
    """

    @pytest.fixture(autouse=True)
    def _fit_options(self):
        self._kw = {"tol": 1.0e-9, "max_iter": 50}

    def _assert_equal(self, mcr_legacy, mcr_new):
        np.testing.assert_allclose(
            np.asarray(mcr_legacy.C.data),
            np.asarray(mcr_new.C.data),
            rtol=1.0e-10,
        )
        np.testing.assert_allclose(
            np.asarray(mcr_legacy.St.data),
            np.asarray(mcr_new.St.data),
            rtol=1.0e-10,
        )

    def test_baseline_empty(self, pr4_data):
        X, C0 = pr4_data
        mcr_legacy = MCRALS(nonnegConc=[], nonnegSpec=[], unimodConc=[], **self._kw)
        mcr_legacy.fit(X, C0)
        mcr_new = MCRALS(constraints=[], **self._kw)
        mcr_new.fit(X, C0)
        self._assert_equal(mcr_legacy, mcr_new)

    def test_nonneg_conc(self, pr4_data):
        X, C0 = pr4_data
        mcr_legacy = MCRALS(nonnegConc="all", nonnegSpec=[], unimodConc=[], **self._kw)
        mcr_legacy.fit(X, C0)
        mcr_new = MCRALS(constraints=[NonNegative("C")], **self._kw)
        mcr_new.fit(X, C0)
        self._assert_equal(mcr_legacy, mcr_new)

    def test_nonneg_spec(self, pr4_data):
        X, C0 = pr4_data
        mcr_legacy = MCRALS(nonnegConc=[], nonnegSpec="all", unimodConc=[], **self._kw)
        mcr_legacy.fit(X, C0)
        mcr_new = MCRALS(constraints=[NonNegative("St")], **self._kw)
        mcr_new.fit(X, C0)
        self._assert_equal(mcr_legacy, mcr_new)

    def test_nonneg_both(self, pr4_data):
        X, C0 = pr4_data
        mcr_legacy = MCRALS(
            nonnegConc="all", nonnegSpec="all", unimodConc=[], **self._kw
        )
        mcr_legacy.fit(X, C0)
        mcr_new = MCRALS(constraints=[NonNegative("C"), NonNegative("St")], **self._kw)
        mcr_new.fit(X, C0)
        self._assert_equal(mcr_legacy, mcr_new)

    def test_unimod_conc(self, pr4_data):
        X, C0 = pr4_data
        mcr_legacy = MCRALS(nonnegConc=[], nonnegSpec=[], unimodConc="all", **self._kw)
        mcr_legacy.fit(X, C0)
        mcr_new = MCRALS(constraints=[Unimodal("C")], **self._kw)
        mcr_new.fit(X, C0)
        self._assert_equal(mcr_legacy, mcr_new)

    def test_unimod_spec(self, pr4_data):
        X, C0 = pr4_data
        mcr_legacy = MCRALS(
            nonnegConc=[],
            nonnegSpec=[],
            unimodConc=[],
            unimodSpec="all",
            **self._kw,
        )
        mcr_legacy.fit(X, C0)
        mcr_new = MCRALS(constraints=[Unimodal("St")], **self._kw)
        mcr_new.fit(X, C0)
        self._assert_equal(mcr_legacy, mcr_new)

    def test_monotonic_increasing(self, pr4_data):
        X, C0 = pr4_data
        mcr_legacy = MCRALS(
            nonnegConc=[],
            nonnegSpec=[],
            unimodConc=[],
            monoIncConc=[1],
            **self._kw,
        )
        mcr_legacy.fit(X, C0)
        mcr_new = MCRALS(
            constraints=[Monotonic("C", "increasing", components=[1])],
            **self._kw,
        )
        mcr_new.fit(X, C0)
        self._assert_equal(mcr_legacy, mcr_new)

    def test_monotonic_decreasing(self, pr4_data):
        X, C0 = pr4_data
        mcr_legacy = MCRALS(
            nonnegConc=[],
            nonnegSpec=[],
            unimodConc=[],
            monoDecConc=[0],
            **self._kw,
        )
        mcr_legacy.fit(X, C0)
        mcr_new = MCRALS(
            constraints=[Monotonic("C", "decreasing", components=[0])],
            **self._kw,
        )
        mcr_new.fit(X, C0)
        self._assert_equal(mcr_legacy, mcr_new)

    def test_closure(self, pr4_data):
        X, C0 = pr4_data
        mcr_legacy = MCRALS(
            nonnegConc=[],
            nonnegSpec=[],
            unimodConc=[],
            closureConc="all",
            **self._kw,
        )
        mcr_legacy.fit(X, C0)
        mcr_new = MCRALS(constraints=[Closure("C")], **self._kw)
        mcr_new.fit(X, C0)
        self._assert_equal(mcr_legacy, mcr_new)

    def test_combined_nonneg_unimod(self, pr4_data):
        X, C0 = pr4_data
        mcr_legacy = MCRALS(
            nonnegConc="all", nonnegSpec=[], unimodConc="all", **self._kw
        )
        mcr_legacy.fit(X, C0)
        mcr_new = MCRALS(constraints=[NonNegative("C"), Unimodal("C")], **self._kw)
        mcr_new.fit(X, C0)
        self._assert_equal(mcr_legacy, mcr_new)

    def test_combined_nonneg_unimod_monotonic(self, pr4_data):
        X, C0 = pr4_data
        mcr_legacy = MCRALS(
            nonnegConc="all",
            nonnegSpec=[],
            unimodConc="all",
            monoIncConc=[1],
            **self._kw,
        )
        mcr_legacy.fit(X, C0)
        mcr_new = MCRALS(
            constraints=[
                NonNegative("C"),
                Unimodal("C"),
                Monotonic("C", "increasing", components=[1]),
            ],
            **self._kw,
        )
        mcr_new.fit(X, C0)
        self._assert_equal(mcr_legacy, mcr_new)

    def test_nonneg_conc_with_selected_components(self, pr4_data):
        X, C0 = pr4_data
        mcr_legacy = MCRALS(nonnegConc=[1], nonnegSpec=[], unimodConc=[], **self._kw)
        mcr_legacy.fit(X, C0)
        mcr_new = MCRALS(constraints=[NonNegative("C", components=[1])], **self._kw)
        mcr_new.fit(X, C0)
        self._assert_equal(mcr_legacy, mcr_new)


# ---------------------------------------------------------------------------------------
# `legacy_to_constraints` round-trip
# ---------------------------------------------------------------------------------------


class TestLegacyConverterRoundTrip:
    """
    The ``legacy_to_constraints`` converter produces public constraints
    that, when passed back to ``MCRALS(constraints=...)``, give identical
    numerical results.
    """

    @pytest.fixture(autouse=True)
    def _fit_options(self):
        self._kw = {"tol": 1.0e-9, "max_iter": 50}

    def _assert_equal(self, mcr_a, mcr_b):
        np.testing.assert_allclose(
            np.asarray(mcr_a.C.data),
            np.asarray(mcr_b.C.data),
            rtol=1.0e-10,
        )
        np.testing.assert_allclose(
            np.asarray(mcr_a.St.data),
            np.asarray(mcr_b.St.data),
            rtol=1.0e-10,
        )

    def test_default_config_round_trip(self, pr4_data):
        X, C0 = pr4_data
        mcr_legacy = MCRALS(**self._kw)
        mcr_legacy.fit(X, C0)

        constraints = legacy_to_constraints(mcr_legacy)
        mcr_new = MCRALS(constraints=constraints, **self._kw)
        mcr_new.fit(X, C0)
        self._assert_equal(mcr_legacy, mcr_new)

    def test_full_config_round_trip(self, pr4_data):
        X, C0 = pr4_data
        mcr_legacy = MCRALS(
            nonnegConc="all",
            nonnegSpec="all",
            unimodConc="all",
            **self._kw,
        )
        mcr_legacy.fit(X, C0)

        constraints = legacy_to_constraints(mcr_legacy)
        mcr_new = MCRALS(constraints=constraints, **self._kw)
        mcr_new.fit(X, C0)
        self._assert_equal(mcr_legacy, mcr_new)

    def test_closure_round_trip(self, pr4_data):
        X, C0 = pr4_data
        mcr_legacy = MCRALS(
            closureConc="all",
            nonnegConc=[],
            nonnegSpec=[],
            unimodConc=[],
            **self._kw,
        )
        mcr_legacy.fit(X, C0)

        constraints = legacy_to_constraints(mcr_legacy)
        mcr_new = MCRALS(constraints=constraints, **self._kw)
        mcr_new.fit(X, C0)
        self._assert_equal(mcr_legacy, mcr_new)

    def test_multi_constraint_round_trip(self, pr4_data):
        X, C0 = pr4_data
        mcr_legacy = MCRALS(
            nonnegConc="all",
            nonnegSpec="all",
            unimodConc="all",
            monoIncConc=[1],
            closureConc="all",
            **self._kw,
        )
        mcr_legacy.fit(X, C0)

        constraints = legacy_to_constraints(mcr_legacy)
        mcr_new = MCRALS(constraints=constraints, **self._kw)
        mcr_new.fit(X, C0)
        self._assert_equal(mcr_legacy, mcr_new)


# ---------------------------------------------------------------------------------------
# ModelProfile integration
# ---------------------------------------------------------------------------------------


class TestModelProfile:
    """
    ``ModelProfile`` in the constraints list must produce the same results
    as the equivalent legacy ``hardConc`` / ``getConc``.
    """

    @pytest.fixture(autouse=True)
    def _options(self):
        self._kw = {
            "tol": 1.0e-9,
            "max_iter": 5,
            "nonnegConc": [],
            "nonnegSpec": [],
            "unimodConc": [],
        }

    def test_model_profile_all_components(self, pr4_data):
        X, C0 = pr4_data

        def my_model(C):
            return np.full(C.shape, 0.5)

        mcr_legacy = MCRALS(hardConc=[0, 1], getConc=my_model, **self._kw)
        mcr_legacy.fit(X, C0)

        mcr_new = MCRALS(
            constraints=[ModelProfile("C", model=my_model)],
            tol=1.0e-9,
            max_iter=5,
        )
        mcr_new.fit(X, C0)

        np.testing.assert_allclose(
            np.asarray(mcr_legacy.C.data),
            np.asarray(mcr_new.C.data),
            rtol=1.0e-10,
        )
        np.testing.assert_allclose(
            np.asarray(mcr_legacy.St.data),
            np.asarray(mcr_new.St.data),
            rtol=1.0e-10,
        )

    def test_model_profile_explicit_components(self, pr4_data):
        X, C0 = pr4_data

        def my_model(C):
            return np.full((C.shape[0], 2), 0.5)

        mcr_legacy = MCRALS(hardConc=[0, 1], getConc=my_model, **self._kw)
        mcr_legacy.fit(X, C0)

        mcr_new = MCRALS(
            constraints=[ModelProfile("C", components=[0, 1], model=my_model)],
            tol=1.0e-9,
            max_iter=5,
        )
        mcr_new.fit(X, C0)

        np.testing.assert_allclose(
            np.asarray(mcr_legacy.C.data),
            np.asarray(mcr_new.C.data),
            rtol=1.0e-10,
        )
        np.testing.assert_allclose(
            np.asarray(mcr_legacy.St.data),
            np.asarray(mcr_new.St.data),
            rtol=1.0e-10,
        )
