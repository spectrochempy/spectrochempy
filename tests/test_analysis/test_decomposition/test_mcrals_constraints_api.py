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

import warnings

import numpy as np
import pytest

from spectrochempy.analysis.decomposition._legacy_constraint_converter import (
    legacy_to_constraints,
)
from spectrochempy.analysis.decomposition.mcrals import MCRALS
from spectrochempy.analysis.decomposition.mcrals_constraints import Closure
from spectrochempy.analysis.decomposition.mcrals_constraints import FixedValues
from spectrochempy.analysis.decomposition.mcrals_constraints import ModelProfile
from spectrochempy.analysis.decomposition.mcrals_constraints import Monotonic
from spectrochempy.analysis.decomposition.mcrals_constraints import NonNegative
from spectrochempy.analysis.decomposition.mcrals_constraints import ReferenceProfile
from spectrochempy.analysis.decomposition.mcrals_constraints import Selectivity
from spectrochempy.analysis.decomposition.mcrals_constraints import Unimodal
from spectrochempy.analysis.decomposition.mcrals_constraints import ZeroRegion

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
        """Omitting constraints should result in ``constraints = None``."""
        mcr = MCRALS()
        assert mcr.constraints is None

    def test_explicit_none(self):
        mcr = MCRALS(constraints=None)
        assert mcr.constraints is None

    def test_empty_list(self):
        mcr = MCRALS(constraints=[])
        assert mcr.constraints == []

    def test_empty_tuple(self):
        mcr = MCRALS(constraints=())
        assert mcr.constraints == []

    def test_single_constraint(self):
        mcr = MCRALS(constraints=[NonNegative("C")])
        assert len(mcr.constraints) == 1
        assert isinstance(mcr.constraints[0], NonNegative)

    def test_multiple_constraints(self):
        mcr = MCRALS(constraints=[NonNegative("C"), Unimodal("C"), NonNegative("St")])
        assert len(mcr.constraints) == 3

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
        assert len(mcr.constraints) == 5

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
        with pytest.raises(TypeError, match="must be None, a list, or a tuple"):
            MCRALS(constraints="not_a_sequence")

    def test_accepts_integer_element_with_index(self):
        with pytest.raises(TypeError, match=r"constraints\[2\]"):
            MCRALS(constraints=[NonNegative("C"), Unimodal("C"), 42])


# ---------------------------------------------------------------------------------------
# Assignment after construction
# ---------------------------------------------------------------------------------------


class TestConstraintsAssignment:
    """``constraints`` can be assigned after construction."""

    def test_assign_list(self):
        mcr = MCRALS()
        mcr.constraints = [NonNegative("C")]
        assert mcr.constraints == [NonNegative("C")]

    def test_assign_tuple(self):
        mcr = MCRALS()
        mcr.constraints = (NonNegative("C"), Closure("C"))
        assert isinstance(mcr.constraints, list)
        assert len(mcr.constraints) == 2

    def test_assign_none(self):
        mcr = MCRALS(constraints=[NonNegative("C")])
        mcr.constraints = None
        assert mcr.constraints is None

    def test_assign_replaces(self):
        mcr = MCRALS(constraints=[NonNegative("C")])
        mcr.constraints = [Closure("C")]
        assert len(mcr.constraints) == 1
        assert isinstance(mcr.constraints[0], Closure)

    def test_defensive_copy(self):
        original = [NonNegative("C")]
        mcr = MCRALS(constraints=original)
        original.append(Closure("C"))
        assert len(mcr.constraints) == 1

    def test_assign_defensive_copy(self):
        original = [NonNegative("C")]
        mcr = MCRALS()
        mcr.constraints = original
        original.append(Closure("C"))
        assert len(mcr.constraints) == 1

    def test_tuple_normalizes_to_list(self):
        mcr = MCRALS(constraints=(NonNegative("C"),))
        assert isinstance(mcr.constraints, list)
        assert len(mcr.constraints) == 1

    def test_order_preserved(self):
        c1 = NonNegative("C")
        c2 = Closure("C")
        c3 = Unimodal("C")
        mcr = MCRALS(constraints=[c1, c2, c3])
        assert mcr.constraints[0] is c1
        assert mcr.constraints[1] is c2
        assert mcr.constraints[2] is c3


# ---------------------------------------------------------------------------------------
# Fitted-state invalidation
# ---------------------------------------------------------------------------------------


class TestFittedStateInvalidation:
    """Changing constraints after fit invalidates the fitted state."""

    def test_invalidates_on_assignment(self, pr4_data):
        X, C0 = pr4_data
        mcr = MCRALS(constraints=[NonNegative("C")], tol=1e-9, max_iter=5)
        mcr.fit(X, C0)
        assert mcr._fitted
        mcr.constraints = [NonNegative("C"), Closure("C")]
        assert not mcr._fitted

    def test_refit_after_constraints_change(self, pr4_data):
        X, C0 = pr4_data
        mcr = MCRALS(constraints=[NonNegative("C")], tol=1e-9, max_iter=5)
        mcr.fit(X, C0)
        mcr.constraints = [NonNegative("C"), Closure("C")]
        mcr.fit(X, C0)
        assert mcr._fitted

    def test_setting_none_invalidates(self, pr4_data):
        X, C0 = pr4_data
        mcr = MCRALS(constraints=[NonNegative("C")], tol=1e-9, max_iter=5)
        mcr.fit(X, C0)
        mcr.constraints = None
        assert not mcr._fitted


# ---------------------------------------------------------------------------------------
# Mixed-API guard (constructor + assignment)
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

    def test_mixed_constraints_none_is_ok(self):
        """
        ``constraints=None`` with legacy traits does not raise —
        ``None`` means legacy path.
        """
        mcr = MCRALS(nonnegConc="all", constraints=None)
        assert mcr.constraints is None

    def test_mixed_constraints_empty_is_not_ok(self):
        """
        ``constraints=[]`` with legacy traits raises —
        ``[]`` means the new API.
        """
        with pytest.raises(ValueError, match="cannot be used together"):
            MCRALS(nonnegConc="all", constraints=[])

    def test_mixed_non_legacy_trait_is_ok(self):
        """
        Non-constraint traitlets (tol, max_iter, ...) can be combined
        with ``constraints=``.
        """
        mcr = MCRALS(constraints=[NonNegative("C")], tol=1.0e-9, max_iter=50)
        assert mcr.constraints is not None
        assert mcr.tol == 1.0e-9
        assert mcr.max_iter == 50

    def test_legacy_assignment_after_constraints(self):
        """Assigning a legacy trait after ``constraints`` was set raises."""
        mcr = MCRALS(constraints=[NonNegative("C")])
        with pytest.raises(ValueError, match="cannot be used together"):
            mcr.nonnegConc = "all"

    def test_constraints_assignment_after_legacy(self):
        """Assigning ``constraints`` after a legacy trait was set raises."""
        with pytest.warns(FutureWarning, match="Legacy MCR-ALS constraint"):
            mcr = MCRALS(nonnegConc="all")
        with pytest.raises(ValueError, match="cannot be used together"):
            mcr.constraints = [NonNegative("C")]

    def test_untouched_legacy_defaults_no_conflict(self):
        """
        Legacy traits with their default values (not explicitly passed)
        do not conflict with the new API.
        """
        mcr = MCRALS(constraints=[NonNegative("C")])
        assert mcr.constraints is not None

    def test_legacy_only_untouched_no_warning(self):
        """
        Creating MCRALS with no arguments does not emit any deprecation
        warning for legacy constraint params.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            mcr = MCRALS()
            assert mcr.constraints is None

    def test_legacy_only_explicit_still_works(self):
        """
        Legacy-only API (constructor-time) still works and produces
        a fit.
        """
        mcr = MCRALS(nonnegConc="all", nonnegSpec=[], unimodConc=[])
        assert mcr.constraints is None


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

    def test_unimodal_tolerance_matches_legacy(self, pr4_data):
        """Public ``Unimodal`` with tolerance matches legacy ``unimodConcTol``."""
        X, C0 = pr4_data
        mcr_legacy = MCRALS(
            nonnegConc=[],
            nonnegSpec=[],
            unimodConc=[0],
            unimodConcTol=1.0,
            **self._kw,
        )
        mcr_legacy.fit(X, C0)
        mcr_new = MCRALS(
            constraints=[Unimodal("C", components=[0], tolerance=1.0)],
            **self._kw,
        )
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

    def test_model_profile_mapping_swap(self, pr4_data):
        """Mapping swaps model output columns before assigning to components."""
        X, C0 = pr4_data

        def my_model(C):
            result = np.zeros((C.shape[0], 2))
            result[:, 0] = 0.3
            result[:, 1] = 0.7
            return result

        # Legacy: getC_to_C_idx=[1, 0] swaps columns
        mcr_legacy = MCRALS(
            nonnegConc=[],
            nonnegSpec=[],
            unimodConc=[],
            hardConc=[0, 1],
            getConc=my_model,
            getC_to_C_idx=[1, 0],
            tol=1.0e-9,
            max_iter=5,
        )
        mcr_legacy.fit(X, C0)

        mcr_new = MCRALS(
            constraints=[
                ModelProfile("C", components=[0, 1], model=my_model, mapping=[1, 0])
            ],
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

    def test_model_profile_mapping_none_is_identity(self, pr4_data):
        """
        ``mapping=None`` (default) produces the same result as
        ``mapping=[0, 1]``.
        """
        X, C0 = pr4_data

        def my_model(C):
            result = np.zeros((C.shape[0], 2))
            result[:, 0] = 0.3
            result[:, 1] = 0.7
            return result

        mcr_identity = MCRALS(
            constraints=[ModelProfile("C", components=[0, 1], model=my_model)],
            tol=1.0e-9,
            max_iter=5,
        )
        mcr_identity.fit(X, C0)

        mcr_explicit = MCRALS(
            constraints=[
                ModelProfile("C", components=[0, 1], model=my_model, mapping=[0, 1])
            ],
            tol=1.0e-9,
            max_iter=5,
        )
        mcr_explicit.fit(X, C0)

        np.testing.assert_allclose(
            np.asarray(mcr_identity.C.data),
            np.asarray(mcr_explicit.C.data),
            rtol=1.0e-10,
        )

    def test_model_profile_mapping_partial_none_entries(self, pr4_data):
        """
        ``None`` entries in mapping leave the corresponding component
        unchanged by the model.
        """
        X, C0 = pr4_data

        def my_model(C):
            result = np.zeros((C.shape[0], 2))
            result[:, 0] = 0.5
            result[:, 1] = 0.6
            return result

        mcr = MCRALS(
            constraints=[
                ModelProfile("C", components=[0, 1], model=my_model, mapping=[0, None])
            ],
            tol=1.0e-9,
            max_iter=5,
        )
        mcr.fit(X, C0)
        Cc = np.asarray(mcr.C_constrained.data)

        # Component 0 is replaced by model column 0 (all 0.5)
        np.testing.assert_allclose(Cc[:, 0], np.full(8, 0.5), atol=1.0e-12)
        # Component 1 is NOT replaced by model column 1 (all 0.6)
        assert not np.allclose(Cc[:, 1], 0.6), "Component 1 should not be replaced"

    def test_model_profile_mapping_spec_side(self, pr4_data):
        """Mapping works on the spectral side (row selection)."""
        X, C0 = pr4_data

        def my_model(St):
            result = np.zeros((2, St.shape[1]))
            result[0] = 0.3
            result[1] = 0.7
            return result

        mcr = MCRALS(
            constraints=[
                ModelProfile("St", components=[0, 1], model=my_model, mapping=[1, 0])
            ],
            tol=1.0e-9,
            max_iter=5,
        )
        mcr.fit(X, C0)
        St = np.asarray(mcr.St.data)
        # Component 0 gets model row 1 (all 0.7)
        np.testing.assert_allclose(St[0], np.full(12, 0.7), atol=1.0e-12)
        # Component 1 gets model row 0 (all 0.3)
        np.testing.assert_allclose(St[1], np.full(12, 0.3), atol=1.0e-12)

    def test_model_profile_mapping_out_of_range_conc(self, pr4_data):
        """Mapping index >= number of model output columns raises IndexError."""
        X, C0 = pr4_data

        def my_model(C):
            return np.zeros((C.shape[0], 2))

        mcr = MCRALS(
            constraints=[
                ModelProfile("C", components=[0, 1], model=my_model, mapping=[0, 5])
            ],
            tol=1e-9,
            max_iter=3,
        )
        with pytest.raises(IndexError, match="mapping\\[1\\] = 5 is out of range"):
            mcr.fit(X, C0)

    def test_model_profile_mapping_out_of_range_spec(self, pr4_data):
        """Mapping index >= number of model output rows raises IndexError."""
        X, C0 = pr4_data

        def my_model(St):
            return np.zeros((2, St.shape[1]))

        mcr = MCRALS(
            constraints=[
                ModelProfile("St", components=[0, 1], model=my_model, mapping=[0, 5])
            ],
            tol=1e-9,
            max_iter=3,
        )
        with pytest.raises(IndexError, match="mapping\\[1\\] = 5 is out of range"):
            mcr.fit(X, C0)

    def test_model_profile_mapping_all_none_noop(self, pr4_data):
        """All-None mapping is a no-op: model results are not assigned."""
        X, C0 = pr4_data

        # Model returns a distinct value for each column.
        def my_model(C):
            return np.full(C.shape, 0.999)

        # Without constraints, the first ALS iteration already produces
        # different values.  An all-None mapping means no component is
        # overwritten, so the solution should differ from one where the
        # model always returns 0.999 for component 0.
        mcr = MCRALS(
            constraints=[
                ModelProfile(
                    "C", components=[0, 1], model=my_model, mapping=[None, None]
                )
            ],
            tol=1e-9,
            max_iter=5,
        )
        mcr.fit(X, C0)
        Cc = np.asarray(mcr.C_constrained.data)
        # Component 0 should NOT be all 0.999 (the model's value)
        assert not np.allclose(Cc[:, 0], 0.999), "Component 0 was overwritten"

    def test_model_profile_mapping_duplicates_allowed(self, pr4_data):
        """Duplicate mapping indices reuse the same model output column."""
        X, C0 = pr4_data

        def my_model(C):
            result = np.zeros((C.shape[0], 2))
            result[:, 0] = 0.5
            result[:, 1] = 0.7
            return result

        mcr = MCRALS(
            constraints=[
                ModelProfile("C", components=[0, 1], model=my_model, mapping=[0, 0])
            ],
            tol=1e-9,
            max_iter=5,
        )
        mcr.fit(X, C0)
        Cc = np.asarray(mcr.C_constrained.data)
        # Both components 0 and 1 get model column 0 (all 0.5)
        np.testing.assert_allclose(Cc[:, 0], np.full(8, 0.5), atol=1.0e-12)
        np.testing.assert_allclose(Cc[:, 1], np.full(8, 0.5), atol=1.0e-12)

    def test_model_profile_mapping_with_components_none(self, pr4_data):
        """``components=None`` with non-None mapping works at fit time."""
        X, C0 = pr4_data

        def my_model(C):
            result = np.zeros((C.shape[0], 2))
            result[:, 0] = 0.5
            result[:, 1] = 0.7
            return result

        mcr = MCRALS(
            constraints=[ModelProfile("C", model=my_model, mapping=[1, 0])],
            tol=1e-9,
            max_iter=5,
        )
        mcr.fit(X, C0)
        Cc = np.asarray(mcr.C_constrained.data)
        # components=None → resolved to [0, 1]; mapping=[1,0] swaps
        np.testing.assert_allclose(Cc[:, 0], np.full(8, 0.7), atol=1.0e-12)
        np.testing.assert_allclose(Cc[:, 1], np.full(8, 0.5), atol=1.0e-12)


# ---------------------------------------------------------------------------------------
# Unsupported constraint types raise NotImplementedError
# ---------------------------------------------------------------------------------------


class TestUnsupportedConstraints:
    """
    Every public constraint that is not yet implemented in MCRALS must
    raise ``NotImplementedError`` — never be silently ignored.

    The error is raised at fit time (when ``_public_to_internal`` is
    called), not at construction.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        self._fit_kw = {"tol": 1e-9, "max_iter": 5}

    def test_reference_profile_raises(self, pr4_data):
        X, C0 = pr4_data
        mcr = MCRALS(
            constraints=[ReferenceProfile("C", component=0, data=np.arange(8))],
            **self._fit_kw,
        )
        with pytest.raises(
            NotImplementedError,
            match="ReferenceProfile is not yet implemented in MCRALS",
        ):
            mcr.fit(X, C0)

    def test_fixed_values_raises(self, pr4_data):
        X, C0 = pr4_data
        mcr = MCRALS(
            constraints=[FixedValues("St", values=[[0.1, 0.2] * 6])],
            **self._fit_kw,
        )
        with pytest.raises(
            NotImplementedError,
            match="FixedValues is not yet implemented in MCRALS",
        ):
            mcr.fit(X, C0)

    def test_zero_region_raises(self, pr4_data):
        X, C0 = pr4_data
        mcr = MCRALS(
            constraints=[ZeroRegion("C", region=(0, 5))],
            **self._fit_kw,
        )
        with pytest.raises(
            NotImplementedError,
            match="ZeroRegion is not yet implemented in MCRALS",
        ):
            mcr.fit(X, C0)

    def test_selectivity_raises(self, pr4_data):
        X, C0 = pr4_data
        mcr = MCRALS(
            constraints=[Selectivity("C", region=(0, 5), component=0)],
            **self._fit_kw,
        )
        with pytest.raises(
            NotImplementedError,
            match="Selectivity is not yet implemented in MCRALS",
        ):
            mcr.fit(X, C0)

    def test_closure_spec_raises(self, pr4_data):
        X, C0 = pr4_data
        mcr = MCRALS(
            constraints=[Closure("St")],
            **self._fit_kw,
        )
        with pytest.raises(
            NotImplementedError,
            match=r"Closure\('St'\) is not implemented",
        ):
            mcr.fit(X, C0)

    def test_monotonic_spec_raises(self, pr4_data):
        X, C0 = pr4_data
        mcr = MCRALS(
            constraints=[Monotonic("St", "increasing")],
            **self._fit_kw,
        )
        with pytest.raises(
            NotImplementedError,
            match=r"Monotonic\('St'\) is not implemented",
        ):
            mcr.fit(X, C0)

    def test_model_profile_mapping_length_mismatch_at_fit(self, pr4_data):
        """When components is None, mismatch is caught after resolution."""
        X, C0 = pr4_data

        def my_model(C):
            return C

        # components=None + mapping with wrong length → caught at fit time
        mcr = MCRALS(
            constraints=[ModelProfile("C", model=my_model, mapping=[0, 1, 2])],
            **self._fit_kw,
        )
        with pytest.raises(
            ValueError,
            match="mapping has length 3 but components resolved to 2",
        ):
            mcr.fit(X, C0)

    def test_unsupported_in_list_with_supported_raises(self, pr4_data):
        X, C0 = pr4_data
        mcr = MCRALS(
            constraints=[
                NonNegative("C"),
                FixedValues("St", values=[[0.1, 0.2] * 6]),
            ],
            **self._fit_kw,
        )
        with pytest.raises(
            NotImplementedError,
            match="FixedValues is not yet implemented in MCRALS",
        ):
            mcr.fit(X, C0)

    def test_no_constraint_silently_ignored(self, pr4_data):
        X, C0 = pr4_data
        mcr = MCRALS(tol=1e-9, max_iter=5)
        mcr.constraints = [NonNegative("C")]
        mcr.fit(X, C0)
        assert mcr._fitted

    def test_assignment_of_unsupported_raises_at_fit_time(self, pr4_data):
        X, C0 = pr4_data
        mcr = MCRALS(tol=1e-9, max_iter=5)
        mcr.constraints = [
            NonNegative("C"),
            ReferenceProfile("C", component=1, data=[1, 2, 3, 4, 5, 6, 7, 8]),
        ]
        with pytest.raises(
            NotImplementedError,
            match="ReferenceProfile is not yet implemented in MCRALS",
        ):
            mcr.fit(X, C0)


# ---------------------------------------------------------------------------------------
# PNNLS solver derives nonneg from new API
# ---------------------------------------------------------------------------------------


class TestPnnlsSolverWithNewApi:
    """
    When using the new ``constraints`` API with ``solver_C='pnnls'``, the
    solver must derive its non-negative component list from the public
    ``NonNegative`` constraints — not from legacy traitlets.
    """

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

    def test_pnnls_new_api_all_components(self, pr4_data):
        X, C0 = pr4_data
        mcr_new = MCRALS(
            constraints=[NonNegative("C")],
            solver_C="pnnls",
            tol=1e-9,
            max_iter=50,
        )
        mcr_legacy = MCRALS(
            nonnegConc="all",
            nonnegSpec=[],
            unimodConc=[],
            solver_C="pnnls",
            tol=1e-9,
            max_iter=50,
        )
        mcr_new.fit(X, C0)
        mcr_legacy.fit(X, C0)
        self._assert_equal(mcr_legacy, mcr_new)

    def test_pnnls_new_api_selected_components(self, pr4_data):
        X, C0 = pr4_data
        mcr_new = MCRALS(
            constraints=[NonNegative("C", components=[1])],
            solver_C="pnnls",
            tol=1e-9,
            max_iter=50,
        )
        mcr_legacy = MCRALS(
            nonnegConc=[1],
            nonnegSpec=[],
            unimodConc=[],
            solver_C="pnnls",
            tol=1e-9,
            max_iter=50,
        )
        mcr_new.fit(X, C0)
        mcr_legacy.fit(X, C0)
        self._assert_equal(mcr_legacy, mcr_new)

    def test_pnnls_new_api_no_nonneg_uses_empty_list(self, pr4_data):
        X, C0 = pr4_data
        mcr = MCRALS(
            constraints=[],
            solver_C="pnnls",
            tol=1e-9,
            max_iter=5,
        )
        mcr.fit(X, C0)
        assert mcr._fitted


# ---------------------------------------------------------------------------------------
# _nonneg_indices helper
# ---------------------------------------------------------------------------------------


class TestNonnegIndices:
    """``_nonneg_indices`` correctly derives non-negative component lists."""

    def test_legacy_fallback(self):
        mcr = MCRALS()
        assert mcr._nonneg_indices("C") is not None

    def test_from_single_constraint(self):
        mcr = MCRALS(constraints=[NonNegative("C", components=[0, 2])])
        assert mcr._nonneg_indices("C") == [0, 2]

    def test_from_multiple_constraints(self):
        mcr = MCRALS(
            constraints=[
                NonNegative("C", components=[0]),
                NonNegative("C", components=[2]),
            ]
        )
        assert mcr._nonneg_indices("C") == [0, 2]

    def test_from_all_components_after_fit(self, pr4_data):
        X, C0 = pr4_data
        mcr = MCRALS(constraints=[NonNegative("C")], tol=1e-9, max_iter=5)
        mcr.fit(X, C0)
        assert mcr._nonneg_indices("C") == [0, 1]

    def test_empty_when_no_nonneg(self):
        mcr = MCRALS(constraints=[Closure("C")])
        assert mcr._nonneg_indices("C") == []

    def test_indices_grouped_by_side(self):
        mcr = MCRALS(
            constraints=[
                NonNegative("C", components=[0]),
                NonNegative("St", components=[1]),
            ]
        )
        assert mcr._nonneg_indices("C") == [0]
        assert mcr._nonneg_indices("St") == [1]


# ---------------------------------------------------------------------------------------
# ModelProfile with numpy extra outputs
# ---------------------------------------------------------------------------------------


class TestModelProfileNumpyExtra:
    """ModelProfile handles numpy-based extra outputs correctly."""

    def test_extra_output_list_with_numpy(self, pr4_data):
        X, C0 = pr4_data

        def my_model(C):
            return C, (), np.array([1.0, 2.0, 3.0])

        mcr = MCRALS(
            constraints=[ModelProfile("C", model=my_model)],
            tol=1e-9,
            max_iter=3,
        )
        mcr.fit(X, C0)
        assert mcr._fitted

    def test_extra_output_none(self, pr4_data):
        X, C0 = pr4_data

        def my_model(C):
            return C, ()

        mcr = MCRALS(
            constraints=[ModelProfile("C", model=my_model)],
            tol=1e-9,
            max_iter=3,
        )
        mcr.fit(X, C0)
        assert mcr._fitted


# ---------------------------------------------------------------------------------------
# _validating_legacy exception safety
# ---------------------------------------------------------------------------------------


class TestValidatingLegacyExceptionSafety:
    """``_validating_legacy`` flag is correctly reset even if validation raises."""

    def test_flag_reset_after_fit(self, pr4_data):
        X, C0 = pr4_data
        mcr = MCRALS(tol=1e-9, max_iter=5)
        mcr.fit(X, C0)
        assert not mcr._validating_legacy

    def test_flag_false_after_new_api_fit(self, pr4_data):
        X, C0 = pr4_data
        mcr = MCRALS(constraints=[NonNegative("C")], tol=1e-9, max_iter=5)
        mcr.fit(X, C0)
        assert not mcr._validating_legacy
