# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: N815
"""
Tests for the private legacy traitlet-to-constraint converter.

These tests verify that each existing MCRALS constraint-related traitlet
is translated into the expected public Constraint object with the correct
attributes.  No numerical behaviour is tested here.
"""

import numpy as np

from spectrochempy.analysis.decomposition._legacy_constraint_converter import (
    legacy_to_constraints,
)
from spectrochempy.analysis.decomposition.mcrals_constraints import Closure
from spectrochempy.analysis.decomposition.mcrals_constraints import ModelProfile
from spectrochempy.analysis.decomposition.mcrals_constraints import Monotonic
from spectrochempy.analysis.decomposition.mcrals_constraints import NonNegative
from spectrochempy.analysis.decomposition.mcrals_constraints import Unimodal

# -----------------------------------------------------------------------------------
# Fixtures: estimator-like objects that mimic the relevant traitlet API
# -----------------------------------------------------------------------------------


class _FakeEstimator:
    """
    Minimal stub that exposes the traitlets that the converter reads.

    This avoids instantiating the full ``MCRALS`` estimator (which requires
    a complex dependency chain).  Each parameter is a plain attribute.
    """

    def __init__(self, **kwargs):
        defaults = {
            "nonnegConc": "all",
            "unimodConc": "all",
            "unimodConcMod": "strict",
            "unimodConcTol": 1.1,
            "monoIncConc": [],
            "monoIncTol": 1.1,
            "monoDecConc": [],
            "monoDecTol": 1.1,
            "closureConc": [],
            "closureMethod": "scaling",
            "closureTarget": "default",
            "hardConc": [],
            "getConc": None,
            "argsGetConc": (),
            "kwargsGetConc": {},
            "getC_to_C_idx": "default",
            "nonnegSpec": "all",
            "unimodSpec": [],
            "unimodSpecMod": "strict",
            "unimodSpecTol": 1.1,
            "hardSpec": [],
            "getSpec": None,
            "argsGetSpec": (),
            "kwargsGetSpec": {},
            "getSt_to_St_idx": "default",
            "normSpec": None,
        }
        defaults.update(kwargs)
        for key, val in defaults.items():
            setattr(self, key, val)


# -----------------------------------------------------------------------------------
# Full conversion
# -----------------------------------------------------------------------------------


class TestLegacyToConstraints:
    """Integration tests for the top-level ``legacy_to_constraints`` function."""

    def test_default_configuration(self):
        """
        Default unfitted estimator produces the always-active constraints.

        Expected (with all defaults):
          - NonNegative("C", components=None)   [nonnegConc="all"]
          - Unimodal("C", components=None)       [unimodConc="all"]
          - NonNegative("St", components=None)   [nonnegSpec="all"]
        """
        est = _FakeEstimator()
        constraints = legacy_to_constraints(est)

        assert len(constraints) == 3

        assert isinstance(constraints[0], NonNegative)
        assert constraints[0].profile == "C"
        assert constraints[0].components is None

        assert isinstance(constraints[1], Unimodal)
        assert constraints[1].profile == "C"
        assert constraints[1].components is None
        assert constraints[1].mod == "strict"

        assert isinstance(constraints[2], NonNegative)
        assert constraints[2].profile == "St"
        assert constraints[2].components is None

    def test_empty_nonneg_conc(self):
        """``nonnegConc=[]`` suppresses the NonNegative(C) constraint."""
        est = _FakeEstimator(nonnegConc=[])
        constraints = legacy_to_constraints(est)

        c_nn = [
            c for c in constraints if isinstance(c, NonNegative) and c.profile == "C"
        ]
        assert len(c_nn) == 0

    def test_nonneg_conc_explicit_components(self):
        est = _FakeEstimator(nonnegConc=[0, 2])
        constraints = legacy_to_constraints(est)

        nn = [c for c in constraints if isinstance(c, NonNegative) and c.profile == "C"]
        assert len(nn) == 1
        assert nn[0].components == [0, 2]

    def test_unimod_conc_disabled(self):
        est = _FakeEstimator(unimodConc=[])
        constraints = legacy_to_constraints(est)

        um = [c for c in constraints if isinstance(c, Unimodal) and c.profile == "C"]
        assert len(um) == 0

    def test_unimod_conc_all_with_tolerance(self):
        """``unimodConc="all"`` with custom tolerance propagates correctly."""
        est = _FakeEstimator(unimodConc="all", unimodConcTol=2.0)
        constraints = legacy_to_constraints(est)

        um = [c for c in constraints if isinstance(c, Unimodal) and c.profile == "C"]
        assert len(um) == 1
        assert um[0].tolerance == 2.0
        assert um[0].components is None

    def test_unimod_conc_smooth_mod(self):
        est = _FakeEstimator(unimodConc=[0], unimodConcMod="smooth")
        constraints = legacy_to_constraints(est)

        um = [c for c in constraints if isinstance(c, Unimodal) and c.profile == "C"]
        assert len(um) == 1
        assert um[0].components == [0]
        assert um[0].mod == "smooth"

    def test_unimod_conc_strict_mod_default(self):
        est = _FakeEstimator(unimodConc=[0, 1])
        constraints = legacy_to_constraints(est)

        um = [c for c in constraints if isinstance(c, Unimodal) and c.profile == "C"]
        assert len(um) == 1
        assert um[0].mod == "strict"

    def test_unimod_conc_tolerance_default(self):
        est = _FakeEstimator(unimodConc=[0])
        constraints = legacy_to_constraints(est)

        um = [c for c in constraints if isinstance(c, Unimodal) and c.profile == "C"]
        assert len(um) == 1
        assert um[0].tolerance == 1.1

    def test_unimod_conc_tolerance_custom(self):
        est = _FakeEstimator(unimodConc=[0], unimodConcTol=3.0)
        constraints = legacy_to_constraints(est)

        um = [c for c in constraints if isinstance(c, Unimodal) and c.profile == "C"]
        assert len(um) == 1
        assert um[0].tolerance == 3.0

    def test_unimod_spec_tolerance_custom(self):
        est = _FakeEstimator(unimodSpec=[1], unimodSpecTol=2.5)
        constraints = legacy_to_constraints(est)

        um = [c for c in constraints if isinstance(c, Unimodal) and c.profile == "St"]
        assert len(um) == 1
        assert um[0].tolerance == 2.5

    def test_mono_inc_conc(self):
        est = _FakeEstimator(monoIncConc=[1], monoIncTol=1.5)
        constraints = legacy_to_constraints(est)

        mi = [
            c
            for c in constraints
            if isinstance(c, Monotonic)
            and c.profile == "C"
            and c.direction == "increasing"
        ]
        assert len(mi) == 1
        assert mi[0].components == [1]
        assert mi[0].tolerance == 1.5

    def test_mono_dec_conc(self):
        est = _FakeEstimator(monoDecConc=[0], monoDecTol=1.2)
        constraints = legacy_to_constraints(est)

        md = [
            c
            for c in constraints
            if isinstance(c, Monotonic)
            and c.profile == "C"
            and c.direction == "decreasing"
        ]
        assert len(md) == 1
        assert md[0].components == [0]
        assert md[0].tolerance == 1.2

    def test_mono_inc_and_dec_both_active(self):
        est = _FakeEstimator(monoIncConc=[0], monoDecConc=[1])
        constraints = legacy_to_constraints(est)

        monos = [c for c in constraints if isinstance(c, Monotonic)]
        assert len(monos) == 2
        directions = {m.direction for m in monos}
        assert directions == {"increasing", "decreasing"}

    def test_closure_conc_default_target(self):
        """Default ``closureTarget`` should be extracted as ``1.0``."""
        est = _FakeEstimator(closureConc="all", closureTarget="default")
        constraints = legacy_to_constraints(est)

        cc = [c for c in constraints if isinstance(c, Closure)]
        assert len(cc) == 1
        assert cc[0].target == 1.0

    def test_closure_conc_array_target_uniform(self):
        """Uniform array target is passed through directly."""
        arr = np.ones(10)
        est = _FakeEstimator(closureConc=[0, 1], closureTarget=arr)
        constraints = legacy_to_constraints(est)

        cc = [c for c in constraints if isinstance(c, Closure)]
        assert len(cc) == 1
        assert cc[0].target is arr
        np.testing.assert_array_equal(cc[0].target, np.ones(10))

    def test_closure_conc_array_target_non_uniform(self):
        """Non-uniform array target is passed through directly."""
        arr = np.array([2.0, 3.0, 4.0])
        est = _FakeEstimator(closureConc=[0], closureTarget=arr)
        constraints = legacy_to_constraints(est)

        cc = [c for c in constraints if isinstance(c, Closure)]
        assert len(cc) == 1
        assert cc[0].target is arr
        np.testing.assert_array_equal(cc[0].target, arr)

    def test_closure_conc_disabled(self):
        est = _FakeEstimator(closureConc=[])
        constraints = legacy_to_constraints(est)

        cc = [c for c in constraints if isinstance(c, Closure)]
        assert len(cc) == 0

    def test_hard_conc_with_get_conc(self):
        def _fake_model(C):
            return C

        est = _FakeEstimator(hardConc=[0, 1], getConc=_fake_model)
        constraints = legacy_to_constraints(est)

        mp = [
            c for c in constraints if isinstance(c, ModelProfile) and c.profile == "C"
        ]
        assert len(mp) == 1
        assert mp[0].components == [0, 1]
        assert mp[0].model is _fake_model
        assert mp[0].model_args == ()
        assert mp[0].model_kwargs == {}

    def test_hard_conc_with_args_and_kwargs(self):
        def _fake_model(C, a, b, **kwargs):
            return C

        est = _FakeEstimator(
            hardConc=[0, 1],
            getConc=_fake_model,
            argsGetConc=("x", 42),
            kwargsGetConc={"verbose": True},
        )
        constraints = legacy_to_constraints(est)

        mp = [
            c for c in constraints if isinstance(c, ModelProfile) and c.profile == "C"
        ]
        assert len(mp) == 1
        assert mp[0].components == [0, 1]
        assert mp[0].model is _fake_model
        assert mp[0].model_args == ("x", 42)
        assert mp[0].model_kwargs == {"verbose": True}

    def test_hard_conc_disabled(self):
        est = _FakeEstimator(hardConc=[])
        constraints = legacy_to_constraints(est)

        mp = [
            c for c in constraints if isinstance(c, ModelProfile) and c.profile == "C"
        ]
        assert len(mp) == 0

    def test_hard_conc_active_but_no_model(self):
        """
        If ``hardConc`` is non-empty but ``getConc`` is ``None``,
        no ``ModelProfile`` is emitted because there is no model to bind.
        """
        est = _FakeEstimator(hardConc=[0], getConc=None)
        constraints = legacy_to_constraints(est)

        mp = [
            c for c in constraints if isinstance(c, ModelProfile) and c.profile == "C"
        ]
        assert len(mp) == 0

    def test_hard_conc_model_not_callable_skipped(self):
        """If ``getConc`` is not callable, no ``ModelProfile`` is emitted."""
        est = _FakeEstimator(hardConc=[0], getConc="not_callable")
        constraints = legacy_to_constraints(est)

        mp = [
            c for c in constraints if isinstance(c, ModelProfile) and c.profile == "C"
        ]
        assert len(mp) == 0

    def test_hard_conc_mapping_default_is_none(self):
        def _fake_model(C):
            return C

        est = _FakeEstimator(
            hardConc=[0, 1], getConc=_fake_model, getC_to_C_idx="default"
        )
        constraints = legacy_to_constraints(est)

        mp = [
            c for c in constraints if isinstance(c, ModelProfile) and c.profile == "C"
        ]
        assert len(mp) == 1
        assert mp[0].mapping is None

    def test_hard_conc_mapping_identity_is_none(self):
        def _fake_model(C):
            return C

        est = _FakeEstimator(hardConc=[0, 1], getConc=_fake_model, getC_to_C_idx=[0, 1])
        constraints = legacy_to_constraints(est)

        mp = [
            c for c in constraints if isinstance(c, ModelProfile) and c.profile == "C"
        ]
        assert len(mp) == 1
        assert mp[0].mapping is None

    def test_hard_conc_mapping_swap(self):
        def _fake_model(C):
            return C

        est = _FakeEstimator(hardConc=[0, 1], getConc=_fake_model, getC_to_C_idx=[1, 0])
        constraints = legacy_to_constraints(est)

        mp = [
            c for c in constraints if isinstance(c, ModelProfile) and c.profile == "C"
        ]
        assert len(mp) == 1
        assert mp[0].mapping == [1, 0]

    def test_hard_conc_mapping_with_none_entries(self):
        def _fake_model(C):
            return C

        est = _FakeEstimator(
            hardConc=[0, 1, 2], getConc=_fake_model, getC_to_C_idx=[2, None, 0]
        )
        constraints = legacy_to_constraints(est)

        mp = [
            c for c in constraints if isinstance(c, ModelProfile) and c.profile == "C"
        ]
        assert len(mp) == 1
        # legacy mapping: col 0->comp 2, col 1->None, col 2->comp 0
        # new mapping: comp 0<-col 2, comp 1<-None, comp 2<-col 0
        assert mp[0].mapping == [2, None, 0]

    def test_nonneg_spec_components(self):
        est = _FakeEstimator(nonnegSpec=[0, 1])
        constraints = legacy_to_constraints(est)

        nn = [
            c for c in constraints if isinstance(c, NonNegative) and c.profile == "St"
        ]
        assert len(nn) == 1
        assert nn[0].components == [0, 1]

    def test_nonneg_spec_disabled(self):
        est = _FakeEstimator(nonnegSpec=[])
        constraints = legacy_to_constraints(est)

        nn = [
            c for c in constraints if isinstance(c, NonNegative) and c.profile == "St"
        ]
        assert len(nn) == 0

    def test_unimod_spec(self):
        est = _FakeEstimator(unimodSpec=[0], unimodSpecMod="smooth")
        constraints = legacy_to_constraints(est)

        um = [c for c in constraints if isinstance(c, Unimodal) and c.profile == "St"]
        assert len(um) == 1
        assert um[0].components == [0]
        assert um[0].mod == "smooth"

    def test_unimod_spec_disabled(self):
        est = _FakeEstimator(unimodSpec=[])
        constraints = legacy_to_constraints(est)

        um = [c for c in constraints if isinstance(c, Unimodal) and c.profile == "St"]
        assert len(um) == 0

    def test_hard_spec_with_get_spec(self):
        def _fake_model(St):
            return St

        est = _FakeEstimator(hardSpec=[2], getSpec=_fake_model)
        constraints = legacy_to_constraints(est)

        mp = [
            c for c in constraints if isinstance(c, ModelProfile) and c.profile == "St"
        ]
        assert len(mp) == 1
        assert mp[0].components == [2]
        assert mp[0].model is _fake_model
        assert mp[0].model_args == ()
        assert mp[0].model_kwargs == {}

    def test_hard_spec_with_args_and_kwargs(self):
        def _fake_model(St, tol, **kw):
            return St

        est = _FakeEstimator(
            hardSpec=[0],
            getSpec=_fake_model,
            argsGetSpec=(0.5,),
            kwargsGetSpec={"max_iter": 10},
        )
        constraints = legacy_to_constraints(est)

        mp = [
            c for c in constraints if isinstance(c, ModelProfile) and c.profile == "St"
        ]
        assert len(mp) == 1
        assert mp[0].components == [0]
        assert mp[0].model is _fake_model
        assert mp[0].model_args == (0.5,)
        assert mp[0].model_kwargs == {"max_iter": 10}

    def test_hard_spec_disabled(self):
        est = _FakeEstimator(hardSpec=[])
        constraints = legacy_to_constraints(est)

        mp = [
            c for c in constraints if isinstance(c, ModelProfile) and c.profile == "St"
        ]
        assert len(mp) == 0

    def test_hard_spec_mapping_default_is_none(self):
        def _fake_model(St):
            return St

        est = _FakeEstimator(
            hardSpec=[0, 1], getSpec=_fake_model, getSt_to_St_idx="default"
        )
        constraints = legacy_to_constraints(est)

        mp = [
            c for c in constraints if isinstance(c, ModelProfile) and c.profile == "St"
        ]
        assert len(mp) == 1
        assert mp[0].mapping is None

    def test_hard_spec_mapping_swap(self):
        def _fake_model(St):
            return St

        est = _FakeEstimator(
            hardSpec=[0, 1], getSpec=_fake_model, getSt_to_St_idx=[1, 0]
        )
        constraints = legacy_to_constraints(est)

        mp = [
            c for c in constraints if isinstance(c, ModelProfile) and c.profile == "St"
        ]
        assert len(mp) == 1
        assert mp[0].mapping == [1, 0]

    def test_all_conc_constraints_active(self):
        """All concentration-side constraints active simultaneously."""

        def _fake_model(C, scale, **kw):
            return C

        est = _FakeEstimator(
            nonnegConc="all",
            unimodConc=[0, 1],
            unimodConcMod="smooth",
            monoIncConc=[0],
            monoDecConc=[1],
            closureConc="all",
            closureTarget="default",
            hardConc=[2],
            getConc=_fake_model,
            argsGetConc=(1.0,),
            kwargsGetConc={"clip": True},
        )
        constraints = legacy_to_constraints(est)

        # Check the ModelProfile carries args/kwargs
        mp = [c for c in constraints if isinstance(c, ModelProfile)]
        assert len(mp) >= 1
        conc_mp = [c for c in mp if c.profile == "C"]
        assert len(conc_mp) == 1
        assert conc_mp[0].model_args == (1.0,)
        assert conc_mp[0].model_kwargs == {"clip": True}

        # Count by type on the conc side
        conc_types = [
            "NonNegative",
            "Unimodal",
            "Monotonic",
            "Monotonic",
            "Closure",
            "ModelProfile",
        ]

        type_names = [type(c).__name__ for c in constraints]

        for t in conc_types:
            assert t in type_names, f"Missing {t}"

    def test_not_connected_to_engine(self):
        """
        Creating constraints via the converter must not affect the estimator.

        This is a smoke test that the converter is purely declarative.
        """
        est = _FakeEstimator()
        original = est.nonnegConc
        _ = legacy_to_constraints(est)
        assert est.nonnegConc == original

    def test_preserves_public_api_contract(self):
        """
        The converter must only produce public Constraint subclasses
        (NonNegative, Closure, Unimodal, Monotonic, ModelProfile).
        """
        est = _FakeEstimator(
            nonnegConc="all",
            unimodConc=[0],
            monoIncConc=[1],
            closureConc="all",
            hardConc=[2],
            getConc=lambda C: C,
            nonnegSpec="all",
        )
        constraints = legacy_to_constraints(est)

        allowed_types = {NonNegative, Closure, Unimodal, Monotonic, ModelProfile}
        for c in constraints:
            assert type(c) in allowed_types, f"Unexpected constraint type: {type(c)}"


# -----------------------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------------------


class TestEdgeCases:
    def test_norm_spec_ignored(self):
        """``normSpec`` is not represented as a public constraint."""
        est = _FakeEstimator(normSpec="max")
        constraints = legacy_to_constraints(est)
        default_est = _FakeEstimator()
        assert len(constraints) == len(legacy_to_constraints(default_est))

    def test_closure_method_translated(self):
        """``closureMethod`` is now translated into the ``method`` parameter."""
        est_scaling = _FakeEstimator(closureConc="all", closureMethod="scaling")
        est_constsum = _FakeEstimator(closureConc="all", closureMethod="constantSum")
        c1 = legacy_to_constraints(est_scaling)
        c2 = legacy_to_constraints(est_constsum)
        closure1 = [c for c in c1 if isinstance(c, Closure)]
        closure2 = [c for c in c2 if isinstance(c, Closure)]
        assert len(closure1) == 1
        assert len(closure2) == 1
        assert closure1[0].method == "scaling"
        assert closure2[0].method == "constantSum"
        assert closure1 != closure2

    def test_unimod_tol_translated(self):
        """
        ``unimodConcTol`` is now translated into the ``tolerance`` parameter
        of public ``Unimodal``.
        """
        est = _FakeEstimator(unimodConc="all", unimodConcTol=2.0)
        constraints = legacy_to_constraints(est)
        um = [c for c in constraints if isinstance(c, Unimodal)]
        assert len(um) == 1
        assert um[0].tolerance == 2.0
        assert um[0].mod == "strict"

    def test_hard_conc_empty_ignores_get_conc(self):
        """
        An empty ``hardConc`` suppresses ``ModelProfile`` even when
        ``getConc`` is a valid callable.
        """

        def _fake_model(C):
            return C

        est = _FakeEstimator(hardConc=[], getConc=_fake_model)
        constraints = legacy_to_constraints(est)

        mp = [
            c for c in constraints if isinstance(c, ModelProfile) and c.profile == "C"
        ]
        assert len(mp) == 0
