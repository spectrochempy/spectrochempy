# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""Tests for the public MCRALS constraint API skeleton.

These tests cover construction, validation, equality, repr, and the
public import surface. They intentionally do NOT test any numerical
behaviour: the public constraint classes are declarative containers and
are not connected to the internal ALS engine (see
``MCRALS_PR6_PUBLIC_CONSTRAINT_API.md``).
"""

import numpy as np
import pytest

from spectrochempy.analysis import constraints
from spectrochempy.analysis.constraints import (
    Closure,
    Constraint,
    FixedValues,
    ModelProfile,
    Monotonic,
    NonNegative,
    ReferenceProfile,
    Selectivity,
    Unimodal,
    ZeroRegion,
)


# --------------------------------------------------------------------------------------
# Public API surface
# --------------------------------------------------------------------------------------

PUBLIC_NAMES = [
    "Closure",
    "Constraint",
    "FixedValues",
    "ModelProfile",
    "Monotonic",
    "NonNegative",
    "ReferenceProfile",
    "Selectivity",
    "Unimodal",
    "ZeroRegion",
]


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_public_symbol_exposed_via_constraints_namespace(name):
    """Each public class must be accessible as ``constraints.<name>``."""
    assert hasattr(constraints, name), f"constraints.{name} is not exposed"
    cls = getattr(constraints, name)
    assert isinstance(cls, type)
    assert cls.__module__ == "spectrochempy.analysis.decomposition.mcrals_constraints"


def test_public_names_listed_in_constraints_all():
    """The ``constraints.__all__`` includes the public constraint set."""
    for name in PUBLIC_NAMES:
        assert name in constraints.__all__, f"{name} missing from constraints.__all__"


def test_constraints_namespace_is_discoverable():
    """The ``constraints`` submodule is importable from ``spectrochempy.analysis``."""
    from spectrochempy.analysis import constraints as cs

    assert cs is constraints
    assert hasattr(cs, "NonNegative")
    assert hasattr(cs, "ReferenceProfile")


# --------------------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------------------


def test_non_negative_construction_default_components():
    c = NonNegative("C")
    assert c.profile == "C"
    assert c.components is None


def test_non_negative_construction_with_components():
    c = NonNegative("St", components=[0, 2])
    assert c.profile == "St"
    assert c.components == [0, 2]


def test_closure_default_target_is_one():
    c = Closure("C")
    assert c.target == 1.0
    assert c.components is None


def test_closure_custom_target_and_components():
    c = Closure("C", components=[0, 1], target=100.0)
    assert c.components == [0, 1]
    assert c.target == 100.0


def test_unimodal_default_mod_is_strict():
    c = Unimodal("C")
    assert c.mod == "strict"
    assert c.components is None


def test_unimodal_smooth_mod():
    c = Unimodal("St", components=[0], mod="smooth")
    assert c.mod == "smooth"
    assert c.components == [0]


def test_monotonic_direction_required():
    c = Monotonic("C", "increasing")
    assert c.direction == "increasing"
    assert c.tolerance == 1.1  # default


def test_monotonic_decreasing_with_tolerance():
    c = Monotonic("C", "decreasing", components=[1], tolerance=1.0)
    assert c.direction == "decreasing"
    assert c.components == [1]
    assert c.tolerance == 1.0


def test_zero_region_construction():
    c = ZeroRegion("C", region=(0, 5))
    assert c.region == (0, 5)
    assert c.components is None


def test_zero_region_with_components():
    c = ZeroRegion("St", region=(40, 60), components=[1])
    assert c.region == (40, 60)
    assert c.components == [1]


def test_selectivity_construction():
    c = Selectivity("C", region=(0, 5), component=0)
    assert c.region == (0, 5)
    assert c.component == 0


def test_fixed_values_construction_list():
    c = FixedValues("St", values=[[0.1, 0.2], [0.3, 0.4]])
    assert c.values == [[0.1, 0.2], [0.3, 0.4]]
    assert c.components is None


def test_fixed_values_construction_array():
    arr = np.array([[0.1, 0.2], [0.3, 0.4]])
    c = FixedValues("St", values=arr, components=[0])
    assert isinstance(c.values, np.ndarray)
    assert c.components == [0]


def test_reference_profile_construction():
    data = [0.1, 0.2, 0.7, 0.5]
    c = ReferenceProfile("C", component=0, data=data)
    assert c.component == 0
    assert c.data == data


def test_reference_profile_with_spectrum_side():
    data = [0.1, 0.9, 0.5, 0.2]
    c = ReferenceProfile("St", component=0, data=data)
    assert c.component == 0
    assert c.data == data
    assert c.profile == "St"


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


def _identity(C):
    return C


def test_model_profile_construction_concentration():
    c = ModelProfile("C", components=[0, 1], model=_identity)
    assert c.profile == "C"
    assert c.components == [0, 1]
    assert c.model is _identity


def test_model_profile_construction_spectrum():
    c = ModelProfile("St", components=[0], model=_identity)
    assert c.profile == "St"
    assert c.components == [0]
    assert c.model is _identity


def test_model_profile_default_components():
    c = ModelProfile("C", model=_identity)
    assert c.profile == "C"
    assert c.components is None


def test_model_profile_model_args_default_empty():
    c = ModelProfile("C", model=_identity)
    assert c.model_args == ()


def test_model_profile_model_kwargs_default_empty():
    c = ModelProfile("C", model=_identity)
    assert c.model_kwargs == {}


def test_model_profile_with_args_and_kwargs():
    c = ModelProfile(
        "C",
        components=[0, 1],
        model=_identity,
        model_args=("a", 42),
        model_kwargs={"verbose": True},
    )
    assert c.model_args == ("a", 42)
    assert c.model_kwargs == {"verbose": True}
    assert c.model is _identity


def test_model_profile_args_list_converted_to_tuple():
    c = ModelProfile("C", model=_identity, model_args=[1, 2, 3])
    assert c.model_args == (1, 2, 3)


def test_model_profile_kwargs_none_becomes_empty_dict():
    c = ModelProfile("C", model=_identity, model_kwargs=None)
    assert c.model_kwargs == {}


# --------------------------------------------------------------------------------------
# ModelProfile validation
# --------------------------------------------------------------------------------------


def test_model_args_must_be_sequence():
    with pytest.raises(TypeError, match="model_args must be a tuple or list"):
        ModelProfile("C", model=_identity, model_args=42)


def test_model_kwargs_must_be_dict():
    with pytest.raises(TypeError, match="model_kwargs must be a dict or None"):
        ModelProfile("C", model=_identity, model_kwargs="not_a_dict")


# --------------------------------------------------------------------------------------
# Tolerance validation
# --------------------------------------------------------------------------------------


def test_monotonic_tolerance_stored_and_validated():
    c = Monotonic("C", "increasing", tolerance=1.05)
    assert c.tolerance == 1.05


@pytest.mark.parametrize("bad_tol", [0.9, 0.0, -1.0, 0.9999999])
def test_monotonic_tolerance_below_one_rejected(bad_tol):
    with pytest.raises(ValueError, match="tolerance must be >= 1.0"):
        Monotonic("C", "increasing", tolerance=bad_tol)


@pytest.mark.parametrize("bad_tol", ["1.1", None, [1.1]])
def test_monotonic_tolerance_type_rejected(bad_tol):
    with pytest.raises(TypeError, match="tolerance must be a real number"):
        Monotonic("C", "increasing", tolerance=bad_tol)


def test_monotonic_tolerance_one_is_valid():
    # Boundary: 1.0 is the strict case and must be accepted.
    c = Monotonic("C", "increasing", tolerance=1.0)
    assert c.tolerance == 1.0


# --------------------------------------------------------------------------------------
# Validation: profile identifier
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("profile", ["C", "St"])
def test_valid_profile_accepted(profile):
    c = NonNegative(profile)
    assert c.profile == profile


@pytest.mark.parametrize(
    "bad_profile", ["X", "c", "st", "C ", " St", "Concentration", "", "All"]
)
def test_invalid_profile_rejected(bad_profile):
    with pytest.raises((ValueError, TypeError)):
        NonNegative(bad_profile)


def test_non_string_profile_rejected():
    with pytest.raises(TypeError):
        NonNegative(None)
    with pytest.raises(TypeError):
        NonNegative(0)
    with pytest.raises(TypeError):
        NonNegative(["C"])


# --------------------------------------------------------------------------------------
# Validation: components
# --------------------------------------------------------------------------------------


def test_components_single_int_rejected_with_hint():
    # Catch the common mistake of passing a single int instead of a list.
    with pytest.raises(TypeError, match="single int"):
        NonNegative("C", components=0)


def test_components_nested_list_rejected():
    with pytest.raises(TypeError):
        NonNegative("C", components=[[0, 1]])


def test_components_empty_list_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        NonNegative("C", components=[])


def test_components_negative_index_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        NonNegative("C", components=[-1])


def test_components_bool_rejected():
    # bool is a subclass of int but is not a valid component index.
    with pytest.raises(TypeError):
        NonNegative("C", components=[True])


def test_components_tuple_accepted():
    # Tuples should be accepted as well as lists for ergonomic usage.
    c = NonNegative("C", components=(0, 1))
    assert c.components == [0, 1]


# --------------------------------------------------------------------------------------
# Validation: incompatible argument combinations
# --------------------------------------------------------------------------------------


def test_reference_profile_invalid_profile_rejected():
    with pytest.raises(ValueError):
        ReferenceProfile("X", component=0, data=[0.1, 0.2])


def test_closure_target_must_be_positive():
    with pytest.raises(ValueError, match="strictly positive"):
        Closure("C", target=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        Closure("C", target=-1.0)


def test_closure_array_target_list():
    target = [1.0, 2.0, 3.0]
    c = Closure("C", target=target)
    assert c.target is target
    assert c.target == [1.0, 2.0, 3.0]


def test_closure_array_target_tuple():
    target = (1.0, 2.0)
    c = Closure("C", target=target)
    assert c.target is target


def test_closure_array_target_numpy():
    target = np.array([1.0, 1.0, 1.0])
    c = Closure("C", target=target)
    assert c.target is target


def test_closure_array_target_list_with_components():
    c = Closure("C", components=[0, 1], target=[2.0, 2.0])
    assert c.components == [0, 1]
    assert c.target == [2.0, 2.0]


def test_closure_bool_target_rejected():
    with pytest.raises(TypeError, match="bool"):
        Closure("C", target=True)


def test_closure_string_target_rejected():
    with pytest.raises(TypeError, match="string|str"):
        Closure("C", target="default")


def test_closure_none_target_rejected():
    with pytest.raises(TypeError, match="None"):
        Closure("C", target=None)


def test_zero_region_requires_two_entries():
    with pytest.raises(ValueError, match="exactly two"):
        ZeroRegion("C", region=(0,))
    with pytest.raises(ValueError, match="exactly two"):
        ZeroRegion("C", region=(0, 5, 10))


def test_zero_region_stop_after_start():
    with pytest.raises(ValueError, match="greater than start"):
        ZeroRegion("C", region=(5, 5))
    with pytest.raises(ValueError, match="greater than start"):
        ZeroRegion("C", region=(5, 0))


def test_zero_region_negative_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        ZeroRegion("C", region=(-1, 5))


def test_monotonic_invalid_direction_rejected():
    with pytest.raises(ValueError, match="direction must be one of"):
        Monotonic("C", "up")
    with pytest.raises(ValueError, match="direction must be one of"):
        Monotonic("C", "Increasing")  # case-sensitive


def test_unimodal_invalid_mod_rejected():
    with pytest.raises(ValueError, match="mod must be one of"):
        Unimodal("C", mod="loose")


def test_fixed_values_scalar_rejected():
    with pytest.raises(TypeError, match="array-like"):
        FixedValues("St", values=0.5)


def test_fixed_values_none_rejected():
    with pytest.raises(TypeError, match="None"):
        FixedValues("St", values=None)


def test_model_must_be_callable():
    with pytest.raises(TypeError, match="model must be callable"):
        ModelProfile("C", components=[0], model=42)
    with pytest.raises(TypeError, match="model must be callable"):
        ModelProfile("St", components=[0], model="not callable")


# --------------------------------------------------------------------------------------
# ModelProfile mapping validation
# --------------------------------------------------------------------------------------


def test_model_profile_mapping_default_is_none():
    c = ModelProfile("C", model=_identity)
    assert c.mapping is None


def test_model_profile_mapping_with_integers():
    c = ModelProfile("C", components=[0, 1], model=_identity, mapping=[1, 0])
    assert c.mapping == [1, 0]


def test_model_profile_mapping_with_none_entries():
    c = ModelProfile("St", components=[0, 1], model=_identity, mapping=[0, None])
    assert c.mapping == [0, None]


def test_model_profile_mapping_tuple_converted_to_list():
    c = ModelProfile("C", components=[0, 1], model=_identity, mapping=(1, 0))
    assert c.mapping == [1, 0]


def test_model_profile_mapping_rejects_non_list():
    with pytest.raises(TypeError, match="mapping must be a list or None"):
        ModelProfile("C", model=_identity, mapping="invalid")


def test_model_profile_mapping_rejects_non_integer():
    with pytest.raises(TypeError, match="mapping\\[0\\] must be an integer or None"):
        ModelProfile("C", components=[0], model=_identity, mapping=["x"])


def test_model_profile_mapping_rejects_negative():
    with pytest.raises(ValueError, match="mapping\\[0\\] must be non-negative"):
        ModelProfile("C", components=[0], model=_identity, mapping=[-1])


def test_model_profile_mapping_rejects_empty():
    with pytest.raises(ValueError, match="mapping must not be empty"):
        ModelProfile("C", model=_identity, mapping=[])


def test_model_profile_mapping_rejects_length_mismatch():
    with pytest.raises(ValueError, match="mapping has length 1 but got 2 components"):
        ModelProfile("C", components=[0, 1], model=_identity, mapping=[0])


def test_model_profile_mapping_allows_duplicates():
    c = ModelProfile("C", components=[0, 1, 2], model=_identity, mapping=[0, 0, 1])
    assert c.mapping == [0, 0, 1]


def test_model_profile_mapping_all_none_is_valid():
    c = ModelProfile("C", components=[0, 1], model=_identity, mapping=[None, None])
    assert c.mapping == [None, None]


def test_model_profile_mapping_length_ok_when_components_none():
    """When ``components=None``, mapping length is not validated."""
    c = ModelProfile("C", model=_identity, mapping=[0, 1])
    assert c.mapping == [0, 1]


# --------------------------------------------------------------------------------------
# Equality
# --------------------------------------------------------------------------------------


def test_equal_constraints_same_params():
    assert NonNegative("C") == NonNegative("C")
    assert NonNegative("St", components=[0, 2]) == NonNegative("St", components=[0, 2])


def test_unequal_constraints_different_profile():
    assert NonNegative("C") != NonNegative("St")


def test_unequal_constraints_different_components():
    assert NonNegative("C", components=[0]) != NonNegative("C", components=[1])


def test_unequal_constraints_different_target():
    assert Closure("C", target=1.0) != Closure("C", target=2.0)


def test_closure_equal_with_array_target():
    assert Closure("C", target=[1.0, 2.0]) == Closure("C", target=[1.0, 2.0])


def test_closure_equal_with_numpy_array_target():
    assert Closure("C", target=np.array([1.0, 2.0])) == Closure(
        "C", target=np.array([1.0, 2.0])
    )


def test_closure_unequal_array_target():
    assert Closure("C", target=[1.0, 2.0]) != Closure("C", target=[3.0, 4.0])


def test_closure_unequal_scalar_vs_array():
    assert Closure("C", target=1.0) != Closure("C", target=[1.0, 1.0])
    assert Closure("C", target=[1.0, 1.0]) != Closure("C", target=1.0)


def test_closure_method_scaling_default():
    c = Closure("C")
    assert c.method == "scaling"


def test_closure_method_constant_sum():
    c = Closure("C", method="constantSum")
    assert c.method == "constantSum"


def test_closure_method_rejects_non_string():
    with pytest.raises(TypeError, match="method must be a string"):
        Closure("C", method=42)


def test_closure_method_rejects_invalid():
    with pytest.raises(ValueError, match="method must be one of"):
        Closure("C", method="invalid_method")


def test_closure_equal_reference_vs_copy():
    target = np.array([1.0, 2.0, 3.0])
    c1 = Closure("C", target=target)
    c2 = Closure("C", target=target.copy())
    assert c1 == c2


def test_unequal_constraints_different_direction():
    assert Monotonic("C", "increasing") != Monotonic("C", "decreasing")


def test_unequal_constraints_different_tolerance():
    assert Monotonic("C", "increasing", tolerance=1.1) != Monotonic(
        "C", "increasing", tolerance=1.2
    )


def test_unequal_constraints_different_class():
    # Different constraint classes must never compare equal even if they
    # share parameters, to prevent silent confusion.
    nn = NonNegative("C")
    cl = Closure("C", components=None, target=1.0)
    assert nn != cl
    assert cl != nn


def test_constraint_not_equal_to_other_object():
    c = NonNegative("C")
    assert c != "NonNegative(profile='C')"
    assert c != 42
    assert c is not None


def test_constraint_equal_to_itself():
    c = NonNegative("C", components=[0, 1])
    assert c == c


def test_mod_equal_and_unequal():
    assert Unimodal("C") == Unimodal("C")
    assert Unimodal("C", mod="strict") != Unimodal("C", mod="smooth")


def test_unimodal_tolerance_default():
    c = Unimodal("C")
    assert c.tolerance == 1.1


def test_unimodal_tolerance_set():
    c = Unimodal("C", components=[0], tolerance=1.0)
    assert c.tolerance == 1.0


def test_unimodal_tolerance_strict():
    c = Unimodal("C", tolerance=1.0)
    assert c.tolerance == 1.0


def test_unimodal_unequal_tolerance():
    assert Unimodal("C", tolerance=1.0) != Unimodal("C", tolerance=2.0)


def test_unimodal_equal_tolerance():
    assert Unimodal("C", tolerance=2.0) == Unimodal("C", tolerance=2.0)


def test_unimodal_tolerance_repr_omitted_when_default():
    r = repr(Unimodal("C", tolerance=1.1))
    assert "tolerance" not in r


def test_unimodal_tolerance_repr_shown_when_nondefault():
    r = repr(Unimodal("C", tolerance=2.0))
    assert "tolerance=2.0" in r


def test_fixed_values_equal_with_numpy_array():
    arr1 = np.array([1.0, 2.0, 3.0])
    arr2 = np.array([1.0, 2.0, 3.0])
    assert FixedValues("St", values=arr1) == FixedValues("St", values=arr2)


def test_fixed_values_unequal_with_numpy_array():
    arr1 = np.array([1.0, 2.0, 3.0])
    arr2 = np.array([4.0, 5.0, 6.0])
    assert FixedValues("St", values=arr1) != FixedValues("St", values=arr2)


def test_reference_profile_equal_with_numpy_array():
    arr1 = np.array([1.0, 2.0, 3.0, 4.0])
    arr2 = np.array([1.0, 2.0, 3.0, 4.0])
    assert ReferenceProfile("C", component=0, data=arr1) == ReferenceProfile(
        "C", component=0, data=arr2
    )


def test_reference_profile_unequal_with_numpy_array():
    arr1 = np.array([1.0, 2.0, 3.0, 4.0])
    arr2 = np.array([5.0, 6.0, 7.0, 8.0])
    assert ReferenceProfile("C", component=0, data=arr1) != ReferenceProfile(
        "C", component=0, data=arr2
    )


def test_models_equal_same_callback():
    assert ModelProfile("C", components=[0], model=_identity) == ModelProfile(
        "C", components=[0], model=_identity
    )
    assert ModelProfile("St", model=_identity) == ModelProfile("St", model=_identity)


def test_models_equal_with_args_and_kwargs():
    assert ModelProfile(
        "C", model=_identity, model_args=(1,), model_kwargs={"a": 2}
    ) == ModelProfile("C", model=_identity, model_args=(1,), model_kwargs={"a": 2})


def test_models_unequal_different_components():
    assert ModelProfile("C", components=[0], model=_identity) != ModelProfile(
        "C", components=[1], model=_identity
    )


def test_models_unequal_different_callback():
    assert ModelProfile("C", components=[0], model=_identity) != ModelProfile(
        "C", components=[0], model=lambda C: C
    )


def test_models_unequal_different_args():
    assert ModelProfile("C", model=_identity, model_args=(1,)) != ModelProfile(
        "C", model=_identity, model_args=(2,)
    )


def test_models_unequal_different_kwargs():
    assert ModelProfile("C", model=_identity, model_kwargs={"a": 1}) != ModelProfile(
        "C", model=_identity, model_kwargs={"a": 2}
    )


def test_profile_model_differs_by_profile():
    assert ModelProfile("C", components=[0], model=_identity) != ModelProfile(
        "St", components=[0], model=_identity
    )


def test_model_profile_equal_with_same_mapping():
    assert ModelProfile(
        "C", components=[0, 1], model=_identity, mapping=[1, 0]
    ) == ModelProfile("C", components=[0, 1], model=_identity, mapping=[1, 0])


def test_model_profile_unequal_different_mapping():
    assert ModelProfile(
        "C", components=[0, 1], model=_identity, mapping=[1, 0]
    ) != ModelProfile("C", components=[0, 1], model=_identity, mapping=[0, 1])


def test_model_profile_equal_mapping_none_vs_identity():
    assert ModelProfile(
        "C", components=[0, 1], model=_identity, mapping=None
    ) != ModelProfile("C", components=[0, 1], model=_identity, mapping=[0, 1])


# --------------------------------------------------------------------------------------
# repr
# --------------------------------------------------------------------------------------


def test_repr_nonnegative_default():
    assert repr(NonNegative("C")) == "NonNegative(profile='C', components=None)"


def test_repr_nonnegative_with_components():
    assert (
        repr(NonNegative("St", components=[0, 2]))
        == "NonNegative(profile='St', components=[0, 2])"
    )


def test_repr_closure():
    assert (
        repr(Closure("C", components=[0, 1], target=100.0))
        == "Closure(profile='C', components=[0, 1], target=100.0)"
    )


def test_repr_monotonic():
    assert (
        repr(Monotonic("C", "decreasing", components=[0], tolerance=1.0))
        == "Monotonic(profile='C', direction='decreasing', components=[0], tolerance=1.0)"
    )


def test_repr_model_profile_shows_profile():
    r = repr(ModelProfile("C", components=[0, 1], model=_identity))
    assert "ModelProfile(" in r
    assert "profile='C'" in r
    assert "components=[0, 1]" in r
    assert "model=" in r


def test_repr_model_profile_with_args():
    r = repr(ModelProfile("C", model=_identity, model_args=(1, 2)))
    assert "model_args=(1, 2)" in r


def test_repr_model_profile_with_kwargs():
    r = repr(ModelProfile("C", model=_identity, model_kwargs={"a": 1}))
    assert "model_kwargs={'a': 1}" in r


def test_repr_model_profile_omits_empty_args_and_kwargs():
    r = repr(ModelProfile("C", model=_identity))
    assert "model_args" not in r
    assert "model_kwargs" not in r


def test_repr_model_profile_with_spectrum():
    r = repr(ModelProfile("St", components=[0], model=_identity))
    assert "ModelProfile(" in r
    assert "profile='St'" in r
    assert "components=[0]" in r
    assert "model=" in r


def test_repr_model_profile_shows_mapping():
    r = repr(ModelProfile("C", components=[0, 1], model=_identity, mapping=[1, 0]))
    assert "mapping=[1, 0]" in r


def test_repr_model_profile_omits_mapping_when_none():
    r = repr(ModelProfile("C", model=_identity))
    assert "mapping" not in r


def test_repr_is_readable_for_all_classes():
    # Smoke test: every constraint class produces a non-empty,
    # ClassName-prefixed repr that starts with the class name.
    for cls, args, kwargs in [
        (NonNegative, ("C",), {}),
        (Closure, ("C",), {"target": 1.0}),
        (Unimodal, ("C",), {}),
        (Monotonic, ("C",), {"direction": "increasing"}),
        (ZeroRegion, ("C",), {"region": (0, 5)}),
        (Selectivity, ("C",), {"region": (0, 5), "component": 0}),
        (FixedValues, ("St",), {"values": [0.1, 0.2]}),
        (ReferenceProfile, ("C",), {"component": 0, "data": [0.1, 0.2]}),
        (ModelProfile, ("C",), {"components": [0], "model": _identity}),
        (ModelProfile, ("St",), {"components": [0], "model": _identity}),
    ]:
        c = cls(*args, **kwargs)
        r = repr(c)
        assert r.startswith(f"{cls.__name__}(")
        assert r.endswith(")")
        assert len(r) > len(cls.__name__) + 2


# --------------------------------------------------------------------------------------
# Base-class behaviour
# --------------------------------------------------------------------------------------


def test_base_constraint_is_not_sealed_but_not_meant_to_be_used_directly():
    # The base class accepts ``profile`` (so subclasses can reuse it),
    # but it carries no scientific meaning and is not part of the public
    # vocabulary. We just check that it can be instantiated via the
    # normal path and that its name property works.
    base = Constraint("C")
    assert base.profile == "C"
    assert base.name == "Constraint"


def test_base_constraint_repr_uses_repr_params():
    assert repr(Constraint("C")) == "Constraint(profile='C')"


def test_constraint_name_property_returns_class_name():
    assert NonNegative("C").name == "NonNegative"
    assert ModelProfile("C", model=_identity).name == "ModelProfile"


def test_constraint_equality_uses_public_state():
    # Equality is by type + public params, not by identity.
    a = NonNegative("C")
    b = NonNegative("C")
    assert a is not b
    assert a == b


# --------------------------------------------------------------------------------------
# No connection to the internal ALS engine
# --------------------------------------------------------------------------------------


def test_constraints_do_not_affect_mcrals_fit_smoke():
    """Creating a constraint object must not change the behaviour of
    ``MCRALS.fit``. This is a smoke test that the public skeleton is
    decoupled from the internal engine.

    (See the main ``test_mcrals.py`` for the full behavioural
    characterisation.)
    """
    from spectrochempy.analysis.decomposition.mcrals import MCRALS
    from spectrochempy.core.dataset.nddataset import Coord, NDDataset

    # Tiny self-contained synthetic dataset (no fixture dependency).
    rng = np.random.RandomState(42)
    n_t, n_wl, n_comp = 8, 12, 2
    t = Coord(np.arange(n_t), title="time")
    wl = Coord(np.arange(n_wl), title="wavelength")
    C_true = rng.rand(n_t, n_comp)
    St_true = np.abs(rng.rand(n_comp, n_wl))
    X = NDDataset(C_true @ St_true, coordset=(t, wl), units="absorbance")
    C0 = NDDataset(np.abs(C_true + 0.05 * rng.randn(n_t, n_comp)), coordset=(t, None))

    mcr = MCRALS()
    mcr.fit(X, C0)
    baseline_C = np.asarray(mcr.C.data).copy()

    # Constructing public constraints should have no effect on the fit.
    _ = NonNegative("C")
    _ = Closure("St", target=1.0)
    _ = ModelProfile("C", components=[0], model=_identity)

    mcr2 = MCRALS()
    mcr2.fit(X, C0)
    np.testing.assert_array_equal(np.asarray(mcr2.C.data), baseline_C)
