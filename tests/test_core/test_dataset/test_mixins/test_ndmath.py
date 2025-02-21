# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""
Tests for the ndmath module

"""

import numpy as np
import pytest
from pint.errors import DimensionalityError
from quaternion import quaternion

import spectrochempy as scp
from spectrochempy.application import error_, info_
from spectrochempy.core.dataset.arraymixins.ndmath import (
    _binary_ufuncs,
    _comp_ufuncs,
    _unary_ufuncs,
)
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import Quantity, Unit, ur
from spectrochempy.utils.constants import MASKED
from spectrochempy.utils.exceptions import CoordinatesMismatchError
from spectrochempy.utils.testing import (
    RandomSeedContext,
    assert_array_equal,
    assert_dataset_equal,
    assert_units_equal,
)

typequaternion = np.dtype(np.quaternion)


# UNARY MATHS
# -----------
@pytest.mark.parametrize(("name", "comment"), _unary_ufuncs().items())
def test_ndmath_unary_ufuncs_simple_data(nd2d, name, comment):
    nd1 = nd2d.copy() / 1.0e10  # divide to avoid some overflow in exp ufuncs

    # simple unitless NDDataset
    # --------------------------
    assert nd1.unitless

    f = getattr(np, name)
    f(nd1)
    # assert isinstance(r, NDDataset)

    # NDDataset with units
    # ---------------------
    nd1.units = ur.absorbance
    f = getattr(np, name)

    # TODO: some ufunc suppress the units! see pint.
    skip = False

    # if name not in NDDataset.__remove_units__:
    #
    #     try:
    #         f(Quantity(1., nd1.units)).units
    #     except TypeError as e:
    #         error_(f"{name} :", e)
    #         skip = True
    #     except AttributeError:
    #         if name in ['positive', 'fabs', 'cbrt', 'spacing',
    #                     'signbit', 'isnan', 'isinf', 'isfinite', 'logical_not',
    #                     'log2', 'log10', 'log1p', 'exp2', 'expm1']:
    #             pass  # already solved
    #         else:
    #             info_(f"\n =======> {name} remove units! \n")
    #     except DimensionalityError as e:
    #         error_(f"{name} :", e)
    #         skip = True

    if not skip:
        try:
            f(nd1)
            # assert isinstance(r, NDDataset)

            nd1 = nd2d.copy()  # reset nd

            # with units and mask
            nd1.units = ur.absorbance
            nd1[1, 1] = MASKED

            f(nd1)

        except DimensionalityError as e:
            error_(f"{name}: ", e)


def test_bug_lost_dimensionless_units():
    import os

    dataset = NDDataset.read_omnic(os.path.join("irdata", "nh4y-activation.spg"))
    assert dataset.units == "absorbance"
    dataset = dataset - 2.0 - 50.0  # artificially negate some values
    assert dataset.units == "absorbance"

    dataset = dataset.clip(-2.0, 2.0)
    y = np.log2(dataset)
    y._repr_html_()


# BINARY MATH
# ------------
@pytest.mark.parametrize(("name", "comment"), _binary_ufuncs().items())
def test_ndmath_binary_ufuncs_two_datasets(nd2d, name, comment):
    nd1 = nd2d.copy()
    nd2 = nd1.copy() * np.ones_like(nd1) * 0.01

    # simple NDDataset
    # -----------------
    f = getattr(np, name)
    r = f(nd1, nd2)

    assert isinstance(r, NDDataset)

    # NDDataset with units
    # -----------------------
    nd1.units = ur.m
    nd2.units = ur.km
    f = getattr(np, name)
    r = f(nd1, nd2)

    assert isinstance(r, NDDataset)
    if name not in [
        "logaddexp",
        "logaddexp2",
        "true_divide",
        "floor_divide",
        "multiply",
        "divide",
    ]:
        assert r.units == nd1.units


# COMP Methods
@pytest.mark.parametrize(("name", "comment"), _comp_ufuncs().items())
def test_ndmath_comp_ufuncs_two_datasets(nd2d, name, comment):
    nd1 = nd2d.copy()
    nd2 = nd1.copy() + np.ones_like(nd1) * 0.001

    # simple NDDataset
    # -----------------
    f = getattr(np, name)
    r = f(nd1, nd2)

    assert isinstance(r, NDDataset)

    # NDDataset with units
    # -----------------------
    nd1.units = ur.absorbance
    nd2.units = ur.absorbance
    f = getattr(np, name)
    r = f(nd1, nd2)

    assert isinstance(r, NDDataset)


@pytest.mark.parametrize(("name", "comment"), _binary_ufuncs().items())
def test_ndmath_binary_ufuncs_scalar(nd2d, name, comment):
    nd1 = nd2d.copy()
    nd2 = 2.0

    # simple NDDataset
    # -----------------
    f = getattr(np, name)
    r = f(nd1, nd2)

    assert isinstance(r, NDDataset)

    # NDDataset with units
    # -----------------------
    nd1.units = ur.absorbance
    f = getattr(np, name)
    r = f(nd1, nd2)

    assert isinstance(r, NDDataset)
    if name not in [
        "logaddexp",
        "logaddexp2",
        "true_divide",
        "floor_divide",
    ]:
        assert r.units == nd1.units


# --------------------------------------------------------------------------------------
# non ufuncs
# --------------------------------------------------------------------------------------
REDUCE_KEEPDIMS_METHODS = [
    "max",
    "min",
    "amax",
    "amin",
    "round",
    "around",
    "clip",
    "cumsum",
]

REDUCE_KEEPUNITS_METHODS = [
    "sum",
    "mean",
    "std",
    # "ptp",  # removed in numpy 2.0
]

REDUCE_METHODS = [
    "all",
    "any",
    "argmax",
    "argmin",
]


# test if a method is implemented


@pytest.mark.parametrize(
    "name", REDUCE_METHODS + REDUCE_KEEPDIMS_METHODS + REDUCE_KEEPUNITS_METHODS
)
def test_ndmath_classmethod_implementation(nd2d, name):
    nd = nd2d.copy()
    try:
        getattr(NDDataset, name)
    except AttributeError:
        info_("\n{} is not yet implemented".format(name))
    try:
        getattr(np.ma, name)
        getattr(np, name)(nd)
    except AttributeError:
        info_("\n{} is not a np.ma method".format(name))
    except TypeError as e:
        if "required positional" in e.args[0]:
            pass
        else:
            raise TypeError(*e.args)


def test_ndmath_absolute_of_quaternion():
    na0 = np.array(
        [
            [1.0, 2.0, 2.0, 0.0, 0.0, 0.0],
            [1.3, 2.0, 2.0, 0.5, 1.0, 1.0],
            [1, 4.2, 2.0, 3.0, 2.0, 2.0],
            [5.0, 4.2, 2.0, 3.0, 3.0, 3.0],
        ]
    )
    nd = NDDataset(na0, dtype=quaternion)

    coords = CoordSet(np.linspace(-1, 1, 2), np.linspace(-10.0, 10.0, 3))
    assert nd.shape == (2, 3)
    nd.set_coordset(**coords)
    np.abs(nd)

    # TODO: add more testings


def test_unary_ops():
    # UNARY_OPS = ['neg', 'pos', 'abs', 'invert']
    d1 = NDDataset(np.ones((5, 5)))
    d2 = +d1  # pos
    assert isinstance(d2, NDDataset)
    assert np.all(d2.data == 1.0)
    d2 = -d1  # neg
    assert isinstance(d2, NDDataset)
    assert np.all(d2.data == -1.0)
    d3 = abs(d2)  # abs
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 1.0)


def test_unary_ops_with_units():
    # UNARY_OPS = ['neg', 'pos', 'abs']
    d1 = NDDataset(np.ones((5, 5)), units="m")
    d2 = +d1  # pos
    assert isinstance(d2, NDDataset)
    assert np.all(d2.data == 1.0)
    assert d2.units == ur.m
    d2 = -d1  # neg
    assert isinstance(d2, NDDataset)
    assert np.all(d2.data == -1.0)
    assert d2.units == ur.m
    d3 = abs(d2)  # abs
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 1.0)
    assert d2.units == ur("m")


def test_nddataset_add():
    d1 = NDDataset(np.ones((5, 5)), name="d1")
    d2 = NDDataset(np.ones((5, 5)), name="d2")
    d3 = -d1
    assert d3.name != d1

    d3 = d1 * 0.5 + d2
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 1.5)


def test_nddataset_add_with_numpy_array():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = np.ones((5, 5))
    d3 = d1 * 0.5 + d2
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 1.5)
    # should commute!
    d3 = d2 + d1 * 0.5
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 1.5)


def test_nddataset_add_inplace():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = NDDataset(np.ones((5, 5)))
    d1 += d2 * 0.5
    assert np.all(d1.data == 1.5)


def test_nddataset_add_mismatch_coords():
    coord1 = Coord(np.arange(5.0))
    coord2 = Coord(np.arange(1.0, 5.5, 1.0))
    d1 = NDDataset(np.ones((5, 5)), coordset=[coord1, coord2])
    d2 = NDDataset(np.ones((5, 5)), coordset=[coord2, coord1])
    with pytest.raises(CoordinatesMismatchError) as exc:
        d1 -= d2
    assert str(exc.value).startswith("Coordinates [[")
    with pytest.raises(CoordinatesMismatchError) as exc:
        d1 += d2
    assert str(exc.value).startswith(
        "Coordinates [["
    )  # TODO= make more tests like this for various functions


def test_nddataset_add_mismatch_units():
    d1 = NDDataset(np.ones((5, 5)), units="cm^2")
    d2 = NDDataset(np.ones((5, 5)), units="cm")

    with pytest.raises(DimensionalityError) as exc:
        d1 + d2
    assert str(exc.value).startswith(
        "Cannot convert from '[length]' to '[length] ** 2', "
        "Units must be compatible for the `add` operator"
    )

    with pytest.raises(DimensionalityError) as exc:
        d1 += d2
    assert str(exc.value).startswith(
        "Cannot convert from '[length]' to '[length] ** 2', "
        "Units must be compatible for the `iadd` operator"
    )


def test_nddataset_add_units_with_different_scale():
    d1 = NDDataset(np.ones((5, 5)), units="m")
    d2 = NDDataset(np.ones((5, 5)), units="cm")

    x = d1 + 1.0 * ur.cm
    assert x[0, 0].values == 1.01 * ur.m

    x = d1 + d2
    assert x.data[0, 0] == 1.01

    x = d2 + d1
    assert x.data[0, 0] == 101.0
    d1 += d2
    assert d1.data[0, 0] == 1.01
    d2 += d1
    assert d2.data[0, 0] == 102.0


def test_nddataset_add_mismatch_shape():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = NDDataset(np.ones((6, 6)))
    with pytest.raises(ArithmeticError) as exc:
        d1 += d2
    assert exc.value.args[0].startswith("operands could not be broadcast together")


def test_nddataset_add_with_masks():
    # numpy masked arrays mask the result of binary operations if the
    # mask of either operand is set.
    # Does NDData?
    ndd1 = NDDataset(np.array([1, 2]))
    ndd2 = NDDataset(np.array([2, 1]))
    result = ndd1 + ndd2
    assert_array_equal(result.data, np.array([3, 3]))

    ndd1 = NDDataset(np.array([1, 2]), mask=np.array([True, False]))
    other_mask = ~ndd1.mask
    ndd2 = NDDataset(np.array([2, 1]), mask=other_mask)
    result = ndd1 + ndd2
    # The result should have all entries masked...
    assert result.mask.all()


def test_nddataset_subtract():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = NDDataset(np.ones((5, 5)) * 2.0)
    d3 = d1 - d2
    assert np.all(d3.data == -1.0)


def test_nddataset_substract_with_numpy_array():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = np.ones((5, 5))
    d3 = d1 * 0.5 - d2
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == -0.5)
    # should commute!
    d3 = d2 - d1 * 0.5
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 0.5)


def test_nddataset_binary_operation_with_other_1D():
    coord1 = Coord(np.linspace(0.0, 10.0, 10))
    coord2 = Coord(np.linspace(1.0, 5.5, 5))
    d1 = NDDataset(np.random.random((10, 5)), coordset=[coord1, coord2])
    d2 = d1[0]
    # this should work independently of the value of the coordinates on dimension y
    d3 = d1 - d2
    assert_array_equal(d3.data, d1.data - d2.data)


def test_nddataset_subtract_mismatch_units():
    d1 = NDDataset(np.ones((5, 5)), units="m")
    d2 = NDDataset(np.ones((5, 5)) * 2.0, units="m/s")
    with pytest.raises(DimensionalityError) as exc:
        d1 -= d2
    assert (
        str(exc.value) == "Cannot convert from '[length] / [time]' to '[length]', "
        "Units must be compatible for the `isub` operator"
    )


def test_nddataset_subtract_mismatch_shape():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = NDDataset(np.ones((6, 6)) * 2.0)
    with pytest.raises(ArithmeticError) as exc:
        d1 -= d2
    assert exc.value.args[0].startswith("operands could not be broadcast together")


def test_nddataset_multiply_with_numpy_array():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = np.ones((5, 5)) * 2.0
    d3 = d1 * d2
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 2.0)
    # should commute!
    d3 = d2 * d1
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 2.0)


def test_nddataset_divide_with_numpy_array():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = np.ones((5, 5)) * 2.0
    d3 = d1 / d2
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 0.5)
    # should commute!
    d3 = d2 / d1
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 2.0)


# first operand has units km, second has units m
@pytest.mark.parametrize(
    ("operation", "result_units"),
    [
        ("__add__", ur.km),
        ("__sub__", ur.km),
        ("__mul__", ur.km * ur.m),
        ("__truediv__", ur.km / ur.m),
    ],
)
def test_ndmath_unit_conversion_operators(operation, result_units):
    in_km = NDDataset(np.array([1, 1]), units=ur.km)
    in_m = NDDataset(in_km.data * 1000, units=ur.m)
    operator_km = in_km.__getattribute__(operation)
    combined = operator_km(in_m)
    assert_units_equal(combined.units, result_units)


@pytest.mark.parametrize(
    ("unit1", "unit2", "op", "result_units"),
    [
        (None, None, "__add__", None),
        (None, None, "__mul__", None),
        (None, ur.m, "__mul__", ur.m),
        (ur.dimensionless, None, "__mul__", ur.dimensionless),
        (ur.eV, ur.eV, "__add__", ur.eV),
        (ur.eV, ur.eV, "__sub__", ur.eV),
        (ur.eV, ur.eV, "__truediv__", ur.dimensionless),
        (ur.eV, ur.m, "__mul__", ur.m * ur.eV),
    ],
)
def test_arithmetic_unit_calculation(unit1, unit2, op, result_units):
    ndd1 = NDDataset(np.array([1]), units=unit1)
    ndd2 = NDDataset(np.array([1]), units=unit2)
    ndd1_method = ndd1.__getattribute__(op)
    result = ndd1_method(ndd2)
    try:
        assert result.units == result_units
    except AssertionError:
        assert_units_equal(ndd1_method(ndd2).units, result_units)


def test_simple_arithmetic_on_full_dataset():
    # due to a bug in notebook with the following
    import os

    dataset = NDDataset.read_omnic(os.path.join("irdata", "nh4y-activation.spg"))
    (
        dataset - dataset[0]
    )  # suppress the first spectrum to all other spectra in the series


def test_ndmath_and_api_methods(IR_dataset_1D, IR_dataset_2D):
    # CREATION _LIKE METHODS
    # ----------------------
    # from a list
    x = [1, 2, 3]

    # _like as an API method
    ds = NDDataset(x).full_like(2.5, title="empty")
    ds = scp.full_like(x, 2)
    assert np.all(ds.data == np.full((3,), 2))
    assert ds._implements("NDDataset")

    # _like as a classmethod
    ds = NDDataset.full_like(x, 2)
    assert np.all(ds.data == np.full((3,), 2))
    assert ds._implements("NDDataset")

    # _like as an instance method
    ds = NDDataset(x).full_like(2)
    assert np.all(ds.data == np.full((3,), 2))
    assert ds._implements("NDDataset")

    # _like as an instance method
    ds = NDDataset(x).empty_like(title="empty")
    assert ds._implements("NDDataset")
    assert ds.title == "empty"

    # from an array
    x = np.array([1, 2, 3])

    ds = NDDataset(x).full_like(2)
    assert np.all(ds.data == np.full((3,), 2))
    assert ds._implements("NDDataset")

    # from a NDArray subclass with units
    x = NDDataset([1, 2, 3], units="km")
    ds = scp.full_like(x, 2)
    assert np.all(ds.data == np.full((3,), 2))
    assert ds._implements("NDDataset")
    assert ds.units == ur.km

    ds1 = scp.full_like(ds, np.nan, dtype=np.double, units="m")
    assert ds1.units == Unit("m")

    # change of units is forced
    ds2 = scp.full_like(ds, 2, dtype=np.double, units="s")
    assert ds2.units == ur.s

    # other like creation functions
    nd = scp.empty_like(ds, dtype=np.double, units="m")
    assert str(nd) == "NDDataset: [float64] m (size: 3)"
    assert nd.dtype == np.dtype(np.double)

    nd = scp.zeros_like(ds, dtype=np.double, units="m")
    assert str(nd) == "NDDataset: [float64] m (size: 3)"
    assert np.all(nd.data == np.zeros((3,)))

    nd = scp.ones_like(ds, dtype=np.double, units="m")
    assert str(nd) == "NDDataset: [float64] m (size: 3)"
    assert np.all(nd.data == np.ones((3,)))

    # FULL
    # ----
    ds = NDDataset.full((6,), 0.1)
    assert ds.size == 6
    assert str(ds) == "NDDataset: [float64] unitless (size: 6)"

    # ZEROS
    # -----
    ds = NDDataset.zeros((6,), units="km")
    assert ds.size == 6
    assert str(ds) == "NDDataset: [float64] km (size: 6)"

    # ONES
    # ----
    ds = NDDataset.ones((6,))
    ds = scp.full((6,), 0.1)
    assert ds.size == 6
    assert str(ds) == "NDDataset: [float64] unitless (size: 6)"

    ds = NDDataset.ones((6,), units="absorbance", dtype="complex128")
    assert ds.size == 3
    assert str(ds) == "NDDataset: [complex128] a.u. (size: 3)"
    assert ds[0].data == 1.0 + 1.0j

    # LINSPACE
    # --------
    c2 = Coord.linspace(1, 20, 200, units="m", name="mycoord")
    assert c2.name == "mycoord"
    assert c2.size == 200
    assert c2[-1].data == 20
    assert c2[0].values == Quantity(1, "m")

    # ARANGE
    # -------
    c3 = Coord.arange(1, 20.0001, 1, units="s", name="mycoord")
    assert c3.name == "mycoord"
    assert c3.size == 20
    assert c3[-1].data == 20
    assert c3[0].values == Quantity(1, "s")

    # EYE
    # ----
    ds1 = scp.NDDataset.eye(2, dtype=int)
    assert str(ds1) == "NDDataset: [float64] unitless (shape: (y:2, x:2))"
    ds = scp.eye(3, k=1, units="km")
    assert (ds.data == np.eye(3, k=1)).all()
    assert ds.units == ur.km

    # IDENTITY
    # --------
    ds = scp.identity(3, units="km")
    assert (
        ds.data
        == np.identity(
            3,
        )
    ).all()
    assert ds.units == ur.km

    # RANDOM
    # ------
    ds = scp.random((3, 3), units="km")
    assert str(ds) == "NDDataset: [float64] km (shape: (y:3, x:3))"

    # adding coordset
    c1 = Coord.linspace(1, 20, 200, units="m", name="axe_x")
    ds = scp.random((200,), units="km", coordset=scp.CoordSet(x=c1))

    # DIAGONAL
    # --------
    # extract diagonal
    nd = scp.full((2, 2), 0.5, units="s", title="initial")
    assert str(nd) == "NDDataset: [float64] s (shape: (y:2, x:2))"
    ndd = scp.diagonal(nd, title="diag")
    assert str(ndd) == "NDDataset: [float64] s (size: 2)"
    assert ndd.units == Unit("s")

    cx = scp.Coord([0, 1])
    cy = scp.Coord([2, 5])
    nd = NDDataset.full(
        (2, 2), 0.5, units="s", coordset=scp.CoordSet(cx, cy), title="initial"
    )
    assert str(nd) == "NDDataset: [float64] s (shape: (y:2, x:2))"
    ndd = nd.diagonal(title="diag2")
    assert str(ndd) == "NDDataset: [float64] s (size: 2)"
    assert ndd.units == Unit("s")
    assert ndd.title == "diag2"

    cx = scp.Coord([0, 1, 2])
    cy = scp.Coord([2, 5])
    nd = NDDataset.full(
        (2, 3), 0.5, units="s", coordset=scp.CoordSet(x=cx, y=cy), title="initial"
    )
    assert str(nd) == "NDDataset: [float64] s (shape: (y:2, x:3))"
    ndd = nd.diagonal(title="diag3")
    assert str(ndd) == "NDDataset: [float64] s (size: 2)"
    assert ndd.units == Unit("s")
    assert ndd.title == "diag3"
    assert_array_equal(nd.x.data[: ndd.x.size], ndd.x.data)

    ndd = nd.diagonal(title="diag4", dim="y")
    assert str(ndd) == "NDDataset: [float64] s (size: 2)"
    assert ndd.units == Unit("s")
    assert ndd.title == "diag4"
    assert_array_equal(nd.y.data[: ndd.y.size], ndd.y.data)

    # DIAG
    # ----
    ref = NDDataset(np.diag((3, 3.4, 2.3)), units="m", title="something")

    # Three forms should return the same NDDataset
    ds = scp.diag((3, 3.4, 2.3), units="m", title="something")
    assert_dataset_equal(ds, ref)

    ds = NDDataset.diag((3, 3.4, 2.3), units="m", title="something")
    assert_dataset_equal(ds, ref)

    ds = NDDataset((3, 3.4, 2.3)).diag(units="m", title="something")
    assert_dataset_equal(ds, ref)

    # and this too
    ds1 = NDDataset((3, 3.4, 2.3), units="s", title="another")

    ds = scp.diag(ds1, units="m", title="something")
    assert_dataset_equal(ds, ref)

    ds = ds1.diag(units="m", title="something")
    assert_dataset_equal(ds, ref)

    # BOOL : ALL and ANY
    # ------------------
    ds = NDDataset([[True, False], [True, True]])
    b = np.all(ds)
    assert not b

    b = ds.all()
    assert not b

    b = NDDataset.any(ds)
    assert b

    b = ds.all(dim="y")
    assert_array_equal(b, np.array([True, False]))

    b = ds.any(dim="y")
    assert_array_equal(b, np.array([True, True]))

    # ARGMAX, MAX
    # -----------
    nd1 = IR_dataset_1D.copy()
    nd1[1290.0:890.0] = MASKED
    assert nd1.is_masked
    assert str(nd1) == "NDDataset: [float64] a.u. (size: 5549)"

    idx = nd1.argmax()
    assert idx == 3122

    mx = nd1.max()
    # alternative
    mx = NDDataset.max(nd1)
    assert mx == Quantity(3.8080601692199707, "absorbance")

    mxk = nd1.max(keepdims=True)
    assert isinstance(mxk, NDDataset)
    assert str(mxk) == "NDDataset: [float64] a.u. (size: 1)"
    assert mxk.values == mx

    # test on a 2D NDDataset
    nd2 = IR_dataset_2D.copy()
    nd2[:, 1290.0:890.0] = MASKED

    mx = nd2.max()  # no axis specified
    assert mx == Quantity(3.8080601692199707, "absorbance")
    mxk = nd2.max(keepdims=True)
    assert str(mxk) == "NDDataset: [float64] a.u. (shape: (y:1, x:1))"

    nd2m = nd2.max("y")  # axis selected
    ax = nd2m.plot()
    nd2[0].plot(ax=ax, clear=False)

    nd2m2 = nd2.max("x")  # axis selected
    nd2m2.plot()

    nd2m = nd2.max("y", keepdims=True)
    assert nd2m.shape == (1, 5549)

    nd2m = nd2.max("x", keepdims=True)
    assert nd2m.shape == (55, 1)

    mx = nd2.min()  # no axis specified
    assert mx == Quantity(-0.022955093532800674, "absorbance")
    mxk = nd2.min(keepdims=True)
    assert str(mxk) == "NDDataset: [float64] a.u. (shape: (y:1, x:1))"

    nd2m = nd2.min("y")  # axis selected
    ax = nd2m.plot()
    nd2[0].plot(ax=ax, clear=False)

    nd2m2 = nd2.min("x")  # axis selected
    nd2m2.plot()

    nd2m = nd2.min("y", keepdims=True)
    assert nd2m.shape == (1, 5549)

    nd2m = nd2.min("x", keepdims=True)
    assert nd2m.shape == (55, 1)

    # CLIP
    # ----
    nd3 = nd2 - 2.0
    assert nd3.units == nd2.units
    nd3c = nd3.clip(-0.5, 1.0)
    assert nd3c.max().m == 1.0
    assert nd3c.min().m == -0.5

    # COORDMIN AND COORDMAX
    # ---------------------
    cm = nd2.coordmin()
    assert np.around(cm["x"], 3) == Quantity(1290.165, "cm^-1")

    cm = nd2.coordmin(dim="y")
    assert cm.size == 1

    cm = nd2.coordmax(dim="y")
    assert cm.size == 1

    cm = nd2.coordmax(dim="x")
    assert cm.size == 1

    # ABS
    # ----
    nd2a = NDDataset.abs(nd2)
    mxa = nd2a.min()
    assert mxa > 0

    nd2a = np.abs(nd2)
    mxa = nd2a.min()
    assert mxa > 0

    ndd = NDDataset([1.0, 2.0 + 1j, 3.0])
    val = np.abs(ndd)

    val = ndd[1] * 1.2 - 10.0
    val = np.abs(val)

    # FROMFUNCTION
    # ------------
    # 1D
    def func1(t, v):
        d = v * t
        return d

    time = Coord.linspace(
        0,
        9,
        10,
    )
    distance = NDDataset.fromfunction(func1, v=134, coordset=CoordSet(t=time))
    assert distance.dims == ["t"]
    assert_array_equal(distance.data, np.fromfunction(func1, (10,), v=134))

    time = Coord.linspace(0, 90, 10, units="min")
    distance = NDDataset.fromfunction(
        func1, v=Quantity(134, "km/hour"), coordset=CoordSet(t=time)
    )
    assert distance.dims == ["t"]
    assert_array_equal(distance.data, np.fromfunction(func1, (10,), v=134) * 10 / 60)

    # 2D
    def func2(x, y):
        d = x + 1 / y
        return d

    c0 = Coord.linspace(0, 9, 3)
    c1 = Coord.linspace(10, 20, 2)

    # implicit ordering of coords (y,x)
    distance = NDDataset.fromfunction(func2, coordset=CoordSet(c1, c0))
    assert distance.shape == (2, 3)
    assert distance.dims == ["y", "x"]

    # or equivalent
    distance = NDDataset.fromfunction(func2, coordset=[c1, c0])
    assert distance.shape == (2, 3)
    assert distance.dims == ["y", "x"]

    # explicit ordering of coords (y,x)  #
    distance = NDDataset.fromfunction(func2, coordset=CoordSet(u=c0, v=c1))

    assert distance.shape == (2, 3)
    assert distance.dims == ["v", "u"]
    assert distance[0, 2].data == distance.u[2].data + 1.0 / distance.v[0].data

    # with units
    def func3(x, y):
        d = x + y
        return d

    c0u = Coord.linspace(0, 9, 3, units="km")
    c1u = Coord.linspace(10, 20, 2, units="m")
    distance = NDDataset.fromfunction(func3, coordset=CoordSet(u=c0u, v=c1u))

    assert distance.shape == (2, 3)
    assert distance.dims == ["v", "u"]
    assert distance[0, 2].values == distance.u[2].values + distance.v[0].values

    c0u = Coord.linspace(0, 9, 3, units="km")
    c1u = Coord.linspace(10, 20, 2, units="m^-1")
    distance = NDDataset.fromfunction(func2, coordset=CoordSet(u=c0u, v=c1u))

    assert distance.shape == (2, 3)
    assert distance.dims == ["v", "u"]
    assert distance[0, 2].values == distance.u[2].values + 1.0 / distance.v[0].values

    # FROMITER
    # --------
    iterable = (x * x for x in range(5))
    nit = scp.fromiter(iterable, float, units="km")
    assert str(nit) == "NDDataset: [float64] km (size: 5)"
    assert_array_equal(nit.data, np.array([0, 1, 4, 9, 16]))

    # MEAN, AVERAGE
    # -----
    nd = IR_dataset_2D.copy()

    m = np.mean(nd)

    assert m.shape == ()
    assert m == Quantity(np.mean(nd.data), "absorbance")

    m = nd.average()
    assert m.shape == ()
    assert m == Quantity(np.average(nd.data), "absorbance")

    mx = nd.mean(keepdims=True)
    assert mx.shape == (1, 1)

    mxd = nd.mean(dim="y")  # cannot make n
    assert str(mxd) == "NDDataset: [float64] a.u. (size: 5549)"
    assert str(mxd.x) in [
        "Coord: [float64] cm⁻¹ (size: 5549)",
        "Coord: [float64] 1/cm (size: 5549)",
    ]

    # ----
    nd2 = NDDataset([[0, 1, 2], [3, 4, 5]])  # no coord (check issues

    m = np.mean(nd2)

    assert m.shape == ()
    assert m == np.mean(nd2.data)
    assert m == 2.5

    m = np.mean(nd2, keepdims=True)
    assert m.shape == (1, 1)
    assert m.data == [[2.5]]

    m = nd2.mean(dim="y")
    assert m.shape == (3,)
    assert_array_equal(m.data, [1.5, 2.5, 3.5])
    assert str(m) == "NDDataset: [float64] unitless (size: 3)"

    m = nd2.mean(dim=0, keepdims=True)
    assert m.shape == (1, 3)
    assert_array_equal(m.data, [[1.5, 2.5, 3.5]])
    assert str(m) == "NDDataset: [float64] unitless (shape: (y:1, x:3))"

    m = nd2.mean(dim="y")
    assert m.shape == (3,)
    assert_array_equal(m.data, [1.5, 2.5, 3.5])
    assert str(m) == "NDDataset: [float64] unitless (size: 3)"


def test_nddataset_fancy_indexing():
    # numpy vs dataset
    rand = np.random.RandomState(42)
    x = rand.randint(100, size=10)

    # single value indexing

    dx = NDDataset(x)
    assert (dx[3].data, dx[7].data, dx[2].data) == (x[3], x[7], x[2])

    # slice indexing

    assert_array_equal(dx[3:7].data, x[3:7])

    # boolean indexingassert

    assert_array_equal(dx[x > 52].data, x[x > 52])

    # fancy indexing
    ind = [3, 7, 4]

    assert_array_equal(dx[ind].data, x[ind])

    ind = np.array([[3, 7], [4, 5]])

    assert_array_equal(dx[ind].data, x[ind])

    with RandomSeedContext(1234):
        a = np.random.random((3, 5)).round(1)
    c = (np.arange(3), np.arange(5))
    nd = NDDataset(a, coordset=c)

    a = nd[[1, 0, 2]]

    a = nd[np.array([1, 0])]


def test_coord_add_units_with_different_scale():
    d1 = Coord.arange(3.0, units="m")
    d2 = Coord.arange(3.0, units="cm")

    x = d1 + 1.0 * ur.cm
    assert x.data[1] == 1.01

    x = d1 + d2
    assert x.data[1] == 1.01
    x = d2 + d1
    assert x.data[1] == 101.0
    d1 += d2
    assert d1.data[1] == 1.01
    d2 += d1
    assert d2.data[1] == 102.0


def test_creation():
    nd = scp.ones(5, units="km")
    assert str(nd) == "NDDataset: [float64] km (size: 5)"
    nd = scp.ones((5,), dtype=np.dtype("int64"), mask=[True, False, False, False, True])
    assert nd.dtype == np.dtype("int64")


def test_from_function_docstring():
    def func1(t, v):
        d = v * t
        return d

    time = scp.Coord.arange(0, 60, 10, units="min")
    scp.fromfunction(
        func1, v=scp.Quantity(134, "km/hour"), coordset=scp.CoordSet(t=time)
    )


def test_round_docstring_example():
    ds = scp.read("wodger.spg")
    ds_transformed1 = np.round(ds, 3)
    ds_transformed2 = np.around(ds, 3)
    ds_transformed5 = ds.round(3)
    ds_transformed6 = NDDataset.round(ds, 3)

    assert_dataset_equal(ds_transformed1, ds_transformed2)
    assert_dataset_equal(ds_transformed1, ds_transformed5)
    assert_dataset_equal(ds_transformed1, ds_transformed6)

    ds[:, 3000.0:3500.0] = scp.MASKED
    dsm_transformed1 = np.ma.round(ds)
    dsm_transformed2 = np.around(ds)
    dsm_transformed4 = ds.round()

    assert_dataset_equal(dsm_transformed1, dsm_transformed2)
    assert_dataset_equal(dsm_transformed1, dsm_transformed4)


def test_issue417():
    X = scp.read_omnic("irdata/nh4y-activation.spg")
    x = X - X[-1]

    print("--subtract with ds from read_omnic")
    print(f"mean x: {np.mean(x.data)}")
    print(f"var x: {np.var(x.data)}")
    print("")

    f = X.write("X.scp")
    X_r = scp.read("X.scp")
    f.unlink()

    assert_array_equal(X.data, X_r.data)
    assert_dataset_equal(X, X_r)
    assert_units_equal(X.units, X_r.units)
    assert_dataset_equal(X[-1], X_r[-1])

    x_r = X_r - X_r[-1]

    print("--subtract after write/read_scp")
    print(f"mean x_r: {np.mean(x_r.data)}")
    print(f"var x_r: {np.var(x_r.data)}")
    print("")

    x_r2 = X_r - X_r[-1].data
    print("--subtract with data field")
    print(f"mean x_r2: {np.mean(x_r2.data)}")
    print(f"var x_r2: {np.var(x_r2.data)}")

    assert_array_equal(x.data, x_r2.data)
    assert_array_equal(x.data, x_r.data)
    assert_dataset_equal(x, x_r)
