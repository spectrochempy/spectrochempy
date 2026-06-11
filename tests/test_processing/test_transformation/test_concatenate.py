# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.units import ur
from spectrochempy.processing.transformation.concatenate import concatenate, stack
from spectrochempy.utils.exceptions import (
    DimensionsCompatibilityError,
    UnitsCompatibilityError,
)
from spectrochempy.utils.testing import assert_dataset_almost_equal


@pytest.mark.data
def test_concatenate(IR_dataset_2D):
    dataset = IR_dataset_2D
    dim = "x"
    # print(dataset)
    s = dataset
    s1 = dataset[:, -10:]
    s2 = dataset[:, :-10]

    # specify axis
    s = concatenate(s1, s2, dims=dim)
    assert s.units == s1.units
    assert s.shape[-1] == (s1.shape[-1] + s2.shape[-1])
    assert s.x.size == (s1.x.size + s2.x.size)
    assert s.x != dataset.x
    s = s.sort(dims=dim, descend=True)  #
    assert_dataset_almost_equal(s.x, Coord(dataset.x, linear=False), decimal=3)

    # default concatenation in the last dimension
    s = concatenate(s1, s2)
    assert s.units == s1.units
    assert s.shape[-1] == (s1.shape[-1] + s2.shape[-1])
    assert s.x.size == (s1.x.size + s2.x.size)
    assert s.x != dataset.x
    s = s.sort(descend=True)  #
    assert_dataset_almost_equal(s.x, Coord(dataset.x, linear=False), decimal=3)

    s1 = dataset[:10]
    s2 = dataset[20:]
    # check with derived units
    s1.to(ur.m, force=True)
    s2.to(ur.dm, force=True)
    s = concatenate(s1, s2, dims=0)
    assert s.units == s1.units
    assert s.shape[0] == (s1.shape[0] + s2.shape[0])
    assert s.y.size == (s1.y.size + s2.y.size)
    s = s.sort(dim="y")

    # second syntax
    s = s1.concatenate(s2, dims=0)
    assert s.units == s1.units
    assert s.shape[0] == (s1.shape[0] + s2.shape[0])
    assert s.y.size == (s1.y.size + s2.y.size)

    # third syntax
    s = concatenate((s1, s2), dims=0)
    assert s.units == s1.units
    assert s.shape[0] == (s1.shape[0] + s2.shape[0])
    assert s.y.size == (s1.y.size + s2.y.size)

    # coordset
    coord_2 = Coord(np.cos(s.y.data), title="cos_time")
    s.set_coordset(y=[s.y, coord_2], x=s.x)
    s1 = s[:2]
    s2 = s[-5:]
    s12 = concatenate(s1, s2, axis=0)
    assert (s2["y"].labels[1] == s12["y"].labels[1][-5:]).all()

    # authors
    s0 = s[0]
    s1 = s[1]
    s0.author = "sdqe65g4rf"
    s2 = concatenate(s0, s1)
    assert "sdqe65g4rf" in s2.author and s1.author in s2.author

    # titles
    s0.title = "new_title"
    assert concatenate(s0, s1).title == "new_title"

    # incompatible dimensions
    s0 = scp.NDDataset(np.zeros((10, 100)))
    s1 = scp.NDDataset(np.zeros((10, 100)))
    with pytest.raises(DimensionsCompatibilityError):
        s0.concatenate(s1[0].squeeze())

    with pytest.raises(DimensionsCompatibilityError):
        s0.concatenate(s1[:, :50], axis=0)

    # incompatible units
    s0 = scp.NDDataset(np.zeros((10, 100)), units="V")
    s1 = scp.NDDataset(np.zeros((10, 100)), units="A")
    with pytest.raises(UnitsCompatibilityError):
        scp.concatenate(s0, s1)

    s1 = scp.NDDataset(np.ones((10, 100)), units="mV")
    s01 = scp.concatenate(s0, s1)
    assert s01.data[-1, -1] == 0.001

    # ----------------------------------------------------------------------------------
    # Stack

    # concatenation using stack
    s1 = dataset[:10]
    s2 = dataset[-10:]
    s = stack(s1, s2)
    assert s.units == s1.units
    assert s.shape == (2, s1.shape[0], s1.shape[1])
    assert s.y.size == s1.y.size
    assert s.x.size == s1.x.size

    with pytest.warns(DeprecationWarning):
        concatenate(s1, s2, force_stack=True)

    # If one of the dimensions is of size one, then this dimension is NOT removed before stacking
    s0 = dataset[0]
    s1 = dataset[1]
    ss = stack(s0, s1)
    assert s0.shape == (1, 5549)
    assert ss.shape == (2, s1.shape[0], s1.shape[1])

    # # stack squeezed nD dataset
    s0 = dataset[0].copy().squeeze()
    assert s0.shape == (5549,)
    s1 = dataset[1].squeeze()
    assert s1.shape == (5549,)
    s = stack(s0, s1)
    assert s.shape == (2, 5549)

    # # stack squeezed nD dataset
    s2 = s1[0:100]
    with pytest.raises(DimensionsCompatibilityError):
        s = stack(s0, s2)


def test_bug_243():
    import spectrochempy as scp

    D = scp.zeros((10, 100))

    x = scp.Coord.arange(100)
    y = scp.Coord.arange(10)

    D.set_coordset(x=x, y=y)
    D1 = D[:, 0.0:10.0]
    D2 = D[:, 20.0:40.0]

    D12 = scp.concatenate(D1, D2, axis=1)

    # D2.x.data[-1] is 40., as expected, but not D12.x.data[-1]:
    assert D12.x.data[-1] == D2.x.data[-1]


# ==============================================================================
# CoordSet lifecycle — concatenate dimension coordinate propagation
# ==============================================================================


def test_concatenate_preserves_coord_values():
    """Concatenate along dim merges coordinate data correctly."""
    x1 = scp.Coord(np.arange(3.0))
    x2 = scp.Coord(np.arange(3.0, 6.0))
    ds1 = scp.NDDataset(np.ones((2, 3)), coordset=[scp.Coord(np.arange(2.0)), x1])
    ds2 = scp.NDDataset(np.ones((2, 3)), coordset=[scp.Coord(np.arange(2.0)), x2])

    result = concatenate(ds1, ds2, dims="x")
    assert result.x.data.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_concatenate_preserves_labels():
    """Concatenate along dim merges labeled coordinate labels."""
    x1 = scp.Coord(np.arange(3.0), labels=["a", "b", "c"])
    x2 = scp.Coord(np.arange(3.0, 6.0), labels=["d", "e", "f"])
    ds1 = scp.NDDataset(np.ones((2, 3)), coordset=[scp.Coord(np.arange(2.0)), x1])
    ds2 = scp.NDDataset(np.ones((2, 3)), coordset=[scp.Coord(np.arange(2.0)), x2])

    result = concatenate(ds1, ds2, dims="x")
    assert result.x.labels.tolist() == ["a", "b", "c", "d", "e", "f"]


def test_concatenate_preserves_multi_coord_labels():
    """Concatenate preserves labels for same-dimension multi-coordinate axes."""
    y1 = scp.Coord(np.arange(2.0), title="rows")
    inner1 = scp.CoordSet(
        scp.Coord(np.arange(3.0), labels=["a", "b", "c"]),
        scp.Coord(np.arange(3.0, 6.0), labels=["d", "e", "f"]),
    )
    inner2 = scp.CoordSet(
        scp.Coord(np.arange(3.0, 6.0), labels=["g", "h", "i"]),
        scp.Coord(np.arange(6.0, 9.0), labels=["j", "k", "l"]),
    )
    ds1 = scp.NDDataset(np.ones((2, 3)), coordset=[y1, inner1])
    ds2 = scp.NDDataset(np.ones((2, 3)), coordset=[y1, inner2])

    result = concatenate(ds1, ds2, dims="x")
    # The x dim has two sub-coords; each should have concatenated labels.
    x_coords = result.coordset["x"]
    assert isinstance(x_coords, scp.CoordSet)
    assert len(x_coords) == 2

    # Collect labels from sub-coords without assuming _coords order.
    all_label_sets = {tuple(c.labels.tolist()) for c in x_coords.coords}
    assert ("a", "b", "c", "g", "h", "i") in all_label_sets
    assert ("d", "e", "f", "j", "k", "l") in all_label_sets


def test_concatenate_multi_coord_default_data_is_not_concatenated():
    """Multi-coord data is not concatenated through ``CoordSet.data``.

    ``CoordSet.data`` delegates to ``self.default.data``, and the old code
    only sets a dead ``_data`` dynamic attribute on the CoordSet that is
    never read.  Behavior is preserved.
    """
    y = scp.Coord(np.arange(2.0), title="rows")
    inner1 = scp.CoordSet(
        scp.Coord(np.arange(3.0)),
        scp.Coord(np.arange(3.0, 6.0)),
    )
    inner2 = scp.CoordSet(
        scp.Coord(np.arange(3.0, 6.0)),
        scp.Coord(np.arange(6.0, 9.0)),
    )
    ds1 = scp.NDDataset(np.ones((2, 3)), coordset=[y, inner1])
    ds2 = scp.NDDataset(np.ones((2, 3)), coordset=[y, inner2])

    result = concatenate(ds1, ds2, dims="x")
    # Each sub-coord data length is unchanged (concatenation not applied).
    for c in result.coordset["x"].coords:
        assert len(c.data) == 3


def test_concatenate_with_empty_coord_returns_coordset_unchanged():
    """Concatenate on an empty dimension coordinate returns the coordset unchanged."""
    x_empty = scp.Coord(None, size=3)
    y1 = scp.Coord(np.arange(2.0))
    y2 = scp.Coord(np.arange(2.0, 4.0))
    ds1 = scp.NDDataset(np.ones((2, 3)), coordset=[y1, x_empty])
    ds2 = scp.NDDataset(np.ones((2, 3)), coordset=[y2, x_empty])

    result = concatenate(ds1, ds2, dims="x")
    # The x coord is empty in both; the result should still have an empty x coord.
    assert result.x.is_empty


def test_concatenate_preserves_non_first_default():
    """Concatenate preserves the selected non-first default for multi-coord dims."""
    y = scp.Coord(np.arange(2.0), title="rows")
    inner1 = scp.CoordSet(
        scp.Coord(np.arange(3.0), title="first"),
        scp.Coord(np.arange(3.0, 6.0), title="second"),
    )
    # select uses 1-indexed ints: select(2) makes the second sub-coord default.
    inner1.select(2)
    inner2 = scp.CoordSet(
        scp.Coord(np.arange(3.0, 6.0), title="first"),
        scp.Coord(np.arange(6.0, 9.0), title="second"),
    )
    ds1 = scp.NDDataset(np.ones((2, 3)), coordset=[y, inner1])
    ds2 = scp.NDDataset(np.ones((2, 3)), coordset=[y, inner2])

    result = concatenate(ds1, ds2, dims="x")
    # The default index should be preserved from the first coordset.
    assert result.coordset["x"]._default == 1


def test_concatenate_none_coord_warns():
    """Concatenate warns when a dataset has None coordinate data along dim."""
    x1 = scp.Coord(np.arange(3.0))
    x2 = scp.Coord(None, size=3)
    ds1 = scp.NDDataset(np.ones((2, 3)), coordset=[scp.Coord(np.arange(2.0)), x1])
    ds2 = scp.NDDataset(np.ones((2, 3)), coordset=[scp.Coord(np.arange(2.0)), x2])

    with pytest.warns(UserWarning, match=".*coordinates.*None.*"):
        concatenate(ds1, ds2, dims="x")


def test_stack_regression():
    """Stack creates prepended dimension and delegates to concatenate."""
    y = scp.Coord(np.arange(2.0), title="rows")
    x = scp.Coord(np.arange(3.0), title="cols")
    ds1 = scp.NDDataset(np.ones((2, 3)), coordset=[y, x])
    ds2 = scp.NDDataset(np.ones((2, 3)), coordset=[y, x])

    result = stack(ds1, ds2)
    assert result.shape == (2, 2, 3)
    # The new leading dimension coordinate should have two labels
    assert len(result.dims) == 3
