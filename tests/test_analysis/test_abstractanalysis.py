# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import numpy as np
import pytest
import traitlets as tr

import spectrochempy as scp
from spectrochempy.analysis._base import AnalysisConfigurable  # , DecompositionAnalysis


def test_analysisconfigurable():
    class BadFoo(AnalysisConfigurable):
        # required name not defined
        a = tr.Integer(help="trait a").tag(config=True)

    with pytest.raises(NameError):
        _ = BadFoo()

    class Foo(AnalysisConfigurable):
        name = tr.Unicode("Foo")
        a = tr.Integer(None, allow_none=True, help="trait a").tag(config=True)
        b = tr.Unicode("foo_b", help="this is a trait b").tag(config=True)

    foo = Foo()
    assert isinstance(foo, AnalysisConfigurable)
    assert foo.name == "Foo"
    assert isinstance(foo.parent, tr.config.Application)
    assert not foo.description
    assert foo.a is None
    # assert foo.help.startswith("Foo.a : Int\n")
    assert foo.traits()["b"].help == "this is a trait b"
    assert foo.config["Foo"] == {}, "not initialized"
    assert foo.section == "Foo", "section in configuration"
    # assert foo.log == ""

    with pytest.raises(tr.TraitError):
        # not an integer
        foo.a = 10.1


def test_analysisconfigurable_validation():
    class Foo(AnalysisConfigurable):
        """a test for mask"""

        name = "Foo"

        def fit(self, X):
            self._X = X
            return self

    # case of 2D array (the classical case for decomposition problems)
    foo = Foo()

    X = [[1, 2], [2, 2], [1, 3]]
    foo.fit(X)
    assert foo.X._implements("NDDataset")
    X1 = foo.X
    X1[1, 0] = scp.MASKED  # this mask both row 1 and column 0
    assert np.all(X1.mask == [[True, False], [True, True], [True, False]])
    foo.fit(X1)
    assert np.all(foo.X.mask == [[True, False], [True, True], [True, False]])

    # 1D X
    X = [1.0, 2.0, 3.0]
    foo.fit(X)
    # A column has been added
    assert repr(foo.X) == "NDDataset: [float64] unitless (shape: (y:1, x:3))"

    X = scp.NDDataset(np.arange(3) + 1.5, coordset=[range(3)], units="m")
    foo.fit(X)
    assert repr(foo.X) == "NDDataset: [float64] m (shape: (y:1, x:3))"

    X = scp.NDDataset(np.arange(3) + 1.5, coordset=[range(3)], units="m")
    # with a mask
    X[1] = scp.MASKED
    foo.fit(X)
    assert repr(foo.X) == "NDDataset: [float64] m (shape: (y:1, x:3))"
