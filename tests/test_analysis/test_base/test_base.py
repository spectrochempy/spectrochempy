#
# # ======================================================================================
# # Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# # CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# # See full LICENSE agreement in the root directory.
# # ======================================================================================
# ruff: noqa

import sys
from os import environ

import numpy as np
import pytest
import traitlets as tr

import spectrochempy as scp
from spectrochempy.analysis._base._analysisbase import (
    AnalysisConfigurable,
    DecompositionAnalysis,
    LinearRegressionAnalysis,
    NotFittedError,
)
from spectrochempy.utils import docutils as chd


# test docstring
# but this is not intended to work with the debugger - use run instead of debug!
@pytest.mark.skipif(
    environ.get("PYDEVD_LOAD_VALUES_ASYNC", None),
    reason="debug mode cause error when checking docstrings",
)
def test_base_docstrings():
    chd.PRIVATE_CLASSES = []  # do not test private class docstring
    module = "spectrochempy.analysis._base._analysisbase"

    # Base exclusions for all Python versions
    base_exclude = ["SA01", "EX01", "ES01", "GL11", "GL08", "PR01"]

    # Temporary workaround for Python 3.11 numpydoc/docstring-generation
    # inconsistencies. SS04 errors (summary heading whitespaces) appear on
    # Python 3.11 due to differences in how docstrings are parsed/generated.
    # Validation remains strict on Python 3.12+.
    # TODO: Revisit when Python 3.11 support is dropped or numpydoc is updated.
    if sys.version_info[:2] == (3, 11):
        base_exclude += ["SS04"]

    # analyse AnalysisConfigurable
    chd.check_docstrings(
        module,
        obj=scp.analysis._base._analysisbase.AnalysisConfigurable,
        exclude=base_exclude,
    )

    # analyse DecompositionAnalysis
    chd.check_docstrings(
        module,
        obj=scp.analysis._base._analysisbase.DecompositionAnalysis,
        exclude=base_exclude,
    )

    # analyse LinearRegressionAnalysis
    chd.check_docstrings(
        module,
        obj=scp.analysis._base._analysisbase.LinearRegressionAnalysis,
        exclude=base_exclude,
    )
    # analyse CrossDecompositionAnalysis
    chd.check_docstrings(
        module,
        obj=scp.analysis._base._analysisbase.CrossDecompositionAnalysis,
        exclude=base_exclude,
    )


class Foo(AnalysisConfigurable):
    a = tr.Integer(None, allow_none=True, help="trait a").tag(config=True)
    b = tr.Unicode("foo_b", help="this is a trait b").tag(config=True)

    def _fit(self, X):
        return X


def test_analysisconfigurable():
    foo = Foo()
    assert isinstance(foo, AnalysisConfigurable)
    assert foo.name == "Foo"
    assert isinstance(foo.parent, tr.config.Application)
    assert foo.a is None
    # assert foo.help.startswith("Foo.a : Int\n")
    assert foo.traits()["b"].help == "this is a trait b"
    assert foo.config["Foo"] == {}, "not initialized"
    # assert foo.log == ""

    with pytest.raises(tr.TraitError):
        # not an integer
        foo.a = 10.1

    # set conf. at init
    foo.reset()  # needed to delete json (in case it was already created)
    cd = scp.get_config_dir()
    assert not (cd / "Foo.json").exists()
    foo = Foo(a=1)
    assert foo.a == 1
    assert (cd / "Foo.json").exists()

    # wrong parameters
    with pytest.raises(KeyError):
        _ = Foo(j=1)


def test_analysisconfigurable_validation():
    # case of 2D array (the classical case for decomposition problems)
    foo = Foo()
    assert foo.name == "Foo"
    with pytest.raises(NotFittedError):
        _ = foo.X

    X = [[1, 2], [2, 2], [1, 3]]
    foo.fit(X)
    assert foo.X._implements("NDDataset")
    X1 = foo.X
    X1[1, 0] = scp.MASKED  # this mask both row 1 and column 0
    assert np.all(X1.mask == [[True, False], [True, True], [True, False]])
    foo.fit(X1)

    # resulting X should have the same mask
    assert np.all(foo.X.mask == X1.mask)

    # 1D X
    X = [1.0, 2.0, 3.0]
    foo.fit(X)
    # A column has been added
    assert repr(foo.X) == "NDDataset: [float64] unitless (shape: (u:1, x:3))"

    X = scp.NDDataset(np.arange(3) + 1.5, coordset=[range(3)], units="m")
    foo.fit(X)
    assert repr(foo.X) == "NDDataset: [float64] m (shape: (u:1, x:3))"

    X = scp.NDDataset(np.arange(3) + 1.5, coordset=[range(3)], units="m")
    # with a mask
    X[1] = scp.MASKED
    foo.fit(X)
    assert repr(foo.X) == "NDDataset: [float64] m (shape: (u:1, x:3))"


# def test_decompositionanalysis():
#
#     X = scp.NDDataset(np.arange(3) + 1.5, coordset=[range(3)], units="m")
#     # with a mask
#     foo.fit(X)
#     assert repr(foo.X) == "NDDataset: [float64] m (shape: (y:1, x:3))"
