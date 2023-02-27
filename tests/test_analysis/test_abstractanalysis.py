# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import pytest
import traitlets as tr

from spectrochempy.analysis.abstractanalysis import (
    AnalysisConfigurable,  # , DecompositionAnalysisConfigurable
)


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
    assert foo.log == ""

    with pytest.raises(tr.TraitError):
        # not an integer
        foo.a = 10.1
