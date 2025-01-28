# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
import numpy as np
import pytest
import traitlets as tr

import spectrochempy as scp
from spectrochempy.utils.traits import NDDatasetType


def test_nddatasettype():
    notifications = []

    class Foo(tr.HasTraits):
        a = NDDatasetType()
        b = NDDatasetType(None, allow_none=True)
        c = NDDatasetType([1, 2, 3, 4])
        b = NDDatasetType(allow_none=True)

        @tr.observe("c")
        def _c_change(self, change):
            notifications.append(change)

    foo = Foo()

    assert foo.a == scp.NDDataset()
    assert foo.b.is_empty
    assert np.any(foo.c.data == np.array([1, 2, 3, 4]))

    with pytest.raises(tr.TraitError):
        foo.a = None
    foo.b = None
    foo.d = None

    foo.c = [1, 2, 3]
    assert foo.c.shape == (3,)
    assert len(notifications) == 1

    foo.c = np.array([1, 2])


# TODO test projecttype
