#  -*- coding: utf-8 -*-

#  =====================================================================================================================
#    Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#    CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory
#  =====================================================================================================================

from spectrochempy.core.scripts.script import Script
from spectrochempy.utils import testing


def test_compare(IR_dataset_1D, simple_project):
    # dataset comparison

    nd1 = IR_dataset_1D.copy()
    nd2 = nd1.copy()

    testing.assert_dataset_equal(nd1, nd2)

    nd3 = nd1.copy()
    nd3.title = 'ddd'

    with testing.raises(AssertionError):
        testing.assert_dataset_equal(nd1, nd3)

    nd4 = nd1.copy()
    nd4.data += 0.001

    with testing.raises(AssertionError):
        testing.assert_dataset_equal(nd1, nd4)

    testing.assert_dataset_almost_equal(nd1, nd4, decimal=3)

    with testing.raises(AssertionError):
        testing.assert_dataset_almost_equal(nd1, nd4, decimal=4)

    # project comparison

    proj1 = simple_project.copy()
    proj1.name = 'PROJ1'
    proj2 = proj1.copy()
    proj2.name = 'PROJ2'

    testing.assert_project_equal(proj1, proj2)

    proj3 = proj2.copy()
    proj3.add_script(Script(content='print()', name='just_a_try'))

    with testing.raises(AssertionError):
        testing.assert_project_equal(proj1, proj3)
