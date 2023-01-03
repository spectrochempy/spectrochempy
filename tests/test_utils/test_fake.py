# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
from spectrochempy.utils import generate_fake  # , show


def test_fake():

    nd, specs, concs = generate_fake()
    assert nd.shape == (50, 4000)

    # specs.plot()
    # concs.plot()
    # nd.plot()
    # show()
