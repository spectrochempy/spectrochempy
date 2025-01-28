# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
from spectrochempy.utils.fake import generate_fake  # , show


def test_fake():
    nd, specs, concs = generate_fake()
    assert nd.shape == (50, 4000)

    # specs.plot()
    # concs.plot()
    # nd.plot()
    # show()
