# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

import pytest

from spectrochempy import application, APIref, log, version

application.set_loglevel('WARNING')
from spectrochempy import *

styles = ['poster','talk','scpy','sans','serif','grayscale','notebook','paper']
@pytest.mark.parametrize('style',styles)
def test_styles(style):
    try:
        plt.style.use(style)
    except OSError:
        plt.style.use(os.path.join(datadir.stylesheets, style+'.mplstyle'))
