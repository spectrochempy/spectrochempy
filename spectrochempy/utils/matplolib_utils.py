# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================


import os
import shutil as sh
from pkg_resources import resource_filename

__all__ = ['install_styles']


def install_styles():
    """
    Install matplotlib styles

    """
    stylelib = os.path.expanduser(
            os.path.join('~', '.matplotlib', 'stylelib'))
    if not os.path.exists(stylelib):
        os.mkdir(stylelib)

    styles_path = resource_filename('scp_data', 'stylesheets')

    styles = os.listdir(styles_path)

    for style in styles:
        src = os.path.join(styles_path, style)
        dest = os.path.join(stylelib, style)
        sh.copy(src, dest)
