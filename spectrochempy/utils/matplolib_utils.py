# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to provide a general
# API for displaying, processing and analysing spectrochemical data.
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
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
