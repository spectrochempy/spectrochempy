# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
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

import uuid

from traitlets import (Dict, List, Bool, Instance, Unicode, HasTraits, This,
                       Any, default)
from traitlets.config.configurable import Configurable


__all__ = ['ProjectsOptions']


# ============================================================================
class ProjectsOptions(Configurable) :

    projects_directory = Unicode(help='location where all projects are '
                                     'strored by defauult').tag(config=True)

import warnings

__all__ = []

# ============================================================================
if __name__ == '__main__' :
    pass
