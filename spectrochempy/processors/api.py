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




"""In this folder are  stored plugins that will extend the methods of a dataset

"""

# # register to dataset
# from ..dataset.nddataset import NDDataset
#
# from .autosub import autosub
# from .baseline import basecor
# from .concatenate import concatenate
# from .interpolate import align, interpolate
#
# setattr(NDDataset, 'autosub', autosub)
# setattr(NDDataset, 'align', align)
# setattr(NDDataset, 'concatenate', concatenate)
# setattr(NDDataset, 'basecor', basecor)
#
# # all (make this function also available as full API functions
#
# __all__ = ['autosub', 'align', 'concatenate', 'basecor', 'interpolate']
#

import sys

from traitlets import import_item

from spectrochempy.dataset.nddataset import NDDataset
from spectrochempy.utils import list_packages

name = 'processors'
pkgs = sys.modules['spectrochempy.%s' % name]
api = sys.modules['spectrochempy.%s.api' % name]

pkgs = list_packages(pkgs)

__all__ = []

for pkg in pkgs:
    if pkg.endswith('api'):
        continue
    pkg = import_item(pkg)
    if not hasattr(pkg, '__all__'):
        continue
    a = getattr(pkg, '__all__')
    __all__ += a
    for item in a:
        setattr(NDDataset, item, getattr(pkg, item))
        setattr(api, item, getattr(pkg, item))

# ===========================================================================
if __name__ == '__main__':
    pass