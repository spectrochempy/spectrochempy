# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================




"""
This package provides classes and functions
to manage a project or a set of projects.

"""

import sys
from traitlets import import_item

from ..utils import list_packages

name = 'projects'
pkgs = sys.modules['spectrochempy.%s' % name]
api = sys.modules['spectrochempy.%s.api' % name]

pkgs = list_packages(pkgs)

__all__ = []

for pkg in pkgs:
    if pkg.endswith('api'):
        continue
    try:
        pkg = import_item(pkg)
    except:
        print('pkg', pkg)

    if not hasattr(pkg, '__all__'):
        continue
    a = getattr(pkg, '__all__')
    __all__ += a
    for item in a:
        setattr(api, item, getattr(pkg, item))

# ===========================================================================
if __name__ == '__main__':
    pass