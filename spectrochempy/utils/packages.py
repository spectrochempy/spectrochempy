# -*- coding: utf-8 -*-
#
# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import sys
import os
from pkgutil import walk_packages

from traitlets import import_item


__all__ = [ 'list_packages', 'generate_api' ]

# ======================================================================================================================
# PACKAGE and API UTILITIES
# ======================================================================================================================

# ......................................................................................................................
def list_packages(package):
    """Return a list of the names of a package and its subpackages.

    This only works if the package has a :attr:`__path__` attribute, which is
    not the case for some (all?) of the built-in packages.
    """
    # Based on response at
    # http://stackoverflow.com/questions/1707709

    names = [package.__name__]
    for __, name, __ in walk_packages(package.__path__,
                                      prefix=package.__name__ + '.',
                                      onerror=lambda x: None):
        names.append(name)

    return names


# ......................................................................................................................
def generate_api(api_path):
    # name of the package

    dirname, name = os.path.split(os.path.split(api_path)[0])
    if not dirname.endswith('spectrochempy'):
        dirname, _name = os.path.split(dirname)
        name = _name + '.' + name
    pkgs = sys.modules['spectrochempy.%s' % name]
    api = sys.modules['spectrochempy.%s.api' % name]

    pkgs = list_packages(pkgs)

    __all__ = []

    for pkg in pkgs:
        if pkg.endswith('api') or "test" in pkg:
            continue
        try:
            pkg = import_item(pkg)
        except Exception:
            raise ImportError(pkg)
        if not hasattr(pkg, '__all__'):
            continue
        a = getattr(pkg, '__all__', [])
        dmethods = getattr(pkg, '__dataset_methods__', [])
        __all__ += a
        for item in a:

            # set general method for the current package API
            setattr(api, item, getattr(pkg, item))

            # some  methods are class method of NDDatasets
            if item in dmethods:
                from spectrochempy.core.dataset.nddataset import NDIO
                setattr(NDIO, item, getattr(pkg, item))

    return __all__

# ======================================================================================================================

if __name__ == '__main__':
    pass
