# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import os
import sys
from pkgutil import walk_packages

from traitlets import import_item


# ======================================================================================
# PACKAGE and API UTILITIES
# ======================================================================================
def list_packages(package):
    """
    Return a list of the names of a package and its subpackages.

    This only works if the package has a :attr:`__path__` attribute, which is
    not the case for some (all?) of the built-in packages.
    """
    # Based on response at
    # http://stackoverflow.com/questions/1707709.

    names = [package.__name__]
    for __, name, __ in walk_packages(
        package.__path__, prefix=package.__name__ + ".", onerror=lambda x: None
    ):
        names.append(name)

    return sorted(names)


def generate_api(api_path, configurables=False):
    # from spectrochempy.application import debug_
    from spectrochempy.core.dataset.nddataset import NDDataset

    # name of the package
    dirname, name = os.path.split(os.path.split(api_path)[0])

    if not dirname.endswith("spectrochempy"):
        dirname, _name = os.path.split(dirname)
        name = _name + "." + name
    pkgs = sys.modules["spectrochempy.%s" % name]
    api = sys.modules["spectrochempy.%s.api" % name]

    pkgs = list_packages(pkgs)

    __all__ = []
    __configurables__ = []

    for pkg in pkgs:
        if pkg.endswith("api") or "test" in pkg:
            continue
        # try:
        pkg = import_item(pkg)
        # except Exception as exc:
        # debug_(f"When trying to load:{pkg} : {exc}")
        # if not hasattr(pkg, "__all__"):
        #    continue
        #    raise ImportError(pkg)
        if not hasattr(pkg, "__all__"):
            continue

        # Some  methods are classmethod of NDDatasets
        dmethods = getattr(pkg, "__dataset_methods__", [])
        for item in dmethods:
            setattr(NDDataset, item, getattr(pkg, item))

        # set general method for the current package API
        a = getattr(pkg, "__all__", [])
        __all__ += a
        for item in a:
            obj = getattr(pkg, item)
            setattr(api, item, obj)
            confs = getattr(pkg, "__configurables__", [])
            for conf in confs:
                __configurables__.append(getattr(pkg, conf))

    # if required get also a list of configurables
    if configurables:
        return __all__, __configurables__

    return __all__


def get_pkg_path(data_name, package=None):
    data_name = os.path.normpath(data_name)

    path = os.path.dirname(import_item(package).__file__)
    path = os.path.join(path, data_name)

    if not os.path.isdir(path):  # pragma: no cover
        return os.path.dirname(path)

    return path
