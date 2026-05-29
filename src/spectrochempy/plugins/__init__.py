# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import sys as _sys
from types import ModuleType as _ModuleType

import lazy_loader as _lazy_loader

from spectrochempy.plugins.inspection import inspect_plugins as _inspect_plugins

# --------------------------------------------------------------------------------------
# Lazy loading of sub-packages
# --------------------------------------------------------------------------------------
__getattr__, __dir__, __all__ = _lazy_loader.attach_stub(__name__, __file__)


class _CallablePluginModule(_ModuleType):
    """Make ``spectrochempy.plugins`` usable as ``scp.plugins()``."""

    def __call__(self, verbose: bool = False):
        return _inspect_plugins(verbose=verbose)


_sys.modules[__name__].__class__ = _CallablePluginModule
