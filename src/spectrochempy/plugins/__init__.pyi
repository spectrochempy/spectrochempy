# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

# ruff: noqa

__all__ = [
    "base",
    "capabilities",
    "contributions",
    "deps",
    "features",
    "hooks",
    "inspection",
    "lifecycle",
    "manager",
    "namespace",
    "proxies",
    "registries",
    "registry",
]

from . import base
from . import capabilities
from . import contributions
from . import deps
from . import features
from . import hooks
from . import inspection
from . import lifecycle
from . import manager
from . import namespace
from . import proxies
from . import registries
from . import registry

def __call__(verbose: bool = False): ...
