# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
MCR-ALS constraint classes.

This module re-exports the public constraint classes for MCR-ALS. They
describe scientific prior knowledge about the concentration (``"C"``)
or spectral (``"St"``) profiles estimated by :class:`MCRALS
<spectrochempy.analysis.decomposition.mcrals.MCRALS>`.

The classes are declarative containers and validators only. They are
**not yet connected** to the internal ALS constraint engine.

Example::

    from spectrochempy.analysis import constraints

    c = [
        constraints.NonNegative("C"),
        constraints.Closure("C"),
        constraints.ReferenceProfile("St", component=0, data=spectrum),
        constraints.ModelProfile("C", model=kinetic_model),
    ]
"""

from spectrochempy.analysis.decomposition.mcrals_constraints import (  # noqa: F401
    Closure,
)
from spectrochempy.analysis.decomposition.mcrals_constraints import (  # noqa: F401
    ComponentPresence,
)
from spectrochempy.analysis.decomposition.mcrals_constraints import (  # noqa: F401
    Constraint,
)
from spectrochempy.analysis.decomposition.mcrals_constraints import (  # noqa: F401
    FixedValues,
)
from spectrochempy.analysis.decomposition.mcrals_constraints import (  # noqa: F401
    ModelProfile,
)
from spectrochempy.analysis.decomposition.mcrals_constraints import (  # noqa: F401
    Monotonic,
)
from spectrochempy.analysis.decomposition.mcrals_constraints import (  # noqa: F401
    NonNegative,
)
from spectrochempy.analysis.decomposition.mcrals_constraints import (  # noqa: F401
    ReferenceProfile,
)
from spectrochempy.analysis.decomposition.mcrals_constraints import (  # noqa: F401
    Selectivity,
)
from spectrochempy.analysis.decomposition.mcrals_constraints import (  # noqa: F401
    Trilinear,
)
from spectrochempy.analysis.decomposition.mcrals_constraints import (  # noqa: F401
    Unimodal,
)
from spectrochempy.analysis.decomposition.mcrals_constraints import (  # noqa: F401
    ZeroRegion,
)

__all__ = [
    "Closure",
    "ComponentPresence",
    "Constraint",
    "FixedValues",
    "ModelProfile",
    "Monotonic",
    "NonNegative",
    "ReferenceProfile",
    "Selectivity",
    "Trilinear",
    "Unimodal",
    "ZeroRegion",
]
