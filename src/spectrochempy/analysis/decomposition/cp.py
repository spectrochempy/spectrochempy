# ======================================================================================
# Copyright (c) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Deprecated import path for CP/PARAFAC tensor decomposition.

CP has been moved to the ``spectrochempy-tensor`` plugin.
Use ``scp.tensor.CP`` instead.

.. deprecated:: 0.9.0
    Importing from ``spectrochempy.analysis.decomposition.cp`` is deprecated.
    Use the ``spectrochempy-tensor`` plugin API (``scp.tensor.CP``).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

__all__ = ["CP"]
__configurables__ = ["CP"]

_CP_DEPRECATION_MSG = (
    "Importing {name!r} from spectrochempy.analysis.decomposition.cp "
    "is deprecated since SpectroChemPy 0.9.0 and will be removed in 0.11.0. "
    "Use scp.tensor.{name} instead."
)

if TYPE_CHECKING:
    from spectrochempy_tensor import CP


def __getattr__(name: str):
    if name == "CP":
        warnings.warn(
            _CP_DEPRECATION_MSG.format(name=name),
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            from spectrochempy_tensor import CP as _CP
        except ImportError:
            raise ImportError(
                "The tensor plugin (spectrochempy-tensor) is required. "
                "Install it with: pip install spectrochempy[tensor]"
            ) from None

        return _CP

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return __all__
