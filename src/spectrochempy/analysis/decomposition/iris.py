# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Deprecated import path for IRIS analysis.

IRIS and IrisKernel have been moved to the ``spectrochempy-iris`` plugin.
Use ``scp.iris.IRIS`` and ``scp.iris.IrisKernel`` instead.

.. deprecated:: 0.9.0
    Importing from ``spectrochempy.analysis.decomposition.iris`` is deprecated.
    Use the ``spectrochempy-iris`` plugin API (``scp.iris.*``).
"""

from __future__ import annotations

import warnings

__all__ = ["IrisKernel", "IRIS"]  # noqa: F822 — resolved via __getattr__
__configurables__ = ["IRIS"]

_IRIS_DEPRECATION_MSG = (
    "Importing {name!r} from spectrochempy.analysis.decomposition.iris "
    "is deprecated since SpectroChemPy 0.9.0 and will be removed in 0.10.0. "
    "Use scp.iris.{name} instead."
)


def __getattr__(name: str):
    if name in __all__:
        warnings.warn(
            _IRIS_DEPRECATION_MSG.format(name=name),
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            from spectrochempy_iris import IRIS as _IRIS
            from spectrochempy_iris import IrisKernel as _IrisKernel
        except ImportError:
            raise ImportError(
                "The IRIS plugin (spectrochempy-iris) is required. "
                "Install it with: pip install spectrochempy[iris]"
            ) from None

        return _IRIS if name == "IRIS" else _IrisKernel

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return __all__
