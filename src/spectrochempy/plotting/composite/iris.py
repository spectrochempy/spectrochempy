# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Deprecated import path for IRIS plotting functions.

IRIS plotting functions have been moved to the ``spectrochempy-iris`` plugin.
Use ``scp.iris.plot_iris_lcurve``, ``scp.iris.plot_iris_distribution``, etc. instead.

.. deprecated:: 0.9.0
    Importing from ``spectrochempy.plotting.composite.iris`` is deprecated.
    Use the ``spectrochempy-iris`` plugin API (``scp.iris.*``).
"""

from __future__ import annotations

import warnings

__all__ = ["plot_iris_lcurve", "plot_iris_distribution", "plot_iris_merit"]  # noqa: F822 — resolved via __getattr__

_IRIS_DEPRECATION_MSG = (
    "Importing {name!r} from spectrochempy.plotting.composite.iris "
    "is deprecated since SpectroChemPy 0.9.0 and will be removed in 0.11.0. "
    "Use scp.iris.{name} instead."
)


def __getattr__(name: str):
    if name in __all__:
        warnings.warn(
            _IRIS_DEPRECATION_MSG.format(name=name),
            DeprecationWarning,
            stacklevel=2,
        )
        from spectrochempy_iris._plotting import (  # noqa: PLC0415
            plot_iris_distribution as _f1,
        )
        from spectrochempy_iris._plotting import (
            plot_iris_lcurve as _f2,  # noqa: PLC0415
        )
        from spectrochempy_iris._plotting import plot_iris_merit as _f3  # noqa: PLC0415

        _mapping = {
            "plot_iris_distribution": _f1,
            "plot_iris_lcurve": _f2,
            "plot_iris_merit": _f3,
        }
        return _mapping[name]

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return __all__
