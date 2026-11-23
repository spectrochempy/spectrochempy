"""Private evaluation of Python model functions, independent of the script DSL."""

import numpy as np


def _evaluate_callable(
    function, x, parameters, /, *, y=None, amplitude_mode="area", **kwargs
):
    """Evaluate a function with physical parameters and optional height scaling."""
    x = np.array(x, dtype=np.float64)
    if y is not None:
        y = np.array(y, dtype=np.float64)

    val = (
        function(x, *parameters, **kwargs)
        if y is None
        else function(x, y, *parameters, **kwargs)
    )

    if amplitude_mode.lower() == "height":
        ampl = parameters[0]
        val = ampl * val / val.max()
    return val
