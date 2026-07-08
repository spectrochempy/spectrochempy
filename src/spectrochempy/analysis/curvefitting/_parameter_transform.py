# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Solver-specific parameter-space transforms.

These functions convert parameters between *physical space* (the space that
users and models operate in) and the *unbounded optimizer space* that solvers
require internally.

Responsibilities
----------------
- Map bounded physical parameters to the unbounded real line so that
  gradient-based and simplex-based optimizers can search freely.
- Map unbounded optimizer values back to the bounded physical space.

Why these belong to the solver layer
-------------------------------------
- Bounds on a model parameter are a solver concern: the model itself does not
  need to know how the optimizer enforces constraints.
- The transforms (arcsin, sqrt, etc.) are dictated by the optimizer's need for
  an unbounded search space, not by the mathematical definition of the model.
- Different solvers may want different transforms; keeping them separate from
  the model representation makes it possible to change the transform strategy
  without touching model-definition code.

These functions operate on atomic values and bounds only — they have no
dependency on ``FitParameters``, ``_FitModelSpec``, or any other
model-definition object.
"""

import sys

import numpy as np

# ======================================================================================
# Threshold constants (preserved from the original FitParameters implementation)
# ======================================================================================

# Bounds equal to or beyond these thresholds are treated as "unbounded".
# The original implementation uses these to detect the sentinel values that
# the parser substitutes for "none" (open) bounds.
_LOB_THRESHOLD = -0.1 / sys.float_info.epsilon
_UPB_THRESHOLD = +0.1 / sys.float_info.epsilon


# ======================================================================================
# Public transform functions
# ======================================================================================


def _to_internal(value, lob, upb):
    """
    Convert a parameter *value* from physical space to optimizer (unbounded) space.

    Parameters
    ----------
    value : float
        The parameter value in physical space.
    lob : float or None
        Lower bound in physical space (``None`` means unbounded below).
    upb : float or None
        Upper bound in physical space (``None`` means unbounded above).

    Returns
    -------
    float
        The transformed value in unbounded optimizer space.
    """
    is_lob = lob is not None and lob > _LOB_THRESHOLD
    is_upb = upb is not None and upb < _UPB_THRESHOLD

    if is_lob and is_upb:
        lob_adj = min(value, lob)
        upb_adj = max(value, upb)
        return np.arcsin((2.0 * (value - lob_adj) / (upb_adj - lob_adj)) - 1.0)

    if is_upb:
        upb_adj = max(value, upb)
        return np.sqrt((upb_adj - value + 1.0) ** 2 - 1.0)

    if is_lob:
        lob_adj = min(value, lob)
        return np.sqrt((value - lob_adj + 1.0) ** 2 - 1.0)

    return value


def _to_external(pi, lob, upb):
    """
    Convert a parameter value from optimizer (unbounded) space to physical space.

    Parameters
    ----------
    pi : float or list of float
        The value(s) in unbounded optimizer space.
    lob : float or None
        Lower bound in physical space (``None`` means unbounded below).
    upb : float or None
        Upper bound in physical space (``None`` means unbounded above).

    Returns
    -------
    float or list of float
        The transformed value(s) in physical space.  Returns a bare float when
        *pi* is a scalar or a single-element list; returns a list when *pi*
        is a multi-element list.
    """
    is_lob = lob is not None and lob > _LOB_THRESHOLD
    is_upb = upb is not None and upb < _UPB_THRESHOLD

    if not isinstance(pi, list):
        pi = [pi]

    pe = []
    for item in pi:
        if is_lob and is_upb:
            pei = lob + ((upb - lob) / 2.0) * (np.sin(item) + 1.0)
        elif is_upb:
            pei = upb + 1.0 - np.sqrt(item**2 + 1.0)
        elif is_lob:
            pei = lob - 1.0 + np.sqrt(item**2 + 1.0)
        else:
            pei = item
        pe.append(pei)

    if len(pe) == 1:
        return pe[0]

    return pe
