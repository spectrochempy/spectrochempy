# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implements the MCRALS class.
"""

# DEVNOTE:
# API methods accessible as  scp.method or scp.class must be defined in __all__
# Configurable class (which requires a configuration file)
# must be declared in __configurable__

__all__ = ["MCRALS"]
__configurables__ = ["MCRALS"]

import base64
import logging
import warnings

import dill
import numpy as np
import scipy
import traitlets as tr
from sklearn import decomposition

from spectrochempy.analysis._base._analysisbase import (
    DecompositionAnalysis,
    NotFittedError,
    _wrap_ndarray_output_to_nddataset,
)
from spectrochempy.application import info_
from spectrochempy.extern.traittypes import Array
from spectrochempy.utils.decorators import deprecated, signature_has_configurable_traits
from spectrochempy.utils.docstrings import _docstring


# DEVNOTE:
# the following decorator allow to correct signature and docs of traitlets.HasTraits
# derived class
@signature_has_configurable_traits
class MCRALS(DecompositionAnalysis):
    _docstring.delete_params("DecompositionAnalysis.see_also", "MCRALS")

    __doc__ = _docstring.dedent(
        """
    Multivariate Curve Resolution Alternating Least Squares (MCRALS).

    :term:`MCR-ALS` ( ``Multivariate Curve Resolution Alternating Least Squares`` )
    resolve's a set (or several sets) of spectra :math:`X` of an evolving mixture
    (or a set of mixtures) into the spectra :math:`S^t` of "pure" species and their
    concentration profiles :math:`C`\ .

    In terms of matrix equation:

    .. math:: X = C.S^t + E

    where :math:`E` is the matrix of residuals.

    Parameters
    ----------
    %(AnalysisConfigurable.parameters)s

    See Also
    --------
    %(DecompositionAnalysis.see_also.no_MCRALS)s
    """
    )

    # Developer notes
    # ----------------
    # Private traits with internal validation (see Traitlets library for more
    # information)
    # Notice that variable not defined this way lack this type validation, so they are
    # more prone to errors.

    # ----------------------------------------------------------------------------------
    # Configuration parameters
    # They will be written in a file from which the default can be modified)
    # Obviously, the parameters can also be modified at runtime as usual by assignment.
    # ----------------------------------------------------------------------------------

    tol = tr.Float(
        0.1,
        help=(
            "Convergence criterion on the change of residuals (percent change of "
            "standard deviation of residuals)."
        ),
    ).tag(config=True)

    max_iter = tr.Integer(50, help="Maximum number of :term:`ALS` iteration.").tag(
        config=True
    )

    maxdiv = tr.Integer(
        5, help="Maximum number of successive non-converging iterations."
    ).tag(config=True)

    solverConc = tr.Enum(
        ["lstsq", "nnls", "pnnls"],
        default_value="lstsq",
        help=(
            r"""Solver used to get `C` from `X` and `St`\ .

- ``'lstsq'``\ : uses ordinary least squares with `~numpy.linalg.lstsq`
- ``'nnls'``\ : non-negative least squares (`~scipy.optimize.nnls`\ ) are applied
  sequentially on all profiles
- ``'pnnls'``\ : non-negative least squares (`~scipy.optimize.nnls`\ ) are applied on
  profiles indicated in `nonnegConc` and ordinary least squares on other profiles.
"""
        ),
    ).tag(config=True)

    nonnegConc = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value="all",
        help=(
            r"""Non-negativity constraint on concentrations.

- ``'all'``\ : all concentrations profiles are considered
  non-negative.
- `list` of indexes: the corresponding profiles are considered non-negative,
  not the others. For instance ``[0, 2]`` indicates that profile \#0 and \#2
  are non-negative while profile \#1 *can* be negative.
- ``[]``\ : all profiles can be negative."""
        ),
    ).tag(config=True)

    unimodConc = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value="all",
        help=(
            r"""Unimodality constraint on concentrations.

- ``'all'``\ : all concentrations profiles are considered unimodal.
- `list` of indexes: the corresponding profiles are considered unimodal, not the others.
  For instance ``[0, 2]`` indicates that profile ``#0`` and ``#2`` are unimodal while
  profile ``#1`` *can* be multimodal.
- ``[]``\ : all profiles can be multimodal."""
        ),
    ).tag(config=True)

    unimodConcMod = tr.Enum(
        ["strict", "smooth"],
        default_value="strict",
        help=(
            r"""Method to apply unimodality.

- ``'strict'``\ : values deviating from :term:`unimodality` are reset to the value of the
  previous point.
- ``'smooth'``\ : both values (deviating point and previous point) are modified to avoid
  steps in the concentration profile."""
        ),
    ).tag(config=True)

    unimodConcTol = tr.Float(
        default_value=1.1,
        help=(
            r"""Tolerance parameter for :term:`unimodality`\ .

Correction is applied only if:

- ``C[i,j] > C[i-1,j] * unimodTol`` on the decreasing branch of profile ``#j``\ ,
- ``C[i,j] < C[i-1,j] * unimodTol`` on the increasing branch of profile ``#j``\ ."""
        ),
    ).tag(config=True)

    monoDecConc = tr.List(
        default_value=[],
        help=(
            r"""Monotonic decrease constraint on concentrations.

- ``[]``\ : no constraint is applied.
- `list` of indexes: the corresponding profiles are considered to decrease
  monotonically, not the others. For instance ``[0, 2]`` indicates that profile ``#0``
  and ``#2`` are decreasing while profile ``#1`` *can* increase."""
        ),
    ).tag(config=True)

    monoDecTol = tr.Float(
        default_value=1.1,
        help=r"""Tolerance parameter for monotonic decrease.

Correction is applied only if: ``C[i,j] > C[i-1,j] * unimodTol`` .""",
    ).tag(config=True)

    monoIncConc = tr.List(
        default_value=[],
        help=(
            r"""Monotonic increase constraint on concentrations.

- ``[]``\ : no constraint is applied.
- `list` of indexes: the corresponding profiles are considered to increase
  monotonically, not the others. For instance ``[0, 2]`` indicates that profile ``#0``
  and ``#2`` are increasing while profile ``#1`` *can* decrease."""
        ),
    ).tag(config=True)

    monoIncTol = tr.Float(
        default_value=1.1,
        help="""Tolerance parameter for monotonic decrease.

Correction is applied only if ``C[i,j] < C[i-1,j] * unimodTol`` along profile ``#j``\ .""",
    ).tag(config=True)

    unimodConc = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value="all",
        help=(
            r"""Unimodality constraint on concentrations.

- ``'all'``\ : all concentrations profiles are considered unimodal.
- `list` of indexes: the corresponding profiles are considered unimodal, not the others.
  For instance ``[0, 2]`` indicates that profile ``#0`` and ``#2`` are unimodal while
  profile ``#1`` *can* be multimodal.
- ``[]``\ : all profiles can be multimodal."""
        ),
    ).tag(config=True)

    closureConc = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value=[],
        help=(
            r"""Defines the concentration profiles subjected to closure constraint.

- ``[]``\ : no constraint is applied.
- ``'all'``\ : all profile are constrained so that their weighted sum equals the
  `closureTarget`
- `list` of indexes: the corresponding profiles are constrained so that their weighted sum
  equals `closureTarget`\ ."""
        ),
    ).tag(config=True)

    closureTarget = tr.Union(
        (tr.Enum(["default"]), Array()),
        default_value="default",
        help=(
            r"""The value of the sum of concentrations profiles subjected to closure.

- ``'default'``\ : the total concentration is set to ``1.0`` for all observations.
- :term:`array-like` of size :term:`n_observations`: the values of concentration for
  each observation. Hence, ``np.ones(X.shape[0])`` would be equivalent to
  ``'default'``\ ."""
        ),
    ).tag(config=True)

    closureMethod = tr.Enum(
        ["scaling", "constantSum"],
        default_value="scaling",
        help=(
            r"""The method used to enforce :term:`closure` (:cite:t:`omidikia:2018`).

- ``'scaling'`` recompute the concentration profiles using least squares:

  .. math::

     C \leftarrow C \cdot \textrm{diag} \left( C_L^{-1} c_t \right)

  where :math:`c_t` is the vector given by `closureTarget` and :math:`C_L^{-1}`
  is the left inverse of :math:`C`\ .
- ``'constantSum'`` normalize the sum of concentration profiles to `closureTarget`\ .
"""
        ),
    ).tag(config=True)

    hardConc = tr.List(
        default_value=[],
        help=(
            r"""Defines hard constraints on the concentration profiles.

- ``[]``\ : no constraint is applied.
- `list` of indexes: the corresponding profiles will set by `getConc`\ ."""
        ),
    ).tag(config=True)

    getConc = tr.Union(
        (tr.Callable(), tr.Unicode()),
        default_value=None,
        allow_none=True,
        help=(
            r"""An external function that provide ``len(hardConc)`` concentration
profiles.

It should be using one of the following syntax:

- ``getConc(Ccurr, *argsGetConc, **kwargsGetConc) -> hardC``
- ``getConc(Ccurr, *argsGetConc, **kwargsGetConc) -> hardC, newArgsGetConc``
- ``getConc(Ccurr, *argsGetConc, **kwargsGetConc) -> hardC, newArgsGetConc,
  extraOutputGetConc``

where:

- ``Ccurr`` is the current `C` dataset,
- ``argsGetConc`` are the parameters needed to completely specify the function.
- ``hardC`` is a `~numpy.ndarray` or `NDDataset` of shape
  (:term:`n_observations` , len(``hardConc``\ ),
- ``newArgsGetConc`` are the updated parameters for the next iteration (can be `None`),
- ``extraOutputGetConc`` can be any other relevant output to be kept in
  ``extraOutputGetConc`` attribute, a list of ``extraOutputGetConc`` at each MCR ALS
  iteration.

.. note::
    ``getConc`` can be also a serialized function created using dill and base64
    python libraries. Normally not used directly, it is here for internal
    process."""
        ),
    ).tag(config=True)

    argsGetConc = tr.Tuple(
        default_value=(),
        help="Supplementary positional arguments passed to the external function.",
    ).tag(config=True)

    kwargsGetConc = tr.Dict(
        default_value={},
        help="Supplementary keyword arguments passed to the external function.",
    ).tag(config=True)

    getC_to_C_idx = tr.Union(
        (tr.Enum(["default"]), tr.List()),
        default_value="default",
        help=(
            r"""Correspondence of the profiles returned by `getConc`
and `C[:,hardConc]`\ .

- ``'default'``: the profiles correspond to those of `C[:,hardConc]`. This is equivalent
  to ``range(len(hardConc))``
- `list` of indexes or of `None`. For instance ``[2, 1, 0]`` indicates that the
  third profile returned by `getC` (index ``2``\ ) corresponds to the 1st profile of
  `C[:, hardConc]`\ , the 2nd returned profile (index ``1``\ ) corresponds to
  second profile of `C[:, hardConc]`, etc..."""
        ),
    ).tag(config=True)

    solverSpec = tr.Enum(
        ["lstsq", "nnls", "pnnls"],
        default_value="lstsq",
        help=(
            r"""Solver used to get `St` from `X` and `C`\ .

- ``'lstsq'``\ : uses ordinary least squares with `~numpy.linalg.lstsq`
- ``'nnls'``\ : non-negative least squares (`~scipy.optimize.nnls`\ ) are applied
  sequentially on all profiles
- ``'pnnls'``\ : non-negative least squares (`~scipy.optimize.nnls`\ ) are applied on
  profiles indicated in `nonnegConc` and ordinary least squares on other profiles."""
        ),
    ).tag(config=True)

    nonnegSpec = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value="all",
        help=(
            r"""Non-negativity constraint on spectra.

- ``'all'``\ : all profiles are considered non-negative.
- `list` of indexes : the corresponding profiles are considered non-negative, not the
  others. For instance ``[0, 2]`` indicates that profile ``#0`` and ``#2`` are
  non-negative while profile ``#1`` *can* be negative.
- ``[]``\ : all profiles can be negative."""
        ),
    ).tag(config=True)

    normSpec = tr.Enum(
        [None, "euclid", "max"],
        default_value=None,
        help=(
            r"""Defines whether the spectral profiles should be normalized.

- `None`\ : no normalization is applied.
- ``'euclid'``\ : spectra are normalized with respect to their total area,
- ``'max'``\ : spectra are normalized with respect to their maximum value."""
        ),
    ).tag(config=True)

    unimodSpec = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value=[],
        help=(
            r"""Unimodality constraint on Spectra.

- ``[]``\ : all profiles can be multimodal.
- ``'all'``\ : all profiles are unimodal (equivalent to ``range(n_components)``\ ).
- array of indexes : the corresponding profiles are considered unimodal, not the others.
  For instance ``[0, 2]`` indicates that profile ``#0`` and ``#2`` are unimodal while
  profile ``#1`` *can* be multimodal."""
        ),
    ).tag(config=True)

    unimodSpecMod = tr.Enum(
        ["strict", "smooth"],
        default_value="strict",
        help=(
            r"""Method used to apply unimodality.

- ``'strict'``\ : values deviating from unimodality are reset to the value of the previous
  point.
- ``'smooth'``\ : both values (deviating point and previous point) are modified to avoid
  steps in the concentration profile."""
        ),
    ).tag(config=True)

    unimodSpecTol = tr.Float(
        default_value=1.1,
        help=(
            r"""Tolerance parameter for unimodality.

Correction is applied only if the deviating point ``St[j, i]`` is larger than
``St[j, i-1] * unimodSpecTol`` on the decreasing branch of profile
``#j``\ , or lower than ``St[j, i-1] * unimodTol`` on the increasing branch of
profile  ``#j``\ ."""
        ),
    ).tag(config=True)

    hardSpec = tr.List(
        default_value=[],
        help=(
            r"""Defines hard constraints on the spectral profiles.

- ``[]``\ : no constraint is applied.
- `list` of indexes : the corresponding profiles will set by `getSpec`\ ."""
        ),
    ).tag(config=True)

    getSpec = tr.Union(
        (tr.Callable(), tr.Unicode()),
        default_value=None,
        allow_none=True,
        help=(
            r"""An external function that will provide ``len(hardSpec)`` concentration
profiles.

It should be using one of the following syntax:

- ``getSpec(Stcurr, *argsGetSpec, **kwargsGetSpec) -> hardSt``
- ``getSpec(Stcurr, *argsGetSpec, **kwargsGetSpec) -> hardSt, newArgsGetSpec``
- ``getSpec(Stcurr, *argsGetSpec, **kwargsGetSpec) -> hardSt, newArgsGetSpec,
  extraOutputGetSpec``

with:

- ``Stcurr``\ : the current value of `St` in the :term:`ALS` loop,
- ``*argsGetSpec`` and ``**kwargsGetSpec``\ : the parameters needed to completely
  specify the function.
- ``hardSt``\ : `~numpy.ndarray` or `NDDataset` of shape
  ``(n_observations, len(hardSpec)``\ ,
- ``newArgsGetSpec``\ : updated parameters for the next ALS iteration (can be None),
- ``extraOutputGetSpec``\ : any other relevant output to be kept in
  `extraOutputGetSpec` attribute, a list of ``extraOutputGetSpec`` at each iterations.

.. note::
    ``getSpec`` can be also a serialized function created using dill and base64
    python libraries. Normally not used directly, it is here for internal process.
"""
        ),
    ).tag(config=True)

    argsGetSpec = tr.Tuple(
        default_value=(),
        help="Supplementary positional arguments passed to the external function.",
    ).tag(config=True)

    kwargsGetSpec = tr.Dict(
        default_value={},
        help="Supplementary keyword arguments passed to the external function.",
    ).tag(config=True)

    getSt_to_St_idx = tr.Union(
        (tr.Enum(["default"]), tr.List()),
        default_value="default",
        help=(
            r"""Correspondence between the indexes of the spectra returned by `getSpec`
and `St`.

- ``'default'``\ : the indexes correspond to those of `St`. This is equivalent
  to ``range(len(hardSpec))``\ .
- `list` of indexes : corresponding indexes in `St`, i.e. ``[2, None, 0]`` indicates that the
  first returned profile corresponds to the third `St` profile (index ``2``\ ), the 2nd
  returned profile does not correspond to any profile in `St`, the 3rd returned profile
  corresponds to the first `St` profile (index ``0`` )."""
        ),
    ).tag(config=True)

    storeIterations = tr.Bool(
        default_value=False,
        help=(r"""Whether to store the C and St generated at each iteration."""),
    ).tag(config=True)

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        *args,
        log_level=logging.WARNING,
        warm_start=False,
        **kwargs,
    ):
        if len(args) > 0:
            raise ValueError(
                "Passing arguments such as MCRALS(X, profile) "
                "is now deprecated. "
                "Instead, use MCRAL() followed by MCRALS.fit(X, profile). "
                "See the documentation and examples"
            )

        # warn about deprecation
        # ----------------------
        # We use pop to remove the deprecated argument before processing the rest
        # TODO: These arguments should be removed in version 0.7 probably

        # verbose
        if "verbose" in kwargs:
            deprecated("verbose", replace="log_level='INFO'", removed="0.7")
            verbose = kwargs.pop("verbose")
            if verbose:
                log_level = "INFO"

        # unimodTol deprecation
        if "unimodTol" in kwargs:
            deprecated("unimodTol", replace="unimodConcTol", removed="0.7")
            kwargs["unimodConcTol"] = kwargs.pop("unimodTol")

        # unimodMod deprecation
        if "unimodMod" in kwargs:
            deprecated("unimodMod", replace="unimodConcMod", removed="0.7")
            kwargs["unimodConcMod"] = kwargs.pop("unimodMod")

        # hardC_to_C_idx deprecation
        if "hardC_to_C_idx" in kwargs:
            deprecated("hardC_to_C_idx", replace="getC_to_C_idx", removed="0.7")
            kwargs["getC_to_C_idx"] = kwargs.pop("hardC_to_C_idx")

        # hardSt_to_St_idx deprecation
        if "hardSt_to_St_idx" in kwargs:
            deprecated("hardSt_to_St_idx", replace="getSt_to_St_idx", removed="0.7")
            kwargs["getSt_to_St_idx"] = kwargs.pop("hardSt_to_St_idx")

        # call the super class for initialisation
        super().__init__(
            log_level=log_level,
            warm_start=warm_start,
            **kwargs,
        )

        # deal with the callable that may have been serialized
        if self.getConc is not None and isinstance(self.getConc, str):
            self.getConc = dill.loads(base64.b64decode(self.getConc))
        if self.getSpec is not None and isinstance(self.getSpec, str):
            self.getSpec = dill.loads(base64.b64decode(self.getSpec))

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------

    def _solve_C(self, St):
        if self.solverConc == "lstsq":
            return _lstsq(St.T, self._X.data.T).T
        elif self.solverConc == "nnls":
            return _nnls(St.T, self._X.data.T).T
        elif self.solverConc == "pnnls":
            return _pnnls(St.T, self._X.data.T, nonneg=self.nonnegConc).T

    def _solve_St(self, C):
        if self.solverSpec == "lstsq":
            return _lstsq(C, self._X.data)
        elif self.solverSpec == "nnls":
            return _nnls(C, self._X.data)
        elif self.solverSpec == "pnnls":
            return _pnnls(C, self._X.data, nonneg=self.nonnegSpec)

    def _guess_profile(self, profile):
        # Set or guess an initial profile.

        if self._X_is_missing:
            return

        # check the dimensions compatibility
        # As the dimension of profile should match the initial shape
        # of X we use self._X_shape not self._X.shape (because for this masked columns
        # or rows have already been removed.
        if (self._X_shape[1] != profile.shape[1]) and (
            self._X_shape[0] != profile.shape[0]
        ):
            raise ValueError(
                f"None of the dimensions of the given profile "
                f"[{profile.shape}] correspond to any of those "
                f"of X [{self._X_shape}]."
            )

        # mask info
        if np.any(self._X_mask):
            masked_rows, masked_columns = self._get_masked_rc(self._X_mask)

        # make the profile
        if profile.shape[0] == self._X_shape[0]:
            # this should be a concentration profile.
            C = profile.copy()
            self._C0 = C
            self._n_components = C.shape[1]
            info_(
                f"Concentration profile initialized with {self._n_components} components"
            )

            # compute initial spectra (using X eventually masked)
            St = self._solve_St(C)
            self._St0 = St
            info_("Initial spectra profile computed")
            # if everything went well here, C and St are set, we return
            # after having removed the eventual C mask!
            if np.any(self._X_mask):
                C = C[~masked_rows]
            return C, St

        else:  # necessarily: profile.shape[1] == profile.shape[0]
            St = profile.copy()
            self._St0 = St
            self._n_components = St.shape[0]
            info_(f"Spectra profile initialized with {self._n_components} components")

            # compute initial spectra
            C = self._solve_C(St)
            self._C0 = C
            info_("Initial concentration profile computed")
            # if everything went well here, C and St are set, we return
            # after having removed the eventual St mask!
            if np.any(self._X_mask):
                St = St[:, ~masked_columns]
            # update the number of components
            return C, St

    @_wrap_ndarray_output_to_nddataset(units=None, title=None, typex="components")
    def _C_2_NDDataset(self, C):
        # getconc takes the C NDDataset as first argument (to take advantage
        # of useful metadata). But the current C in fit method is a ndarray (without
        # the masked rows and colums, nor the coord information: this
        # function will create the corresponding dataset
        return C

    @_wrap_ndarray_output_to_nddataset(units=None, title=None, typey="components")
    def _St_2_NDDataset(self, St):
        # getconc takes the C NDDataset as first argument (to take advantage
        # of useful metadata). The current St in fit method is a ndarray (without
        # the masked rows and columns, nor the coord information: this
        # function will create the corresponding dataset
        return St

    # ----------------------------------------------------------------------------------
    # Private validation methods and default getter
    # ----------------------------------------------------------------------------------
    @tr.validate("nonnegConc")
    def _validate_nonnegConc(self, proposal):
        if self._X_is_missing:
            return proposal.value
        nonnegConc = proposal.value
        if not self._n_components:  # not initialized or 0
            return nonnegConc
        if nonnegConc == "all":
            nonnegConc = np.arange(
                self._n_components
            ).tolist()  # IMPORTANT! .tolist, not list()
            # to get integer type not int64 which are not compatible with the setting
        elif np.any(nonnegConc) and (
            len(nonnegConc) > self._n_components
            or max(nonnegConc) + 1 > self._n_components
        ):  # note that we use np.any(nnonnegConc) instead of nnonnegConc != []
            # due to a deprecation warning from traitlets.
            raise ValueError(
                f"The profile has only {self._n_components} species, please check "
                f"the `nonnegConc` configuration (value: {nonnegConc})"
            )
        return nonnegConc

    @tr.validate("unimodConc")
    def _validate_unimodConc(self, proposal):
        if self._X_is_missing:
            return proposal.value
        unimodConc = proposal.value
        if not self._n_components:  # not initialized or 0
            return unimodConc
        if unimodConc == "all":
            unimodConc = np.arange(self._n_components).tolist()
        elif np.any(unimodConc) and (
            len(unimodConc) > self._n_components
            or max(unimodConc) + 1 > self._n_components
        ):
            raise ValueError(
                f"The profile has only {self._n_components} species, please check the "
                f"`unimodConc` configuration (value: {unimodConc})"
            )
        return unimodConc

    @tr.validate("closureConc")
    def _validate_closureConc(self, proposal):
        if self._X_is_missing:
            return proposal.value
        closureConc = proposal.value
        if closureConc == "all":
            closureConc = np.arange(self._n_components)
        elif len(closureConc) > self._n_components:
            raise ValueError(
                f"The model contains only {self._n_components} components, please check "
                f"the 'closureConc' configuration (value: {closureConc})"
            )
        return closureConc

    @tr.validate("closureTarget")
    def _validate_closureTarget(self, proposal):
        if self._X_is_missing:
            return proposal.value
        closureTarget = proposal.value
        ny = self.X.shape[0]
        if isinstance(closureTarget, str):
            if closureTarget == "default":
                closureTarget = np.ones(ny)
        elif len(closureTarget) != ny:
            raise ValueError(
                f"The data contain only {ny} observations, please check "
                f"the 'closureTarget' configuration (value: {closureTarget})"
            )
        return closureTarget

    @tr.validate("getC_to_C_idx")
    def _validate_getC_to_C_idx(self, proposal):
        if self._X_is_missing:
            return proposal.value
        getC_to_C_idx = proposal.value
        if not self._n_components:  # not initialized or 0
            return getC_to_C_idx
        if getC_to_C_idx == "default":
            getC_to_C_idx = np.arange(self._n_components).tolist()
        elif (
            len(getC_to_C_idx)
            > self._n_components
            #   or max(getC_to_C_idx) + 1 > self._n_components
        ):
            raise ValueError(
                f"The profile has only {self._n_components} species, please check "
                f"the `getC_to_C_idx`  configuration (value: {getC_to_C_idx})"
            )
        return getC_to_C_idx

    @tr.validate("nonnegSpec")
    def _validate_nonnegSpec(self, proposal):
        if self._X_is_missing:
            return proposal.value
        nonnegSpec = proposal.value
        if not self._n_components:  # not initialized or 0
            return nonnegSpec
        if nonnegSpec == "all":
            nonnegSpec = np.arange(self._n_components).tolist()
        elif np.any(nonnegSpec) and (
            len(nonnegSpec) > self._n_components
            or max(nonnegSpec) + 1 > self._n_components
        ):
            raise ValueError(
                f"The profile has only {self._n_components} species, please check "
                f"the `nonnegSpec`configuration (value: {nonnegSpec})"
            )
        return nonnegSpec

    @tr.validate("unimodSpec")
    def _validate_unimodSpec(self, proposal):
        if self._X_is_missing:
            return proposal.value
        unimodSpec = proposal.value
        if not self._n_components:  # not initialized or 0
            return unimodSpec
        if unimodSpec == "all":
            unimodSpec = np.arange(self._n_components).tolist()
        elif np.any(unimodSpec) and (
            len(unimodSpec) > self._n_components
            or max(unimodSpec) + 1 > self._n_components
        ):
            raise ValueError(
                f"The profile has only {self._n_components} species, please check the "
                f"`unimodSpec`configuration"
            )
        return unimodSpec

    @tr.validate("getSt_to_St_idx")
    def _validate_getSt_to_St_idx(self, proposal):
        if self._X_is_missing:
            return proposal.value
        getSt_to_St_idx = proposal.value
        if not self._n_components:  # not initialized or 0
            return getSt_to_St_idx
        if getSt_to_St_idx == "default":
            getSt_to_St_idx = np.arange(self._n_components).tolist()
        elif (
            len(getSt_to_St_idx) > self._n_components
            or max(getSt_to_St_idx) + 1 > self._n_components
        ):
            raise ValueError(
                f"The profile has only {self._n_components} species, please check "
                f"the `getSt_to_St_idx`  configuration (value: {getSt_to_St_idx})"
            )
        return getSt_to_St_idx

    @tr.observe("_n_components")
    def _n_components_change(self, change):
        # tiggered in _guess_profile
        if self._n_components > 0:
            # perform a validation of default configuration parameters
            # Indeed, if not forced here these parameters are validated only when they
            # are set explicitely.
            # Here is an ugly trick to force this validation. # TODO: better way?
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                self.closureTarget = self.closureTarget
                self.getC_to_C_idx = self.getC_to_C_idx
                self.getSt_to_St_idx = self.getSt_to_St_idx
                self.nonnegConc = self.nonnegConc
                self.nonnegSpec = self.nonnegSpec
                self.unimodConc = self.unimodConc
                self.unimodSpec = self.unimodSpec
                self.closureConc = self.closureConc

    @tr.default("_components")
    def _components_default(self):
        if self._fitted:
            # note: _outfit = (C, St, C_constrained, St_ls, extraOutputGetConc, extraOutputGetSpec, ...)
            return self._outfit[1]
        else:
            raise NotFittedError("The model was not yet fitted. Execute `fit` first!")

    # ----------------------------------------------------------------------------------
    # Private methods (overloading abstract classes)
    # ----------------------------------------------------------------------------------
    # To see all accessible members it is interesting to use the structure tab of
    # PyCharm
    @tr.observe("_Y")
    def _preprocess_as_Y_changed(self, change):
        # should be a tuple of profiles or only concentrations/spectra profiles
        profiles = change.new
        if isinstance(profiles, (list, tuple)):
            # we assume that the starting C and St are already computed
            # (for ex. from a previous run of fit)
            C, St = [item.data for item in profiles]
            self._n_components = C.shape[1]
            # eventually remove mask
            if np.any(self._X_mask):
                masked_rows, masked_columns = self._get_masked_rc(self._X_mask)
                St = St[:, ~masked_columns]
                C = C[~masked_rows]
        else:
            # not passed explicitly, try to guess.
            C, St = self._guess_profile(profiles.data)

        # we do a last validation
        shape = self._X.shape
        if shape[0] != C.shape[0]:
            # An error will be raised before if X is None.
            raise ValueError("The dimensions of C do not match those of X.")
        if shape[1] != St.shape[1]:
            # An error will be raised before if X is None.
            raise ValueError("The dimensions of St do not match those of X.")
        # return the list of C and St data
        # (with mask removed to fit the size of the _X data)
        self._Y_preprocessed = (C, St)

    def _fit(self, X, Y):
        # this method is called by the abstract class fit.
        # Input X is a np.ndarray
        # Y is a tuple of guessed profiles (each of them being np.ndarray)
        # So every computation below implies only numpy arrays, not NDDataset
        # as in previous versions

        C, St = Y
        ny, _ = X.shape
        n_components = self._n_components
        change = self.tol + 1
        stdev = X.std()
        niter = 0
        ndiv = 0

        C_constrained_list = []
        C_ls_list = []
        St_constrained_list = []
        St_ls_list = []

        info_("***           ALS optimisation log            ***")
        info_("#iter     RSE / PCA        RSE / Exp      %change")
        info_("-------------------------------------------------")

        # get sklearn PCA with same number of components for further comparison
        pca = decomposition.PCA(n_components=n_components)
        Xtransf = pca.fit_transform(X)
        Xpca = pca.inverse_transform(Xtransf)

        while change >= self.tol and niter < self.max_iter and ndiv < self.maxdiv:
            niter += 1

            # Force non-negative concentration
            # ------------------------------------------
            if np.any(self.nonnegConc):
                C[:, self.nonnegConc] = C[:, self.nonnegConc].clip(min=0)

            # Force unimodal concentration
            # ------------------------------------------
            if np.any(self.unimodConc):
                C = _unimodal_2D(
                    C,
                    idxes=self.unimodConc,
                    axis=0,
                    tol=self.unimodConcTol,
                    mod=self.unimodConcMod,
                )

            # Force monotonic increase
            # ------------------------------------------
            if np.any(self.monoIncConc):
                for s in self.monoIncConc:
                    for curid in np.arange(ny - 1):
                        if C[curid + 1, s] < C[curid, s] / self.monoIncTol:
                            C[curid + 1, s] = C[curid, s]

            # Force monotonic decrease
            # ------------------------------------------
            if np.any(self.monoDecConc):
                for s in self.monoDecConc:
                    for curid in np.arange(ny - 1):
                        if C[curid + 1, s] > C[curid, s] * self.monoDecTol:
                            C[curid + 1, s] = C[curid, s]

            # Closure
            # ------------------------------------------
            if self.closureConc:
                if self.closureMethod == "scaling":
                    Q = _lstsq(C[:, self.closureConc], self.closureTarget.T)
                    C[:, self.closureConc] = np.dot(C[:, self.closureConc], np.diag(Q))
                elif self.closureMethod == "constantSum":
                    totalConc = np.sum(C[:, self.closureConc], axis=1)
                    C[:, self.closureConc] = (
                        C[:, self.closureConc]
                        * self.closureTarget[:, None]
                        / totalConc[:, None]
                    )

            # external concentration profiles
            # ------------------------------------------
            extraOutputGetConc = []
            if np.any(self.hardConc):
                _C = self._C_2_NDDataset(C)
                if self.kwargsGetConc != {} and self.argsGetConc != ():
                    output = self.getConc(_C, *self.argsGetConc, **self.kwargsGetConc)
                elif self.kwargsGetConc == {} and self.argsGetConc != ():
                    output = self.getConc(_C, *self.argsGetConc)
                elif self.kwargsGetConc != {} and self.argsGetConc == ():
                    output = self.getConc(_C, **self.kwargsGetConc)
                else:
                    output = self.getConc(_C)

                if isinstance(output, tuple):
                    fixedC = output[0]
                    self.argsGetConc = output[1]
                    if len(output) == 3:
                        extraOutputGetConc.append(output[2])
                else:
                    fixedC = output

                C[:, self.hardConc] = fixedC[:, self.getC_to_C_idx]

            # stores C
            # ---------------------------
            C_constrained = C.copy()
            if self.storeIterations:
                C_constrained_list.append(C_constrained)

            # Compute St
            # -----------
            St = self._solve_St(C)
            St_ls = St.copy()

            # stores St in St_ls_list
            # -----------------------------
            if self.storeIterations:
                St_ls_list.append(St.copy())

            # Force non-negative spectra
            # ------------------------------------------
            if np.any(self.nonnegSpec):
                St[self.nonnegSpec, :] = St[self.nonnegSpec, :].clip(min=0)

            # Force unimodal spectra
            # ------------------------------------------
            if np.any(self.unimodSpec):
                St = _unimodal_2D(
                    St,
                    idxes=self.unimodSpec,
                    axis=1,
                    tol=self.unimodSpecTol,
                    mod=self.unimodSpecMod,
                )

            # External spectral profile
            # ------------------------------------------
            extraOutputGetSpec = []
            if np.any(self.hardSpec):
                _St = self._St_2_NDDataset(St)
                if self.kwargsGetSpec != {} and self.argsGetSpec != ():
                    output = self.getSpec(_St, *self.argsGetSpec, **self.kwargsGetSpec)
                elif self.kwargsGetSpec == {} and self.argsGetSpec != ():
                    output = self.getSpec(_St, *self.argsGetSpecc)
                elif self.kwargsGetSpec != {} and self.argsGetSpec == ():
                    output = self.getSpec(_St, **self.kwargsGetSpec)
                else:
                    output = self.getSpec(_St)

                if isinstance(output, tuple):
                    fixedSt = output[0].data
                    self.argsGetSpec = output[1]
                    if len(output) == 3:
                        extraOutputGetSpec.append(output[2])
                    else:
                        fixedSt = output.data
                else:
                    fixedSt = output.data

                St[self.hardSpec, :] = fixedSt[self.getSt_to_St_idx, :]

            # recompute C
            # -----------
            C = self._solve_C(St)

            # rescale spectra and concentrations
            # ------------------------------------------
            if self.normSpec == "max":
                alpha = np.max(St, axis=1).reshape(self._n_components, 1)
                St = St / alpha
                C = C * alpha.T
            elif self.normSpec == "euclid":
                alpha = np.linalg.norm(St, axis=1).reshape(self._n_components, 1)
                St = St / alpha
                C = C * alpha.T

            # store profiles
            # --------------

            if self.storeIterations:
                C_ls_list.append(C)
                St_constrained_list.append(St)

            # compute residuals
            # ------------------------------------------
            Xhat = C @ St
            stdev2 = np.std(Xhat - X)
            change = 100 * (stdev2 - stdev) / stdev
            stdev = stdev2

            stdev_PCA = np.std(Xhat - Xpca)
            info_(
                f"{niter: 3d}{' ' * 6}{stdev_PCA: 10f}{' ' * 6}"
                f"{stdev2: 10f}{' ' * 6}{change: 10f}"
            )

            # check convergence
            # -----------------

            if change > 0:
                ndiv += 1
            else:
                ndiv = 0
                change = -change

            if change < self.tol:
                info_("converged !")

            if ndiv == self.maxdiv:
                info_(
                    f"Optimization not improved after {self.maxdiv} iterations"
                    f"... unconverged or 'tol' set too small ?"
                )
                info_("Stop ALS optimization.")

            if niter == self.max_iter:
                info_(
                    f"Convergence criterion ('tol') not reached after "
                    f"{self.max_iter: d} iterations."
                )
                info_("Stop ALS optimization.")

        # return _fit results
        self._components = St
        _outfit = (
            C,
            St,
            C_constrained,
            St_ls,
            extraOutputGetConc,
            extraOutputGetSpec,
            C_constrained_list,
            C_ls_list,
            St_constrained_list,
            St_ls_list,
        )
        return _outfit

    def _transform(self, X=None):
        # X is ignored for MCRALS
        return self._outfit[0]

    def _inverse_transform(self, X_transform=None):
        # X_transform is ignored for MCRALS
        return np.dot(self._transform(), self._components)

    def _get_components(self):
        return self._components

    # ----------------------------------------------------------------------------------
    # Public methods and properties
    # ----------------------------------------------------------------------------------
    @_docstring.dedent
    def fit(self, X, Y):
        """
        Fit the MCRALS model on an X dataset using initial concentration or spectra.

        Parameters
        ----------
        %(analysis_fit.parameters.X)s
        Y : :term:`array-like` or list of :term:`array-like`
            Initial concentration or spectra.

        Returns
        -------
        %(analysis_fit.returns)s

        See Also
        --------
        %(analysis_fit.see_also)s
        """
        return super().fit(X, Y)

    @_docstring.dedent
    def fit_transform(self, X, Y, **kwargs):
        """
        Fit the model with ``X`` and apply the dimensionality reduction on ``X``.

        Parameters
        ----------
        %(analysis_fit.parameters.X)s
        Y : :term:`array-like` or list of :term:`array-like`
            Initial concentration or spectra.
        %(kwargs)s

        Returns
        -------
        %(analysis_transform.returns)s

        Other Parameters
        ----------------
        %(analysis_transform.other_parameters)s
        """
        return super().fit_transform(X, Y, **kwargs)

    @_docstring.dedent
    def inverse_transform(self, X_transform=None, **kwargs):
        """
        Transform data back to its original space.

        In other words, return an input `X_original` whose reduce/transform would be X.

        Parameters
        ----------
        %(analysis_inverse_transform.parameters)s

        Returns
        -------
        `NDDataset`
            Dataset with shape (:term:`n_observations`\ , :term:`n_features`\ ).

        Other Parameters
        ----------------
        %(analysis_transform.other_parameters)s
        """
        return super().inverse_transform(X_transform, **kwargs)

    @property
    def C(self):
        """
        The final concentration profiles.
        """
        C = self.transform()
        C.name = "Pure concentration profile, mcs-als of " + self.X.name
        return C

    @property
    def St(self):
        """
        The final spectra profiles.
        """
        St = self.components
        St.name = "Pure spectra profile, mcs-als of " + self.X.name
        return St

    @property
    @_wrap_ndarray_output_to_nddataset(units=None, title=None, typex="components")
    def C_constrained(self):
        """
        The last constrained concentration profiles, i.e. after applying the hard and soft constraints.
        """
        return self._outfit[2]

    @property
    @_wrap_ndarray_output_to_nddataset(units=None, title=None, typey="components")
    def St_ls(self):
        r"""
        The last spectral profiles obtained by least-square optimization, before constraints.

        Spectra obtained after solving :math:`C_{\textrm{constrained}} \cdot St = X`
        for :math:`St`\ .
        """
        return self._outfit[3]

    @property
    @deprecated(replace="St_ls")
    def St_unconstrained(self):
        """
        Deprecated. Equivalent to `St_ls`.
        """
        return self.St_ls

    @property
    @deprecated(replace="St_ls")
    def S_soft(self):
        """
        Deprecated. Equivalent to `St_ls`.
        """
        return self.St_ls

    @property
    def extraOutputGetConc(self):
        """
        The extra outputs of the external function used to get concentrations.
        """
        return self._outfit[4]

    @property
    def extraOutputGetSpec(self):
        """
        The extra outputs of the external function used to get spectra.
        """
        return self._outfit[5]

    @property
    def C_constrained_list(self):
        """
        The list of constrained concentration profiles at each ALS iteration.

        Requires `MCRALS.storeIterations` set to True.
        """
        return self._outfit[6]

    @property
    def C_ls_list(self):
        """
        The list of concentration profiles obtained by least square optimization and scaling at each ALS iteration.

        Requires `MCRALS.storeIterations` set to True.
        """
        return self._outfit[7]

    @property
    def St_constrained_list(self):
        """
        The list of constrained spectral profiles at each ALS iteration.

        Requires `MCRALS.storeIterations` set to True.
        """
        return self._outfit[9]

    @property
    def St_ls_list(self):
        """
        The list of optimized spectral profiles at each ALS iteration.

        Requires `MCRALS.storeIterations` set to True.
        """
        return self._outfit[10]


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


# LS solvers for W in the linear matrix equation X @ W = Y
def _lstsq(X, Y, rcond=None):
    # Least-squares solution to a linear matrix equation X @ W = Y
    # Return W
    W = np.linalg.lstsq(X, Y, rcond)[0]
    return W


def _nnls(X, Y, withres=False):
    # Non negative least-squares solution to a linear matrix equation X @ W = Y
    # Return W >= 0
    # TODO: look at may be faster algorithm: see: https://gist.github.com/vene/7224672
    nsamp, nfeat = X.shape
    nsamp, ntarg = Y.shape
    W = np.empty((nfeat, ntarg))
    residuals = 0
    for i in range(ntarg):
        Y_ = Y[:, i]
        W[:, i], res = scipy.optimize.nnls(X, Y_)
        residuals += res**2
    return (W, np.sqrt(residuals)) if withres else W


def _pnnls(X, Y, nonneg=[], withres=False):
    # Least-squares  solution to a linear matrix equation X @ W = Y
    # with partial nonnegativity (indicated by the nonneg list of targets)
    # Return W with eventually some column non negative.
    nsamp, nfeat = X.shape
    nsamp, ntarg = Y.shape
    W = np.empty((nfeat, ntarg))
    residuals = 0
    for i in range(ntarg):
        Y_ = Y[:, i]
        if i in nonneg:
            W[:, i], res = scipy.optimize.nnls(X, Y_)
        else:
            W[:, i], res = np.linalg.lstsq(X, Y_)[:2]
        residuals += res**2
    return (W, np.sqrt(residuals)) if withres else W


def _unimodal_2D(a, axis, idxes, tol, mod):
    # Force unimodality on given lines or columnns od a 2D ndarray
    #
    # a: ndarray
    #
    # axis: int
    #     axis along which the correction is applied
    #
    # idxes: list of int
    #     indexes at which the correction is applied
    #
    # mod : str
    #     When set to `"strict"`\ , values deviating from unimodality are reset to the
    #     value of the previous point. When set to `"smooth"`\ , both values (deviating
    #     point and previous point) are modified to avoid "steps" in the profile.
    #
    # tol: float
    #     Tolerance parameter for unimodality. Correction is applied only if:
    #     `a[i] > a[i-1] * unimodTol`  on a decreasing branch of profile,
    #     `a[i] < a[i-1] * unimodTol`  on an increasing branch of profile.

    if axis == 0:
        a_ = a
    elif axis == 1:
        a_ = a.T

    for col, idx in zip(a_[:, idxes].T, idxes):
        a_[:, idx] = _unimodal_1D(col, tol, mod)

    return a


def _unimodal_1D(a: np.ndarray, tol: str, mod: str) -> np.ndarray:
    # force unimodal concentration
    #
    # makes a vector unimodal
    #
    # Parameters
    # ----------
    # a : 1D ndarray
    #
    # mod : str
    #     When set to `"strict"`\ , values deviating from unimodality are reset to the value
    #     of the previous point. When set to `"smooth"`\ , both values (deviating point and
    #     previous point) are modified to avoid "steps"
    #     in the profile.
    #
    # tol: float
    #     Tolerance parameter for unimodality. Correction is applied only if:
    #     `a[i] > a[i-1] * unimodTol`  on a decreasing branch of profile,
    #     `a[i] < a[i-1] * unimodTol`  on an increasing branch of profile.

    maxid = np.argmax(a)
    curmax = max(a)
    curid = maxid

    while curid > 0:
        # run backward
        curid -= 1
        if a[curid] > curmax * tol:
            if mod == "strict":
                a[curid] = a[curid + 1]
            if mod == "smooth":
                a[curid] = (a[curid] + a[curid + 1]) / 2
                a[curid + 1] = a[curid]
                curid = curid + 2
        curmax = a[curid]

    curid = maxid
    curmax = a[maxid]
    while curid < len(a) - 1:
        curid += 1
        if a[curid] > curmax * tol:
            if mod == "strict":
                a[curid] = a[curid - 1]
            if mod == "smooth":
                a[curid] = (a[curid] + a[curid - 1]) / 2
                a[curid - 1] = a[curid]
                curid = curid - 2
        curmax = a[curid]
    return a
