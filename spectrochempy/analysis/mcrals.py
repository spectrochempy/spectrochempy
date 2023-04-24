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
import traitlets as tr
from sklearn import decomposition

from spectrochempy.analysis._base import (
    DecompositionAnalysis,
    NotFittedError,
    _wrap_ndarray_output_to_nddataset,
)
from spectrochempy.core import info_
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
    (or a set of mixtures) into the spectra :math:`S^T` of "pure" species and their
    concentration profiles :math:`C` .

    In terms of matrix equation:

    .. math:: X = C.S^T + E

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

    nonnegConc = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value="all",
        help=(
            "Non-negativity constraint on concentrations. If set to ``'all'`` (default) "
            "all concentrations profiles are considered non-negative."
            " If an array of indexes is passed, the corresponding profiles are "
            "considered non-negative, not the others. "
            "For instance ``[0, 2]`` indicates that profile \#0 and \#2 are "
            "non-negative while profile \#1 *can* be negative. If set to ``[]`` , "
            "all profiles can be negative."
        ),
    ).tag(config=True)

    unimodConc = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value="all",
        help=(
            "Unimodality constraint on concentrations. If set to ``'all'`` "
            "(default) all concentrations profiles are considered unimodal. "
            "If an array of indexes is passed, the corresponding profiles are "
            "considered unimodal, not the others."
            "For instance ``[0, 2]`` indicates that profile ``#0`` and ``#2`` are "
            "unimodal while profile ``#1`` *can* be multimodal. If set to ``[]``\ , "
            "all profiles can be multimodal."
        ),
    ).tag(config=True)

    unimodConcMod = tr.Enum(
        ["strict", "smooth"],
        default_value="strict",
        help="""When set to ``'strict'``\ , values deviating from :term:`unimodality` are
reset to the value of the previous point. When set to ``'smooth'``\ , both values
(deviating point and previous point) are modified to avoid steps in the concentration
profile.""",
    ).tag(config=True)

    unimodConcTol = tr.Float(
        default_value=1.1,
        help="""Tolerance parameter for :term:`unimodality`\ . Correction is applied only
if:

* ``C[i,j] > C[i-1,j] * unimodTol`` on the decreasing branch of profile ``#j``\ ,
* ``C[i,j] < C[i-1,j] * unimodTol`` on the increasing branch of profile ``#j``\ .""",
    ).tag(config=True)

    monoDecConc = tr.List(
        default_value=[],
        help="""Monotonic decrease constraint on concentrations.  If set to ``[]``
(default) no constraint is applied. If an array of indexes is passed,
the corresponding profiles are considered do decrease monotonically, not the
others. For instance ``[0, 2]`` indicates that profile ``#0`` and ``#2`` are decreasing
while profile ``#1`` *can* increase.""",
    ).tag(config=True)

    monoDecTol = tr.Float(
        default_value=1.1,
        help="""Tolerance parameter for monotonic decrease. Correction is applied only
if: ``C[i,j] > C[i-1,j] * unimodTol``  along profile ``#j``\ .""",
    ).tag(config=True)

    monoIncConc = tr.List(
        default_value=[],
        help="""Monotonic increase constraint on concentrations.  If set to ``[]``
(default) no constraint is applied. If an array of indexes is passed,
the corresponding profiles are considered to increase monotonically, not the
others. For instance ``[0, 2]`` indicates that profile ``#0`` and ``#2`` are increasing
while profile ``#1`` *can* decrease.""",
    ).tag(config=True)

    monoIncTol = tr.Float(
        default_value=1.1,
        help="""Tolerance parameter for monotonic decrease. Correction is applied only
if: ``C[i,j] < C[i-1,j] * unimodTol`` along profile ``#j``\ .""",
    ).tag(config=True)

    closureConc = tr.List(
        default_value=[],
        help="""Defines the concentration profiles subjected to closure constraint.
If set to ``[]``\ , no constraint is applied. If an array of indexes is
passed, the corresponding profile will be constrained so that their
weighted sum equals the `closureTarget`\ .""",
    ).tag(config=True)

    closureTarget = tr.Union(
        (tr.Enum(["default"]), Array()),
        default_value="default",
        help="""The value of the sum of concentrations profiles subjected to closure.
If set to ``'default'``\ , the total concentration is set to ``1.0`` for all observations.
If an array is passed: the values of concentration for each observation. Hence,
``np.ones(X.shape[0])`` would be equivalent to ``'default'``\ .""",
    ).tag(config=True)

    closureMethod = tr.Enum(
        ["scaling", "constantSum"],
        default_value="scaling",
        help="""The method used to enforce :term:`closure` .

* ``'scaling'`` recompute the concentration profiles using linear algebra:

   .. code-block:: python

      C[:, closureConc] = np.dot(
                            C.[:, closureConc],
                            np.diag(
                              np.linalg.lstsq(
                                C[:, closureConc], closureTarget.T
                                )[0]
                            )
                          )

* ``'constantSum'`` normalize the sum of concentration profiles to `closureTarget`\ .""",
    ).tag(config=True)

    hardConc = tr.List(
        default_value=[],
        help="""Defines hard constraints on the concentration profiles. If set to
``[]``\ , no constraint is applied. If an array of indexes is passed, the
corresponding profiles will set by `getC`\ .""",
    ).tag(config=True)

    getConc = tr.Union(
        (tr.Callable(), tr.Unicode()),
        default_value=None,
        allow_none=True,
        help="""An external function that will provide ``len(hardConc)`` concentration
profiles.

It should be using one of the following syntax:

* ``getConc(Ccurr, *argsGetConc, **kwargsGetConc) -> hardC``
* ``getConc(Ccurr, *argsGetConc, **kwargsGetConc) -> hardC, newArgsGetConc``
* ``getConc(Ccurr, *argsGetConc, **kwargsGetConc) -> hardC, newArgsGetConc, extraOutputGetConc``

where ``Ccurr`` is the current `C` dataset, ``\*argsGetConc`` are the parameters needed
to completely specify the function. `hardC` is a `~numpy.ndarray` or `NDDataset` of shape
``(C.y, len(hardConc)``\ , ``newArgsGetConc`` are the updated parameters for the next
iteration (can be None), and ``extraOutputGetConc`` can be any other relevant output to
be kept in `extraOutputGetConc` attribute, a list of ``extraOutputGetConc`` at each
MCR ALS iterations.

.. note::
    it can be also a serialized function created using dill and base64 python libraries.
    Normally not used directly, it is here for internal process.""",
    ).tag(config=True)

    argsGetConc = tr.Tuple(
        default_value=(),
        help="Supplementary positional arguments passed to the external function.",
    ).tag(config=True)

    kwargsGetConc = tr.Dict(
        default_value={},
        help="Supplementary keyword arguments passed to the external function.",
    ).tag(config=True)

    hardC_to_C_idx = tr.Union(
        (tr.Enum(["default"]), tr.List()),
        default_value="default",
        help="""Indicates the correspondence between the indexes of the columns of
`hardC` and of the `C` matrix. ``[1, None, 0]`` indicates that the first profile in
`hardC` (index ``O``\ ) corresponds to the second profile of `C` (index ``1``\ ).""",
    ).tag(config=True)

    nonnegSpec = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value="all",
        help="""Indicates non-negative spectral profile. If set to ``'all'`` (default)
all spectral profiles are considered non-negative. If an array of indexes is
passed, the corresponding profiles are considered non-negative, not the others.
For instance ``[0, 2]`` indicates that profile ``#0`` and ``#2`` are non-negative while
profile ``#1`` *can* be negative. If set to ``None`` or ``[]``\ , all profiles can be
negative.""",
    ).tag(config=True)

    normSpec = tr.Enum(
        [None, "euclid", "max"],
        default_value=None,
        help="""Defines whether the spectral profiles should be normalized. If set to
``None`` no normalization is applied.
when set to ``"euclid"``\ , spectra are normalized with respect to their total area,
when set to ``"max"``\ , spectra are normalized with respect to the maximum af their
value.""",
    ).tag(config=True)

    unimodSpec = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value=[],
        help="""Unimodality constraint on Spectra. If the list of spectral profiles is
void, all profiles can be multimodal. If set to ``'all'``\ , all profiles are unimodal.
If an array of indexes is passed, the corresponding profiles are considered unimodal,
not the others. For instance ``[0, 2]`` indicates that profile ``#0`` and ``#2`` are
unimodal while profile ``#1`` *can* be multimodal.""",
    ).tag(config=True)

    unimodSpecMod = tr.Enum(
        ["strict", "smooth"],
        default_value="strict",
        help=""" When set to ``"strict"``\ , values deviating from unimodality are reset
to the value of the previous point. When set to ``"smooth"``\ , both values (deviating
point and previous point) are modified to avoid steps
in the concentration profile.""",
    ).tag(config=True)

    unimodSpecTol = tr.Float(
        default_value=1.1,
        help="""Tolerance parameter for unimodality. Correction is applied only if the
deviating point is larger/lower than ``St[j,i] > St[j, i-1] * unimodSpecTol``
on the decreasing branch of profile ``#j``\ , ``St[j,i] < St[j, i-1] * unimodTol`` on
the increasing branch of profile  ``#j``\ .""",
    ).tag(config=True)

    hardSpec = tr.List(
        default_value=[],
        help="""Defines hard constraints on the spectral profiles. If set to ``[]`` ,
no constraint is applied. If an array of indexes is passed, the corresponding profiles
will set by `getSt`\ .""",
    ).tag(config=True)

    getSpec = tr.Union(
        (tr.Callable(), tr.Unicode()),
        default_value=None,
        allow_none=True,
        help="""An external function that will provide ``len(hardSpec)`` concentration
profiles.

It should be using one of the following syntax:

* ``getSpec(Stcurr, *argsGetSpec, **kwargsGetSpec) -> hardSt``
* ``getSpec(Stcurr, *argsGetSpec, **kwargsGetSpec) -> hardSt, newArgsGetSpec``
* ``getSpec(Stcurr, *argsGetSpec, **kwargsGetSpec) -> hardSt, newArgsGetSpec, extraOutputGetSpec``

where ``Stcurr`` is the current `St` dataset, ``\*argsGetSpec`` are the parameters
needed to completely specify the function. `hardSt` is a `~numpy.ndarray` or `NDDataset` of
shape ``(C.y, len(hardSpec)``\ , ``newArgsGetSpec`` are the updated parameters for the
next iteration (can be None), and ``extraOutputGetSpec`` can be any other relevant
output to be kept in `extraOutputGetSpec` attribute, a list of ``extraOutputGetSpec``
at each iterations.

.. note::
    it can be also a serialized function created using dill and base64 python libraries.
    Normally not used directly, it is here for internal process.""",
    ).tag(config=True)

    argsGetSpec = tr.Tuple(
        default_value=(),
        help="Supplementary positional arguments passed to the external function.",
    ).tag(config=True)

    kwargsGetSpec = tr.Dict(
        default_value={},
        help="Supplementary keyword arguments passed to the external function.",
    ).tag(config=True)

    hardSt_to_St_idx = tr.Union(
        (tr.Enum(["default"]), tr.List()),
        default_value="default",
        help="""Indicates the correspondence between the indexes of the lines of
`hardSt` and of the `St` matrix. ``[1, None, 0]`` indicates that the first profile in
`hardSt` (index ``O`` ) corresponds to the second profile of `St` (index ``1``\ ).""",
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
                "Passing arguments such as MCRALS(X , profile) "
                "is now deprecated. "
                "Instead, use MCRAL() followed by MCRALS.fit(X , profile). "
                "See the documentation and exemples"
            )

        # warn about deprecation
        # ----------------------
        # We use pop to remove the deprecated argument before processing the rest
        # TODO: These arguments should be removed in version 0.6 probably

        # verbose
        if "verbose" in kwargs:
            deprecated("verbose", replace="log_level='INFO'", removed="0.6.5")
            verbose = kwargs.pop("verbose")
            if verbose:
                log_level = "INFO"

        # unimodTol deprecation
        if "unimodTol" in kwargs:
            deprecated("unimodTol", replace="unimodConcTol", removed="0.6.5")
            kwargs["unimodConcTol"] = kwargs.pop("unimodTol")

        # unimodMod deprecation
        if "unimodMod" in kwargs:
            deprecated("unimodMod", replace="unimodConcMod", removed="0.6.5")
            kwargs["unimodConcMod"] = kwargs.pop("unimodMod")

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
    def _guess_profile(self, profile):
        """
        Set or guess an initial profile.

        Parameters
        ----------
        profile : np.ndarray
            Initial guess for the concentration or spectra profile.
        """
        if self._X_is_missing:
            return

        # check the dimensions compatibility
        # however as the dimension of profile should match the initial shape
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

        # data array
        Xdata = self._X.data

        # mask info
        if np.any(self._X_mask):
            masked_rows, masked_columns = self._get_masked_rc(self._X_mask)

        # make the profile
        try:  # first try on concentration
            # The data are validated in _C_validate()
            # if it fails here due to shape mismatch, it goes to the except

            Cdata = profile.copy()
            n_components = Cdata.shape[1]
            info_(f"Concentration profile initialized with {n_components} components")
            # compute initial spectra (using the Xdata eventually masked
            Stdata = np.linalg.lstsq(Cdata, Xdata, rcond=None)[0]
            info_("Spectra profile computed")
            # if everything went well here, C and St are set, we return
            # after having removed the eventual C mask!
            if np.any(self._X_mask):
                Cdata = Cdata[~masked_rows]
            # update the number of components
            self._n_components = n_components

            return Cdata, Stdata

        except np.linalg.LinAlgError as exc:
            if "Incompatible dimensions" not in exc.args[0]:
                raise exc
            pass

        # Again if something is wrong we let it raise the error
        # as there is no other possibility (but this should not occur as we did
        # already the test on the dimension's compatibility.
        Stdata = profile.copy()
        n_components = Stdata.shape[0]
        info_(f"Spectra profile initialized with {n_components} components")
        # compute initial concentration
        Ctdata = np.linalg.lstsq(Stdata.T, Xdata.T, rcond=None)[0]
        Cdata = Ctdata.T
        info_("Concentration profile computed")
        # if everything went well here, C and St are set, we return
        # after having removed the eventual St mask!
        if np.any(self._X_mask):
            Stdata = Stdata[:, ~masked_columns]
        # update the number of components
        self._n_components = n_components

        return Cdata, Stdata

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
                f"the `nonnegConc` configuration (value:{nonnegConc})"
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
                f"`unimodConc` configuration (value:{unimodConc})"
            )
        return unimodConc

    @tr.validate("closureTarget")
    def _validate_closureTarget(self, proposal):
        if self._X_is_missing:
            return proposal.value
        closureTarget = proposal.value
        ny = self.X.shape[0]
        if closureTarget == "default":
            closureTarget = np.ones(ny)
        elif len(closureTarget) != ny:
            raise ValueError(
                f"The data contain only {ny} observations, please check "
                f"the 'closureTarget' configuration (value:{closureTarget})"
            )
        return closureTarget

    @tr.validate("hardC_to_C_idx")
    def _validate_hardC_to_C_idx(self, proposal):
        if self._X_is_missing:
            return proposal.value
        hardC_to_C_idx = proposal.value
        if not self._n_components:  # not initialized or 0
            return hardC_to_C_idx
        if hardC_to_C_idx == "default":
            hardC_to_C_idx = np.arange(self._n_components).tolist()
        elif (
            len(hardC_to_C_idx) > self._n_components
            or max(hardC_to_C_idx) + 1 > self._n_components
        ):
            raise ValueError(
                f"The profile has only {self._n_components} species, please check "
                f"the `hardC_to_C_idx`  configuration (value:{hardC_to_C_idx})"
            )
        return hardC_to_C_idx

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
                f"the `nonnegSpec`configuration (value:{nonnegSpec})"
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

    @tr.validate("hardSt_to_St_idx")
    def _validate_hardSt_to_St_idx(self, proposal):
        if self._X_is_missing:
            return proposal.value
        hardSt_to_St_idx = proposal.value
        if not self._n_components:  # not initialized or 0
            return hardSt_to_St_idx
        if hardSt_to_St_idx == "default":
            hardSt_to_St_idx = np.arange(self._n_components).tolist()
        elif (
            len(hardSt_to_St_idx) > self._n_components
            or max(hardSt_to_St_idx) + 1 > self._n_components
        ):
            raise ValueError(
                f"The profile has only {self._n_components} species, please check "
                f"the `hardSt_to_St_idx`  configuration (value:{hardSt_to_St_idx})"
            )
        return hardSt_to_St_idx

    @tr.observe("_Y_preprocessed")
    def _Y_preprocessed_change(self, change):
        if self._n_components > 0:
            # perform a validation of default configuration parameters
            # Indeed, if not forced here these parameters are validated only when they
            # are set explicitely.
            # Here is an ugly trick to force this validation. # TODO: better way?
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                self.closureTarget = self.closureTarget
                self.hardC_to_C_idx = self.hardC_to_C_idx
                self.hardSt_to_St_idx = self.hardSt_to_St_idx
                self.nonnegConc = self.nonnegConc
                self.nonnegSpec = self.nonnegSpec
                self.unimodConc = self.unimodConc
                self.unimodSpec = self.unimodSpec

    @tr.default("_components")
    def _components_default(self):
        if self._fitted:
            # note: _outfit = (C, St, C_hard, St_soft, extraOutputGetConc, extraOutputGetSpec)
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
            Cdata, Stdata = [item.data for item in profiles]
            self._n_components = Cdata.shape[1]
            # eventually remove mask
            if np.any(self._X_mask):
                masked_rows, masked_columns = self._get_masked_rc(self._X_mask)
                Stdata = Stdata[:, ~masked_columns]
                Cdata = Cdata[~masked_rows]
        else:
            # not passed explicitly, try to guess
            Cdata, Stdata = self._guess_profile(profiles.data)

        # we do a last validation
        shape = self._X.shape
        if shape[0] != Cdata.shape[0]:
            # An error will be raised before if X is None.
            raise ValueError("The dimensions of C do not match those of X.")
        if shape[1] != Stdata.shape[1]:
            # An error will be raised before if X is None.
            raise ValueError("The dimensions of St do not match those of X.")
        # return the list of C and St data
        # (with mask removed to fit the size of the _X data)
        self._Y_preprocessed = (Cdata, Stdata)

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

        info_("***           ALS optimisation log            ***")
        info_("#iter     RSE / PCA        RSE / Exp      %change")
        info_("-------------------------------------------------")

        # get sklearn PCA with same number of components for further comparison
        pca = decomposition.PCA(n_components=n_components)
        Xtransf = pca.fit_transform(X)
        Xpca = pca.inverse_transform(Xtransf)

        while change >= self.tol and niter < self.max_iter and ndiv < self.maxdiv:
            C = np.linalg.lstsq(St.T, X.T, rcond=None)[0].T
            niter += 1

            # Force non-negative concentration
            # --------------------------------
            if np.any(self.nonnegConc):
                for s in self.nonnegConc:
                    C[:, s] = C[:, s].clip(min=0)

            # Force unimodal concentration
            # ----------------------------
            if np.any(self.unimodConc):
                C = _unimodal_2D(
                    C,
                    idxes=self.unimodConc,
                    axis=0,
                    tol=self.unimodConcTol,
                    mod=self.unimodConcMod,
                )

            # Force monotonic increase
            # ------------------------
            if np.any(self.monoIncConc):
                for s in self.monoIncConc:
                    for curid in np.arange(ny - 1):
                        if C[curid + 1, s] < C[curid, s] / self.monoIncTol:
                            C[curid + 1, s] = C[curid, s]

            # Force monotonic decrease
            # ----------------------------------------------
            if np.any(self.monoDecConc):
                for s in self.monoDecConc:
                    for curid in np.arange(ny - 1):
                        if C[curid + 1, s] > C[curid, s] * self.monoDecTol:
                            C[curid + 1, s] = C[curid, s]

            # Closure
            # ------------------------------------------
            if self.closureConc:
                if self.closureMethod == "scaling":
                    Q = np.linalg.lstsq(
                        C[:, self.closureConc],
                        self.closureTarget.T,
                        rcond=None,
                    )[0]
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
                else:
                    fixedC = output

                C[:, self.hardConc] = fixedC[:, self.hardC_to_C_idx]

            # stores C in C_hard
            C_hard = C.copy()

            # compute St
            St = np.linalg.lstsq(C, X, rcond=None)[0]

            # stores St in St_soft
            St_soft = St.copy()

            # Force non-negative spectra
            # --------------------------
            if np.any(self.nonnegSpec):
                St[self.nonnegSpec, :] = St[self.nonnegSpec, :].clip(min=0)

            # Force unimodal spectra
            # ----------------------------
            if np.any(self.unimodSpec):
                St = _unimodal_2D(
                    St,
                    idxes=self.unimodSpec,
                    axis=1,
                    tol=self.unimodSpecTol,
                    mod=self.unimodSpecMod,
                )

            # External spectral profile
            # -----------------------------
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

                print(self.hardSt_to_St_idx)
                St[self.hardSpec, :] = fixedSt[self.hardSt_to_St_idx, :]

            # recompute C for consistency(soft modeling)
            C = np.linalg.lstsq(St.T, X.T, rcond=-1)[0].T

            # rescale spectra & concentrations
            if self.normSpec == "max":
                alpha = np.max(St, axis=1).reshape(self._n_components, 1)
                St = St / alpha
                C = C * alpha.T
            elif self.normSpec == "euclid":
                alpha = np.linalg.norm(St, axis=1).reshape(self._n_components, 1)
                St = St / alpha
                C = C * alpha.T

            # compute residuals
            # -----------------
            Xhat = np.dot(C, St)
            stdev2 = np.std(Xhat - X)
            change = 100 * (stdev2 - stdev) / stdev
            stdev = stdev2

            stdev_PCA = np.std(Xhat - Xpca)  #

            info_(
                f"{niter:3d}{' '*6}{stdev_PCA:10f}{' '*6}"
                f"{stdev2:10f}{' '*6}{change:10f}"
            )

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
                    f"{ self.max_iter:d} iterations."
                )
                info_("Stop ALS optimization.")

        # return _fit results
        self._components = St
        _outfit = (C, St, C_hard, St_soft, extraOutputGetConc, extraOutputGetSpec)
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
    def C_hard(self):
        """
        The hard concentration profiles.
        """
        return self._outfit[2]

    @property
    @_wrap_ndarray_output_to_nddataset(units=None, title=None, typey="components")
    def St_soft(self):
        """
        The soft spectra profiles.
        """
        return self._outfit[3]

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


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _unimodal_2D(a, axis, idxes, tol, mod):
    # """Force unimodality on given lines or columnns od a 2D ndarray
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
    # """

    if axis == 0:
        a_ = a
    elif axis == 1:
        a_ = a.T

    for col, idx in zip(a_[:, idxes].T, idxes):
        a_[:, idx] = _unimodal_1D(col, tol, mod)

    return a


def _unimodal_1D(a: np.ndarray, tol: str, mod: str) -> np.ndarray:
    # """force unimodal concentration
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
    # """

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
