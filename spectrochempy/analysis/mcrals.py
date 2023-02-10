# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implements the MCRALS class.
"""

__all__ = ["MCRALS"]

__dataset_methods__ = []

import logging
import warnings

import numpy as np
import traitlets as tr
from traitlets.config import Configurable

from spectrochempy.analysis.pca import PCA
from spectrochempy.core import app, debug_, info_, set_loglevel
from spectrochempy.core.common.meta import Meta
from spectrochempy.core.dataset.arraymixins.npy import dot
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils.exceptions import deprecated

# Developper notes
# ----------------
# MCRALS could be (as in the previous version) a subclass of traitlets.HasTraits
# which offer the possibility of defining internal traits with full validation
# BUT better we will use a traitlet Configurable class so that we can have all
# parameters set from a configuration object.
# TODO: may be allow load/save of this configuration to let the user permanently change
# some parameters.


# class MCRALS(tr.HasTraits):
class MCRALS(Configurable):
    """
    Performs MCR-ALS of a dataset knowing the initial C or St matrix.

    MCR-ALS (Multivariate Curve Resolution - Alternating Least Squares) resolve"s a set
    (or several sets) of spectra X of an evolving mixture (or a set of mixtures) into
    the spectra St of ‘pure’ species and their concentration profiles C. In terms of
    matrix equation:
    .. math::`X = CS^T + E`
    where :math:`E` is the matrix of residuals.

    Parameters
    ----------
     Parameters
    ----------
    X : NDDataset or an array-like object
        The dataset on which to perform the MCR-ALS analysis.

    guess : NDDataset or an array-like object
        Initial concentration or spectra.

    **kwargs
        Optional parameters, see Other parameters below.

    Other Parameters
    ----------------
    log_level : ["INFO", "DEBUG", "WARNING", "ERROR"], optional, default;"WARNING"
        The log level at startup

    config :

    See Also
    --------
    PCA : Performs MCR-ALS of a |NDDataset| knowing the initial C or St matrix.
    NNMF : Performs a Non-Negative Matrix Factorization of a |NDDataset|.
    EFA : Perform an Evolving Factor Analysis (forward and reverse) of the input
          |NDDataset|.
    """

    # Developer notes
    # ----------------
    # Private traits with internal validation (see Traitlets librairie for more
    # information)
    # Notice that variable not defined this way lack this type validation, so they are
    # more prone to errors.

    # ----------------------------------------------------------------------------------
    # Runtime Parameters
    # ----------------------------------------------------------------------------------

    _X = tr.Instance(NDDataset)
    _C = tr.Instance(NDDataset)
    _Chard = tr.Instance(NDDataset)
    _St = tr.Instance(NDDataset)
    _StSoft = tr.Instance(NDDataset)
    _extOutput = tr.Instance(NDDataset, allow_none=True)
    _nspecies = tr.Integer(allow_none=True)

    # ----------------------------------------------------------------------------------
    # Configuration parameters
    # They will be written in a file from which the default can be modified)
    # Obviously, the parameters can also be modified at runtime as usual by affectation.
    # ----------------------------------------------------------------------------------

    tol = tr.Float(
        0.1,
        help="""Convergence criterion on the change of residuals (percent change of
        standard deviation of residuals).""",
    ).tag(config=True)

    maxit = tr.Integer(50, help="Maximum number of ALS iteration").tag(config=True)

    maxdiv = tr.Integer(
        5, help="Maximum number of successive non-converging iterations."
    ).tag(config=True)

    nonnegConc = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value="all",
        help="""Non negativity constraint on concentrations. If set to `'all'` (default)
         all concentrations profiles are considered non-negative. If an array of indexes
         is passed, the corresponding profiles are considered non-negative, not the
         others. For instance `[0, 2]` indicates that profile #0 and #2 are non-negative
        while profile #1 *can* be negative. If set to `[]`, all profiles can
        be negative.""",
    ).tag(config=True)

    unimodConc = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value="all",
        help="""Unimodality constraint on concentrations. If set to 'all' (default) all
        concentrations profiles are considered unimodal. If an array of indexes is
        passed, the corresponding profiles are considered unimodal, not the others.
        For instance `[0, 2]` indicates that profile #0 and #2 are unimodal while
        profile #1 *can* be multimodal. If set to `[]`, all profiles can be
        multimodal.""",
    ).tag(config=True)

    unimodConcMod = tr.Enum(
        ["strict"],
        default_value="strict",
        help="""When set to 'strict', values deviating from unimodality are reset to the
         value of the previous point. When set to 'smooth', both values (deviating point
          and previous point) are modified to avoid ='steps' in the concentration
          profile.""",
    ).tag(config=True)

    unimodConcTol = tr.Float(
        default_value=1.1,
        help="""Tolerance parameter for unimodality. Correction is applied only if:
         ```C[i,j] > C[i-1,j] * unimodTol```  on the decreasing branch of profile #j,
         ```C[i,j] < C[i-1,j] * unimodTol```  on the increasing branch of profile  #j.
         """,
    ).tag(config=True)

    monoDecConc = tr.List(
        default_value=[],
        help="""Monotonic decrease constraint on concentrations.  If set to `[]`
        (default) no constraint is applied. If an array of indexes is passed,
        the corresponding profiles are considered do decrease monotonically, not the
        others. For instance `[0, 2]` indicates that profile #0 and #2 are decreasing
        while profile #1 *can* increase.""",
    ).tag(config=True)

    unimodDecTol = tr.Float(
        default_value=1.1,
        help="""Tolerance parameter for monotonic decrease. Correction is applied only
        if:```C[i,j] > C[i-1,j] * unimodTol```  along profile #j.""",
    ).tag(config=True)

    monoIncConc = tr.List(
        default_value=[],
        help="""Monotonic increase constraint on concentrations.  If set to `[]`
        (default) no constraint is applied. If an array of indexes is passed,
        the corresponding profiles are considered to increase monotonically, not the
        others. For instance `[0, 2]` indicates that profile #0 and #2 are increasing
        while profile #1 *can* decrease.""",
    ).tag(config=True)

    monoIncTol = tr.Float(
        default_value=1.1,
        help="""Tolerance parameter for monotonic decrease. Correction is applied only
        if: ```C[i,j] < C[i-1,j] * unimodTol``` along profile  #j.""",
    ).tag(config=True)

    closureConc = tr.List(
        default_value=[],
        help="""Defines the concentration profiles subjected to closure constraint.
        If set to `[]`, no constraint is applied. If an array of indexes is
        passed, the corresponding profile will be constrained so that their weighted
        sum equals the `closureTarget`.""",
    ).tag(config=True)

    closureTarget = tr.Union(
        (tr.Enum(["default"]), tr.List()),
        default_value="default",
        help="""The value of the sum of concentrations profiles subjected to closure.
        If set to `default`, the total concentration is set to 1.0 for all observations.
        If an array is passed: the values of concentration for each observation. Hence,
        `np.ones[X.shape[0]` would be equivalent to 'default'.""",
    ).tag(config=True)

    closureMethod = tr.Enum(
        ["scaling", "constantSum"],
        default_value="scaling",
        help="""The method used to enforce Closure. 'scaling' recompute the
        concentration profiles using linear algebra:
        ```
        C.data[:, closureConc] = np.dot(C.data[:, closureConc], np.diag(
        np.linalg.lstsq(C.data[:, closureConc], closureTarget.T, rcond=None)[0]))
        ```
        'constantSum' normalize the sum of concentration profiles to `closureTarget`.""",
    ).tag(config=True)

    hardConc = tr.List(
        default_value=[],
        help="""Defines hard constraints on the concentration profiles. If set to
      `[]`, no constraint is applied. If an array of indexes is passed, the
        corresponding profiles will set by `getC` (see below)..""",
    ).tag(config=True)

    getC = tr.Callable(
        default_value=None,
        allow_none=True,
        help="""An external function that will provide `len(hardConc)` concentration
        profiles:
        ```
        getC(Ccurr, *argsGetC, **kwargsGetC) -> hardC
        ```
        or:
         ```
        getC(Ccurr, *argsGetC, **kwargsGetC) -> hardC, newArgsGetC
        ```
        or:
        ```
        getC(Ccurr, *argsGetCn, **kargsGetC) -> hardC, newArgsGetC, extOutput
        ```
        where Ccurr  is the current C matrix, *argsGetC are the parameters needed to
        completely specify the function. `hardC` is a nadarray or NDDataset of shape
        `(C.y, len(hardConc)`, newArgsGetC are the updated parameters for the next
        iteration (can be None), and extOutput can be any relevant output to be kept
        in extOutput attribute (only the last iteration extOutput is kept)""",
    ).tag(config=True)

    argsGetConc = tr.Tuple(
        default_value=(),
        help="supplementary positional arguments passed to the external function",
    ).tag(config=True)

    kwargsGetConc = tr.Dict(
        default_value={},
        help="supplementary keyword arguments passed to the external function",
    ).tag(config=True)

    hardC_to_C_idx = tr.Union(
        (tr.Enum(["default"]), tr.List()),
        default_value="default",
        help="""Indicates the correspondence between the indexes of the columns of
        hardC and of the C matrix. [1, None, 0] indicates that the first profile in
        hardC (index O) corrsponds to the second profile of C (index 1).""",
    ).tag(config=True)

    nonnegSpec = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value="all",
        help="""Indicates non-negative spectral profile. If set to `'all'` (default)
        all spectral profiles are considered non-negative. If an array of indexes is
        passed, the corresponding profiles are considered non-negative, not the others.
        For instance `[0, 2]` indicates that profile #0 and #2 are non-negative while
        profile #1 *can* be negative. If set to `None` or `[]`, all profiles can be
        negative.""",
    ).tag(config=True)

    normSpec = tr.Enum(
        [None, "euclid", "max"],
        default_value=None,
        help="""
        Defines whether the spectral profiles should be normalized. If set to `None`
        no normalization is applied.
        when set to "euclid", spectra are normalized with respect to their total area,
        when set to "max", spectra are
        normalized with respect to the maximum af their value.
        """,
    ).tag(config=True)

    unimodSpec = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value=[],
        help="""Unimodality constraint on Spectra. If the list of spectral
        profiles is void, all profiles can be multimodal. If set to `all`, all profiles
         are unimodal. If an array of indexes is
        passed, the corresponding profiles are considered unimodal, not the others.
        For instance `[0, 2]` indicates that profile #0 and #2 are unimodal while
        profile #1 *can* be multimodal.
    """,
    ).tag(config=True)

    unimodSpecMod = tr.Enum(
        ["strict"],
        default_value="strict",
        help=""" When set to `"strict"`, values deviating from unimodality are reset to
        the value of the previous point. When set to `"smooth"`, both values (deviating
        point and previous point) are modified to avoid ="steps"
        in the concentration profile.
        """,
    ).tag(config=True)

    unimodSpecTol = tr.Float(
        default_value=1.1,
        help="""Tolerance parameter for unimodality. Correction is applied only if the
        deviating point is larger/lower than
        ```St[j,i] > St[j, i-1] * unimodSpecTol```  on the decreasing branch of
        profile #j,
        ```St[j,i] < St[j, i-1] * unimodTol```  on the increasing branch of profile  #j.
        """,
    ).tag(config=True)

    def __init__(
        self, X=None, guess=None, *, log_level=logging.WARNING, config=None, **kwargs
    ):
        # warn about deprecation
        # ----------------------
        # We use pop to remove the deprecated argument before processing the rest
        # TODO: These arguments should be removed in version 0.6 probably

        # VERBOSE
        verbose = kwargs.pop("verbose", None)
        if verbose is not None:
            warnings.warn(
                "verbose deprecated. Instead, use log_level= 'INFO' instead.",
                DeprecationWarning,
            )
        if verbose:
            log_level = "INFO"

        # set log_level of the console report
        set_loglevel(log_level)

        # unimodTol deprecation
        if "unimodTol" in kwargs.keys():
            warnings.warn(
                "unimodTol is deprecated, use unimodConcTol instead", DeprecationWarning
            )
            self.unimodConcTol = kwargs.pop("unimodTol", 1.1)

        # unimodMod deprecation
        if "unimodMod" in kwargs.keys():
            warnings.warn(
                "unimodMod is deprecated, use unimodConcMod instead", DeprecationWarning
            )
            self.unimodConcMod = kwargs.pop("unimodMod", "strict")

        # initialize the Configurable superclass
        # we could also pass the kwargs, but the superclass unfortunately doenst check
        # if the parameters exists or if it is a configuration parameters
        # thus we will do it after
        super().__init__(config=config)

        # Now, initialize the data
        if X is not None:
            self.X = X
        if guess is not None:
            self.set_profile(guess)

        for kw in list(kwargs.keys()):
            if kw in MCRALS.class_trait_names(config=True):
                v = kwargs.pop(kw)
                setattr(self, kw, v)
            else:
                raise KeyError(
                    f"'{kw}' is not a valid configuration parameters. "
                    f"Use the method `parameters()` to check the current "
                    f"allowed parameters and their current value."
                )

    # -----
    # Data
    # -----
    @property
    def X(self):
        if self._X is not None:
            return self._X
        else:
            raise ValueError(
                "input X dataset must be initialized" " before any other operation."
            )

    @X.setter
    def X(self, value):
        self._X = value
        # this implies an automatic validation of the X value

    @tr.validate("_X")
    def _X_validate(self, proposal):
        X = proposal.value
        if not isinstance(X, NDDataset):
            X = NDDataset(X)
        return X

    @property
    def C(self):
        """
        The final concentration profiles.
        """
        return self._C

    @C.setter
    def C(self, value):
        # do the profile initialisation process
        self.set_profile(value)

    @tr.validate("_C")
    def _C_validate(self, proposal):
        C = proposal["value"]
        if self.X.shape[0] != C.shape[0]:
            # An error will be raised before if X is None.
            raise ValueError(
                f"the dimensions of C [{C.shape}] do not match those "
                f"of X [{self.X.shape}]."
            )
        return C

    @property
    def St(self):
        """
        The final spectra profiles.
        """
        return self._St

    @St.setter
    def St(self, value):
        # do the profile initialisation process
        self.set_profile(value)

    @tr.validate("_St")
    def _St_validate(self, proposal):
        St = proposal["value"]
        if self.X.shape[1] != St.shape[1]:
            # An error will be raised before if X is None.
            raise ValueError(
                f"The dimensions of St [{St.shape}] do not match those "
                f"of X [{self.X.shape}]."
            )
        # initialisation required
        return St

    def set_profile(self, profile):
        """
        Set or guess an initial profile.

        Parameters
        ----------
        profile : array-like
            Initial guess for the concentration or spectra profile.
        """
        if self.X is None:
            raise ValueError("X dataset must be defined first")

        # Eventually transform the given profile to a NDDataset
        if not isinstance(profile, NDDataset):
            profile = NDDataset(profile)

        # check the dimensions compatibility
        if (self.X.shape[1] != profile.shape[1]) and (
            self.X.shape[0] != profile.shape[0]
        ):
            raise ValueError(
                f"None of the dimensions of the given profile "
                f"[{profile.shape}] correspond to any of those "
                f"of X [{self.X.shape}]."
            )

        # make the profile

        try:  # first try on concentration

            # The data are validated in _C_validate()
            # if it fails here due to shape mismatch, it goes to the except
            C = self._C = profile.copy()

            C.name = "Pure conc. profile, mcs-als of " + self.X.name
            if C.coordset is None:
                C.set_coordset(y=self.X.y, x=C.x)
            self._nspecies = C.shape[1]
            debug_("Concentration profile initialized")

            # compute initial spectra
            St = self._St = NDDataset(
                np.linalg.lstsq(C.data, self.X.data, rcond=None)[0]
            )
            St.name = "Pure spectra profile, mcs-als of " + self.X.name
            St.title = self.X.title
            cy = C.x.copy() if C.x else None
            cx = self.X.x.copy() if self.X.x else None
            St.set_coordset(y=cy, x=cx)
            debug_("Spectra profile computed")

            # if everything went well here, C and St are set, we return !
            return

        except ValueError:
            pass

        # Again if something is wrong we let it raise the error
        # as there is no other possibility (but this should not occur as we did
        # already the test on the dimension's compatibility.
        St = self._St = profile.copy()
        St.name = "Pure spectra profile, mcs-als of " + self.X.name
        if St.coordset is None:
            St.set_coordset(y=St.y, x=self.X.x)
        self._nspecies = St.shape[0]
        debug_("Spectra profile initialized")

        # compute initial concentration
        Ct = np.linalg.lstsq(St.data.T, self.X.data.T, rcond=None)[0]
        C = self._C = NDDataset(Ct.T)
        C.name = "Pure conc. profile, mcs-als of " + self.X.name
        C.title = "concentration"
        cx = St.y.copy() if St.y else None
        cy = self.X.y.copy() if self.X.y else None
        C.set_coordset(y=cy, x=cx)
        debug_("Concentration profile computed")
        return

    @property
    def nspecies(self):
        if self._nspecies is None:
            raise ValueError(
                "Species has not yet been initialized. "
                "Use `set_profile` function to set the concentration "
                "or spectra profiles"
            )
        return self._nspecies

    @tr.validate("nonnegConc")
    def _validate_nonnegConc(self, proposal):
        nonnegConc = proposal.value
        if nonnegConc == "all":
            nonnegConc = np.arange(self.nspecies)
        elif np.any(nonnegConc) and (
            len(nonnegConc) > self.nspecies or max(nonnegConc) + 1 > self.nspecies
        ):  # note that we use np.any(nnonnegConc) instead of nnonnegConc != []
            # due to a deprecation warning from traitlets.
            raise ValueError(
                f"The profile has only {self.nspecies} species, please check "
                f"the `nonnegConc` configuration (value:{nonnegConc})"
            )
        return nonnegConc

    @tr.validate("unimodConc")
    def _validate_unimodConc(self, proposal):
        unimodConc = proposal.value
        if unimodConc == "all":
            unimodConc = np.arange(self.nspecies)
        elif np.any(unimodConc) and (
            len(unimodConc) > self.nspecies or max(unimodConc) + 1 > self.nspecies
        ):
            raise ValueError(
                f"The profile has only {self.nspecies} species, please check the "
                f"`unimodConc` configuration (value:{unimodConc})"
            )
        return unimodConc

    @tr.validate("closureTarget")
    def _validate_closureTarget(self, proposal):
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

    # constraints on spectra

    @tr.validate("hardC_to_C_idx")
    def _validate_hardC_to_C_idx(self, proposal):
        hardC_to_C_idx = proposal.value
        if hardC_to_C_idx == "default":
            hardC_to_C_idx = np.arange(self.nspecies)
        elif (
            len(hardC_to_C_idx) > self.nspecies
            or max(hardC_to_C_idx) + 1 > self.nspecies
        ):
            raise ValueError(
                f"The profile has only {self.nspecies} species, please check "
                f"the `hardC_to_C_idx`  configuration (value:{hardC_to_C_idx})"
            )
        return hardC_to_C_idx

    @tr.validate("nonnegSpec")
    def _validate_nonnegSpec(self, proposal):
        nonnegSpec = proposal.value
        if nonnegSpec == "all":
            nonnegSpec = np.arange(self.nspecies)
        elif np.any(nonnegSpec) and (
            len(nonnegSpec) > self.nspecies or max(nonnegSpec) + 1 > self.nspecies
        ):
            raise ValueError(
                f"The profile has only {self.nspecies} species, please check "
                f"the `nonnegSpec`configuration (value:{nonnegSpec})"
            )
        return nonnegSpec

    @tr.validate("unimodSpec")
    def _validate_unimodSpec(self, proposal):
        unimodSpec = proposal.value
        if unimodSpec == "all":
            unimodSpec = np.arange(self.nspecies)
        elif np.any(unimodSpec) and (
            len(unimodSpec) > self.nspecies or max(unimodSpec) + 1 > self.nspecies
        ):
            raise ValueError(
                f"The profile has only {self.nspecies} species, please check the "
                f"`unimodSpec`configuration"
            )
        return unimodSpec

    @tr.observe("_nspecies")
    def _observe_nspecies_change(self, change):
        if self._nspecies > 0:
            # perform a validation of default configuration parameters
            # Indeed, if not forced here these parameters are validated only when they
            # are set explicitely.
            # Here is a ugly trick to force this validation.
            self.nonnegConc = self.nonnegConc
            self.nonnegSpec = self.nonnegSpec
            self.unimodConc = self.unimodConc
            self.unimodSpec = self.unimodSpec
            self.closureTarget = self.closureTarget
            self.hardC_to_C_idx = self.hardC_to_C_idx

    # ----------------------
    # Main execution routine
    # ----------------------
    def run(self):

        ny, _ = self.X.shape
        change = self.tol + 1
        stdev = self.X.std()
        niter = 0
        ndiv = 0

        info_("***           ALS optimisation log            ***")
        info_("#iter     RSE / PCA        RSE / Exp      %change")
        info_("-------------------------------------------------")

        # Get the data member only once
        Stdata = self.St.data
        Xdata = self.X.data

        # init the Chard and Stsoft
        self._Chard = self._C.copy()
        self._Stsoft = self._St.copy()

        # makes a PCA with same number of species for further comparison
        Xpcadata = PCA(self.X).reconstruct(n_pc=self.nspecies).data

        while change >= self.tol and niter < self.maxit and ndiv < self.maxdiv:

            Cdata = np.linalg.lstsq(Stdata.T, Xdata.T, rcond=None)[0].T
            niter += 1

            # Force non-negative concentration
            # --------------------------------
            if np.any(self.nonnegConc):
                for s in self.nonnegConc:
                    Cdata[:, s] = Cdata[:, s].clip(min=0)

            # Force unimodal concentration
            # ----------------------------
            if np.any(self.unimodConc):
                Cdata = _unimodal_2D(
                    Cdata,
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
                        if Cdata[curid + 1, s] < Cdata[curid, s] / self.monoIncTol:
                            Cdata[curid + 1, s] = Cdata[curid, s]

            # Force monotonic decrease
            # ----------------------------------------------
            if np.any(self.monoDecConc):
                for s in self.monoDecConc:
                    for curid in np.arange(ny - 1):
                        if Cdata[curid + 1, s] > Cdata[curid, s] * self.monoDecTol:
                            Cdata[curid + 1, s] = Cdata[curid, s]

            # Closure
            # ------------------------------------------
            if self.closureConc:
                if self.closureMethod == "scaling":
                    Q = np.linalg.lstsq(
                        Cdata[:, self.closureConc],
                        self.closureTarget.T,
                        rcond=None,
                    )[0]
                    Cdata[:, self.closureConc] = np.dot(
                        Cdata[:, self.closureConc], np.diag(Q)
                    )
                elif self.closureMethod == "constantSum":
                    totalConc = np.sum(Cdata[:, self.closureConc], axis=1)
                    Cdata[:, self.closureConc] = (
                        Cdata[:, self.closureConc]
                        * self.closureTarget[:, None]
                        / totalConc[:, None]
                    )

            # external concentration profiles
            # ------------------------------------------
            if np.any(self.hardConc):
                if self.kwargsGetConc != {} and self.argsGetConc != ():
                    output = self.getConc(
                        Cdata, *self.argsGetConc, **self.kwargsGetConc
                    )
                elif self.kwargsGetConc == {} and self.argsGetConc != ():
                    output = self.getConc(Cdata, *self.argsGetConc)
                elif self.kwargsGetConc != {} and self.argsGetConc == ():
                    output = self.getConc(Cdata, **self.kwargsGetConc)
                else:
                    output = self.getConc(Cdata)

                if isinstance(output, tuple):
                    fixedC = output[0]
                    self.argsGetConc = output[1]
                    if len(output) == 3:
                        extOutput = output[2]
                    else:
                        fixedC = output
                else:
                    fixedC = output
                    extOutput = None

                Cdata[:, self.hardConc] = fixedC[:, self.hardC_to_C_idx]

            # stores C in Chard
            Charddata = Cdata.copy()

            # compute St
            Stdata = np.linalg.lstsq(Cdata, Xdata, rcond=None)[0]

            # stores St in Stsoft
            Stsoftdata = Stdata.copy()

            # Force non-negative spectra
            # --------------------------
            if np.any(self.nonnegSpec):
                Stdata[self.nonnegSpec, :] = Stdata[self.nonnegSpec, :].clip(min=0)

            # Force unimodal spectra
            # ----------------------------
            if np.any(self.unimodSpec):
                Stdata = _unimodal_2D(
                    Stdata,
                    idxes=self.unimodSpec,
                    axis=1,
                    tol=self.unimodSpecTol,
                    mod=self.unimodSpecMod,
                )

            # recompute C for consistency(soft modeling)
            Cdata = np.linalg.lstsq(Stdata.T, Xdata.T)[0].T

            # rescale spectra & concentrations
            if self.normSpec == "max":
                alpha = np.max(Stdata, axis=1).reshape(self.nspecies, 1)
                Stdata = Stdata / alpha
                Cdata = Cdata * alpha.T
            elif self.normSpec == "euclid":
                alpha = np.linalg.norm(Stdata, axis=1).reshape(self.nspecies, 1)
                Stdata = Stdata / alpha
                Cdata = Cdata * alpha.T

            # compute residuals
            # -----------------
            Xhatdata = np.dot(Cdata, Stdata)
            stdev2 = np.std(Xhatdata - Xdata)
            change = 100 * (stdev2 - stdev) / stdev
            stdev = stdev2

            stdev_PCA = np.std(Xhatdata - Xpcadata)  #

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

            if niter == self.maxit:
                info_(
                    f"Convergence criterion ('tol') not reached after "
                    f"{ self.maxit:d} iterations."
                )
                info_("Stop ALS optimization.")

        if np.any(self.hardConc):
            self._extOutput = extOutput
        else:
            self._extOutput = None

        self._Stsoft.data = Stsoftdata
        self._Chard.data = Charddata

    @property
    def extOutput(self):
        """
        The last output of the external function used to get concentrations.
        """
        return self._extOutput

    @property
    def Stsoft(self):
        """
        The soft spectra profiles.
        """
        return self._Stsoft

    @property
    def Chard(self):
        """
        The hard concentration profiles.
        """
        return self._Chard

    @property
    def log(self):
        """
        Logs output.
        """
        # A string handler (#2) is defined for the Spectrochempy logger,
        # thus we will return it's content
        return app.log.handlers[2].stream.getvalue().rstrip()

    @property
    @deprecated("Use log instead. This attribute may be removed in future version")
    def logs(self):
        """
        Logs output.
        """
        return self.log

    def reconstruct(self):
        """
        Transform data back to the original space.

        The following matrice operation is performed : :math:`X'_{hat} = C'.S'^t`.

        Returns
        -------
        X_hat : |NDDataset|
            The reconstructed dataset based on the MCS-ALS optimization.
        """

        # reconstruct from concentration and spectra profiles
        C = self.C
        St = self.St

        X_hat = dot(C, St)

        X_hat.history = "Dataset reconstructed by MCS ALS optimization"
        X_hat.title = "X_hat: " + self.X.title
        return X_hat

    def plotmerit(self, **kwargs):
        """
        Plots the input dataset, reconstructed dataset and residuals.

        Parameters
        ----------
        **kwargs
            optional "colors" argument: tuple or array of 3 colors for :math:`X`, :math:`\hat X` and :math:`E`.

        Returns
        -------
        ax
            subplot.
        """
        colX, colXhat, colRes = kwargs.get("colors", ["blue", "green", "red"])

        X_hat = self.reconstruct()
        res = self.X - X_hat
        ax = self.X.plot()
        if self.X.x is not None:
            ax.plot(self.X.x.data, X_hat.T.data, color=colXhat)
            ax.plot(self.X.x.data, res.T.data, color=colRes)
        else:
            ax.plot(X_hat.T.data, color=colXhat)
            ax.plot(res.T.data, color=colRes)
        ax.autoscale(enable=True)
        ax.set_title("MCR ALS merit plot")
        return ax

    # Utility functions

    def parameters(self, default=False):
        """
        Return current or default configuration values

        Parameters
        ----------
        default : Bool, optional, default: False
            If 'default' is True, the default parameters are returned,
            else the current values.

        Returns
        -------
        dict
        """
        d = Meta()
        if not default:
            d.update(self.trait_values(config=True))
        else:
            d.update(self.trait_defaults(config=True))
        return d

    def help(self):
        """
        Return a description of all configuration parameters with their default value

        """
        return self.class_config_rst_doc()

    def reset(self):
        """
        Reset configuration to default
        """
        for k, v in self.parameters(default=True).items():
            setattr(self, k, v)


# ---------------------------------
def _unimodal_2D(a, axis, idxes, tol, mod):
    """Force unimodality on given lines or columnns od a 2D ndarray

    a: ndarray

    axis: int
        axis along which the correction is applied

    idxes: list of int
        indexes at which the correction is applied

    mod : str
        When set to `"strict"`, values deviating from unimodality are reset to the value of the previous point.
        When set to `"smooth"`, both values (deviating point and previous point) are modified to avoid "steps"
        in the profile.

    tol: float
        Tolerance parameter for unimodality. Correction is applied only if:
        `a[i] > a[i-1] * unimodTol`  on a decreasing branch of profile,
        `a[i] < a[i-1] * unimodTol`  on an increasing branch of profile.
    """

    if axis == 0:
        a_ = a
    elif axis == 1:
        a_ = a.T

    for col, idx in zip(a_[:, idxes].T, idxes):
        a_[:, idx] = _unimodal_1D(col, tol, mod)

    return a


def _unimodal_1D(a: np.ndarray, tol: str, mod: str) -> np.ndarray:
    """force unimodal concentration

    makes a vector unimodal

    parameters:
    ----------
    a : 1D ndarray

    mod : str
        When set to `"strict"`, values deviating from unimodality are reset to the value of the previous point.
        When set to `"smooth"`, both values (deviating point and previous point) are modified to avoid "steps"
        in the profile.

    tol: float
        Tolerance parameter for unimodality. Correction is applied only if:
        `a[i] > a[i-1] * unimodTol`  on a decreasing branch of profile,
        `a[i] < a[i-1] * unimodTol`  on an increasing branch of profile.
    """

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
