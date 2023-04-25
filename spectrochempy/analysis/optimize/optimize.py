# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

__all__ = ["Optimize"]
__configurables__ = __all__

import re
import sys

import numpy as np
import scipy.optimize
import traitlets as tr
from IPython import display

from spectrochempy.analysis._base import DecompositionAnalysis
from spectrochempy.analysis.optimize import _models as models_
from spectrochempy.analysis.optimize._parameters import FitParameters
from spectrochempy.application import info_, warning_
from spectrochempy.extern.traittypes import Array
from spectrochempy.utils.decorators import signature_has_configurable_traits
from spectrochempy.utils.docstrings import _docstring


# ======================================================================================
@signature_has_configurable_traits
class Optimize(DecompositionAnalysis):
    __doc__ = _docstring.dedent(
        """
    Non-linear Least-Square Optimization and Curve-Fitting.

    Works on a 1D or 2D dataset.

    # TODO: complete this description

    Parameters
    ----------
    %(AnalysisConfigurable.parameters)s
    """
    )

    name = "Optimize"
    description = "Non-linear Least-Squares Optimization"

    # ----------------------------------------------------------------------------------
    # Configuration parameters (mostly defined in subclass
    # as they depend on the model estimator)
    # ----------------------------------------------------------------------------------
    max_iter = tr.Integer(
        default_value=500, help="Maximum number of fitting iteration."
    ).tag(config=True)

    max_fun_calls = tr.Integer(
        allow_none=True, help="Maximum number of function calls at each iteration."
    ).tag(config=True)

    callback_every = tr.Integer(
        default_value=10,
        help="Number of iteration between each callback report. "
        "Used for printing or display intermediate results.",
    ).tag(config=True)

    method = tr.CaselessStrEnum(
        ["SIMPLEX", "HOPPING"], default_value="SIMPLEX", help="Optimization method."
    ).tag(config=True)

    script = tr.Unicode(help="Script defining models and parameters for fitting.").tag(
        config=True
    )

    constraints = tr.Any(allow_none=True, help="Constraints.").tag(
        config=True
    )  # TODO: adjust this

    dry = tr.Bool(
        default_value=False,
        help="If True perform a dry run. "
        "Mainly used to check the validity of the input parameters.",
    ).tag(config=True)

    autobase = tr.Bool(
        default_value=False, help="Whether to apply an automatic baseline correction."
    ).tag(config=True)

    autoampl = tr.Bool(
        default_value=False, help="Whether to apply an automatic amplitude correction."
    ).tag(config=True)

    amplitude_mode = tr.CaselessStrEnum(
        ["area", "height"],
        default_value="height",
        help="Initial amplitude setting mode.",
    ).tag(config=True)

    # ----------------------------------------------------------------------------------
    # Runtime Parameters (in addition to those of AnalysisConfigurable)
    # ----------------------------------------------------------------------------------
    fp = tr.Instance(FitParameters, allow_none=True)
    modeldata = tr.List(Array())

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        *,
        log_level="WARNING",
        warm_start=False,
        **kwargs,
    ):
        """ """
        # An empty __doc__ must be placed here, else Configurable.__doc__ will appear

        # call the super class for initialisation of the configuration parameters
        # to do before anything else!
        super().__init__(
            log_level=log_level,
            warm_start=warm_start,
            **kwargs,
        )

    # ----------------------------------------------------------------------------------
    # Private methods ( overriding abstract methods)
    # ----------------------------------------------------------------------------------
    def _fit(self, X, Y=None):

        # NMR
        # sequence = kargs.get('sequence', 'ideal_pulse')
        # self.sequence = PulseSequence(type=sequence)

        # create model data
        modeldata, modelnames, model_A, model_a, model_b = self._get_modeldata(X)

        info_("*" * 50)
        info_("  Entering fitting procedure")
        info_("*" * 50)

        global niter, chi2, everyiter, ncalls
        ncalls = 0
        everyiter = self.callback_every
        niter = 0

        # # internally defined function chi2
        # def funchi2(params, datasets, *constraints):
        #     """
        #     Return sum((y - x)**2)
        #     """
        #     global chi2, ncalls
        #     # model spectrum
        #
        #     chi2 = 0
        #     som = 0
        #     ncalls += 1
        #
        #     for exp_idx, dataset in enumerate(datasets):
        #         modeldata = self._get_modeldata(dataset, exp_idx)[0]
        #         # baseline is already summed with modeldata[-1]
        #
        #         # important to work with the real component of dataset
        #         # not the complex number
        #         data = dataset.real.data.squeeze()
        #
        #         # if not dataset.is_2d:
        #         mdata = modeldata[-1]  # modelsum
        #
        #         # else:
        #         #    mdata = modeldata.values
        #
        #         merror = 1.0
        #         # if dataset.is_2d:
        #         #     if constraints:
        #         #
        #         #         # Case of SQ-DQ experiments
        #         #         if self.kind == 'SQ-DQ' and \
        #         #                         'max_connections' in constraints[0]:
        #         #             # check connectivity numbers
        #         #             nbconnections = {}
        #         #             for key in params.keys():
        #         #                 if 'pos1' in key:
        #         #                     connect = key[-2:]
        #         #                     key = 'ampl_line_' + connect  # get amplitude
        #         #                     ki = connect[0].upper()
        #         #                     if ki not in nbconnections.keys():
        #         #                         nbconnections[ki] = 0
        #         #                     if int(params[key]) > 0:
        #         #                         nbconnections[ki] += 1
        #         #             for k, v in nbconnections.iteritems():
        #         #                 if v > constraints[0]['max_connections']:
        #         #                     merror *= v * 10.
        #
        #         diff = data - mdata
        #         chi2 += np.sum(diff**2) * merror
        #         som += np.sum(data[0] ** 2)
        #
        #     chi2 = np.sqrt(chi2 / som)
        #     # reset log_level
        #     return chi2
        # internally defined function chi2
        def funchi2(params, X, *constraints):
            """
            Return sum((y - x)**2)
            """
            global chi2, ncalls
            # model spectrum

            chi2 = 0
            ncalls += 1

            # model
            modeldata = self._get_modeldata(X)[0]
            # baseline is already summed with modeldata[-1]
            mdata = modeldata[-1]  # modelsum

            # important to work with the real component of dataset
            # not the complex number
            data = X.real.squeeze()

            diff = data - mdata
            chi2 += np.sum(diff**2)  # * merror
            return chi2

        # end chi2 function ---------------------------------------------------

        # callback function--------------------------------------------------------
        def callback(*args, **kwargs):
            """
            callback log.info function
            """
            global niter, chi2, everyiter, ncalls
            niter += 1

            if niter % everyiter != 0:
                return

            display.clear_output(wait=True)
            info_(f"Iterations: {niter}, Calls: {ncalls} (chi2: {chi2:.5f})")
            sys.stdout.flush()

        # end callback function ---------------------------------------------------

        fp = self.fp  # starting parameters

        if not self.dry:
            fp, fopt = _optimize(
                funchi2,
                fp,
                args=(X,),
                maxfun=self.max_fun_calls,
                maxiter=self.max_iter,
                method=self.method,
                constraints=self.constraints,
                callback=callback,
            )

        # replace the previous script with new fp parameters
        self.script = str(fp)

        # log.info the results
        info_("\n")
        info_("*" * 50)
        if not self.dry:
            info_("  Result:")
        else:
            info_("  Starting parameters:")
        info_("*" * 50)
        info_(self.script)

        # reset dry and continue to show starting model
        self.dry = False

        # return fit results
        modeldata, names, A, a, b = self._get_modeldata(X)
        if X.squeeze().ndim == 1:
            # C in this case is just the A for all species
            # (not very useful here but it will be necessary for 2D
            # we eventually add baseline to the components
            start = 0 if self.autobase else 1
            C = np.ones((start, self._n_components)) * A
            components = modeldata[start:-1]
            total = modeldata[-1]
        else:
            # todo
            pass
        _outfit = C, components, total, A, a, b
        return _outfit

    # ----------------------------------------------------------------------------------
    # Private methods for validation
    # ----------------------------------------------------------------------------------
    @tr.validate("script")
    def _script_validate(self, proposal):
        script = proposal.value

        # init some flags
        modlabel = None
        common = False
        fixed = False
        reference = False

        # create a new FitParameters instance
        fp = FitParameters()

        # start interpreting -----------------------------------------------------------
        lines = script.split("\n")
        lc = 0

        for item in lines:
            lc += 1  # -------------- count the lines
            line = item.strip()
            if line == "" or line.startswith("#"):
                # this is a blank or comment line, go to next line
                continue
            # split around the semi-column
            s = line.split(":")
            if len(s) != 2:
                raise ValueError(
                    f"Cannot interpret line {lc}: A semi-column is missing?"
                )

            key, values = s
            key = key.strip().lower()
            if key.startswith("model"):
                modlabel = values.lower().strip()
                if modlabel not in fp.models:
                    fp.models.append(modlabel)
                common = False
                continue
            elif key.startswith("common") or key.startswith("vars"):
                common = True
                modlabel = "common"
                continue
            elif key.startswith("shape"):
                shape = values.lower().strip()
                if shape is None:  # or (shape not in self._list_of_models and shape not
                    # in self._list_of_baselines):
                    raise ValueError(
                        f"Shape of this model `{shape}` was not specified"
                        f" or is not implemented"
                    )
                fp.model[modlabel] = shape
                common = False
                continue
            # elif key.startswith("experiment"):  # must be in common
            #     if not common:
            #         raise ValueError(
            #             "'experiment_...' specification was found outside the common
            #             block."
            #         )
            #     if "variables" in key:
            #         expvars = values.lower().strip()
            #         expvars = expvars.replace(",", " ").replace(";", " ")
            #         expvars = expvars.split()
            #         fp.expvars.extend(expvars)
            #     continue
            else:
                if modlabel is None and not common:
                    raise ValueError(
                        "The first definition should be a label for a model or a block "
                        "of variables or constants."
                    )
                # get the parameters
                if key.startswith("*"):
                    fixed = True
                    reference = False
                    key = key[1:].strip()
                elif key.startswith("$"):
                    fixed = False
                    reference = False
                    key = key[1:].strip()
                elif key.startswith(">"):
                    fixed = True
                    reference = True
                    key = key[1:].strip()
                else:
                    raise ValueError(
                        f"Cannot interpret line {lc}: A parameter definition must start"
                        f" with *,$ or >"
                    )

                # store this parameter
                s = values.split(",")
                s = [ss.strip() for ss in s]
                if len(s) > 1 and ("[" in s[0]) and ("]" in s[1]):  # list
                    s[0] = "%s, %s" % (s[0], s[1])
                    if len(s) > 2:
                        s[1:] = s[2:]
                if len(s) > 3:
                    raise ValueError(
                        f"line {lc}: value, min, max should be defined in this order"
                    )
                elif len(s) == 2:
                    raise ValueError(f"only two items in line {lc}")
                    # s.append('none')
                elif len(s) == 1:
                    s.extend(["none", "none"])
                value, mini, maxi = s
                if mini.strip().lower() in ["none", ""]:
                    mini = str(-1.0 / sys.float_info.epsilon)
                if maxi.strip().lower() in ["none", ""]:
                    maxi = str(+1.0 / sys.float_info.epsilon)
                if modlabel != "common":
                    ks = f"{key}_{modlabel}"
                    fp.common[key] = False
                else:
                    ks = f"{key}"
                    fp.common[key] = True
                fp.reference[ks] = reference
                if not reference:
                    val = value.strip()
                    val = eval(val)
                    # if isinstance(val, list):
                    #     # if the parameter is already a list, that's ok if the number
                    #     # of parameters is ok
                    #     if len(val) != fp.expnumber:
                    #         raise ValueError(
                    #             f"the number of parameters {len(val)} is not the number "
                    #             f"of experiments."
                    #         )
                    #     if key not in fp.expvars:
                    #         raise ValueError(
                    #             f"parameter {key} is not declared as variable"
                    #         )
                    # else:
                    #     if key in fp.expvars:
                    #         # we create a list of parameters corresponding
                    #         val = [val] * fp.expnumber
                    fp[ks] = val, mini.strip(), maxi.strip(), fixed
                else:
                    fp[ks] = value.strip()

        # update global fp
        self.fp = fp

        # return validated script
        return script

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------
    @tr.default("_script")
    def _script_default(self):
        """
        Return a default script.
        """
        return """
        # -----------------------------------------------------------
        # syntax for parameters definition:
        # name: value, low_bound,  high_bound
        # prefix:
        #  # for comments
        #  * for fixed parameters
        #  $ for variable parameters
        #  > for reference to a parameter in the COMMON block
        #    (> is forbidden in the COMMON block)
        # common block parameters should not have a _ in their names
        # -----------------------------------------------------------
        #

        COMMON:
        # common parameters
        # $ gwidth: 1.0, 0.0, none
          $ gratio: 0.5, 0.0, 1.0

        MODEL: LINE_1
        shape: voigtmodel
            $ ampl:  1.0, 0.0, none
            $ pos:   0.0, -100.0, 100.0
            > ratio: gratio
            $ width: 1.0, 0, 100
        """

    # def _repr_html_(self):
    #     if not self.datasets:
    #         return htmldoc(self.script)
    #     else:
    #         return self.message

    def _get_modeldata(self, X, exp_idx=1):
        # exp_idx is not used for the moment, but will be necessary for multidataset
        # fitting

        # Prepare parameters
        parameters = self._prepare(self.fp, exp_idx)

        # Get the list of models
        models = self.fp.models
        self._n_components = nbmodels = len(models)

        # Make an array 'modeldata' with the size of the dataset of data
        # which will contains the data produced by the models
        # This name must always be 'modeldata'
        # which will be returned to the main program.

        expedata = X.real.squeeze()

        # we need to calculate the model with the full unmasked coordinates

        if expedata.ndim > 1:
            # nD data
            raise NotImplementedError("Fit not implemented for nD data yet!")

        # we need to keep track of the x axis before masking
        axis, dim = self._X.get_axis(-1)
        _xaxis = self._X_coordset[dim].data

        x = _xaxis
        modeldata = np.zeros((nbmodels + 2, x.size), dtype=np.float64)

        if nbmodels < 1:
            names = ["baseline", "modelsum"]
            return modeldata, names

        # Calculates model data
        # The first row (i=0) of the modeldata array is the baseline,
        # so we fill the array starting at row 1
        row = 0
        names = [
            "baseline",
        ]

        for model in models:
            calc = getmodel(
                x, modelname=model, par=parameters, amplitude_mode=self.amplitude_mode
            )
            if not model.startswith("baseline"):
                row += 1
                modeldata[row] = calc
                names.append(model)
            else:
                modeldata[0] += calc

        # make the sum
        modeldata[row + 1] = modeldata.sum(axis=0)
        names.append("modelsum")

        # remove unused column
        modeldata = modeldata[: row + 2]

        # remove masked column
        if np.any(self._X_mask):
            masked_columns = np.all(self._X_mask, axis=-2)
            modeldata = modeldata[:, ~masked_columns]
            x = x[~masked_columns]

        if self.autobase:
            A, a, b = self._ampbas(x, expedata, modeldata[-1])
        else:
            A, a, b = 1.0, 0.0, 0.0

        # (fitzone-fitzone[0], data.take(fitzone),
        # modeldata[-1].take(fitzone))

        modeldata = A * modeldata
        baseline = a * x + b  # a*(xi-fitzone[0]) + b

        # update the modeldata
        modeldata[0] += baseline
        modeldata[-1] += baseline

        # return modeldata
        return modeldata, names, A, a, b

    @staticmethod
    def _parsing(expr, param):
        keyword = r"\b([a-z]+[0-9]*)\b"  # match a whole word
        pat = re.compile(keyword)
        mo = pat.findall(str(expr))
        for kw in mo:
            if kw in param:
                expr = expr.replace(kw, str(param[kw]))
            elif kw in np.__dict__:  # check if it is a recognized math function
                expr = expr.replace(kw, "np.%s" % kw)
        return expr

    def _prepare(self, param, exp_idx=1):
        # This function is needed for the script related to modelfunction
        #
        # exp_idx: int, contains the index of the experiment

        new_param = param.copy()

        for key in param:

            if param.reference[key]:
                new_param.reference[key] = False
                # important to put it here
                # before other instruction
                # try to interpret the given refpar expression
                refpar = param[key]
                while True:
                    par = self._parsing(refpar, new_param)
                    if par == refpar:
                        break
                    refpar = par
                try:
                    new_param[key] = eval(str(refpar))
                except Exception:
                    raise ValueError(
                        "Cannot evaluate the expression %s: %s" % (key, param[refpar])
                    )

                new_param.fixed[key] = True
                new_param.reference[key] = True  # restore it for the next call

            # if isinstance(new_param[key], list):
            #     new_param.data[key] = new_param.data[key][exp_idx]

        return new_param

    # ==================================================================================
    # automatic calculation of amplitude and baseline
    # ==================================================================================
    @staticmethod
    def _ampbas(xi, expe, calc):
        # Automatically calculate correct amplitude A and baseline
        # (baseline linear model a*i+b) by determining the zero of the first derivative
        # with respect to A, a, and b
        expe = expe.squeeze()
        n = xi.size
        sE = sum(expe)
        sF = sum(calc)
        sFI = sum(xi * calc)
        sFd = sum(calc * calc)
        sI = sum(xi)
        sEF = sum(expe * calc)
        sEI = sum(xi * expe)
        sId = sum(xi**2)

        den = (
            n * sFI**2
            - n * sFd * sId
            + sF**2 * sId
            - 2 * sF * sFI * sI
            + sFd * sI**2
        )

        a = (
            -sE * (sF * sFI - sFd * sI)
            + sEF * (n * sFI - sF * sI)
            - sEI * (n * sFd - sF**2)
        ) / den

        A = (
            sE * (sF * sId - sFI * sI)
            - sEF * (n * sId - sI**2)
            + sEI * (n * sFI - sF * sI)
        ) / den

        b = (
            sE * (sFI**2 - sFd * sId)
            + sEF * (sF * sId - sFI * sI)
            - sEI * (sF * sFI - sFd * sI)
        ) / den

        # in case the modeldata is zero, to avoid further errors
        if np.isnan(A):  # pragma: no cover
            A = 0.0
        if np.isnan(a):  # pragma: no cover
            a = 0.0
        if np.isnan(b):  # pragma: no cover
            b = 0.0
        return A, a, b

    @staticmethod
    def _ampbas2D(xi, yj, expe, calc):  # pragma: no cover
        n = float(xi.size)
        m = float(yj.size)
        sE = expe.sum()
        sF = calc.sum()
        sFI = (xi * calc).sum()
        sFJ = (yj * calc.T).sum()
        sFd = (calc * calc).sum()
        sI = sum(xi)
        sJ = sum(yj)
        sIJ = ((np.ones_like(calc) * xi).T * yj).sum()
        sEF = (expe * calc).sum()
        sEI = (xi * expe).sum()
        sEJ = (yj * expe.T).sum()
        sId = sum(xi**2)
        sJd = sum(yj**2)

        den = (
            -(m**2) * n**2 * sFd * sId * sJd
            + m**2 * n * sFJ**2 * sId
            + m**2 * n * sFd * sI**2 * sJd
            - m**2 * sFJ**2 * sI**2
            + m * n**2 * sFI**2 * sJd
            + m * n**2 * sFd * sId * sJ**2
            + m * n * sF**2 * sId * sJd
            - 2 * m * n * sF * sFI * sI * sJd
            - 2 * m * n * sF * sFJ * sId * sJ
            + 2 * m * n * sFI * sFJ * sI * sJ
            - 2 * m * n * sFI * sFJ * sIJ
            - 2 * m * n * sFd * sI * sIJ * sJ
            + m * n * sFd * sIJ**2
            + 2 * m * sF * sFJ * sI * sIJ
            - n**2 * sFI**2 * sJ**2
            + 2 * n * sF * sFI * sIJ * sJ
            - sF**2 * sIJ**2
        )

        c = (
            sE
            * (
                -m * n * sFd * sId * sJd
                + m * sFJ**2 * sId
                + n * sFI**2 * sJd
                - 2 * sFI * sFJ * sIJ
                + sFd * sIJ**2
            )
            + sEF
            * (
                m * n * sF * sId * sJd
                - m * n * sFI * sI * sJd
                - m * n * sFJ * sId * sJ
                + m * sFJ * sI * sIJ
                + n * sFI * sIJ * sJ
                - sF * sIJ**2
            )
            + sEI
            * (
                m * n * sFd * sI * sJd
                - m * sFJ**2 * sI
                - n * sF * sFI * sJd
                + n * sFI * sFJ * sJ
                - n * sFd * sIJ * sJ
                + sF * sFJ * sIJ
            )
            + sEJ
            * (
                m * n * sFd * sId * sJ
                - m * sF * sFJ * sId
                + m * sFI * sFJ * sI
                - m * sFd * sI * sIJ
                - n * sFI**2 * sJ
                + sF * sFI * sIJ
            )
        ) / den

        a = (
            n
            * sEF
            * (
                m * n * sFI * sJd
                - m * sF * sI * sJd
                + m * sFJ * sI * sJ
                - m * sFJ * sIJ
                - n * sFI * sJ**2
                + sF * sIJ * sJ
            )
            + n
            * sEI
            * (
                -m * n * sFd * sJd
                + m * sFJ**2
                + n * sFd * sJ**2
                + sF**2 * sJd
                - 2 * sF * sFJ * sJ
            )
            + sE
            * (
                m * n * sFd * sI * sJd
                - m * sFJ**2 * sI
                - n * sF * sFI * sJd
                + n * sFI * sFJ * sJ
                - n * sFd * sIJ * sJ
                + sF * sFJ * sIJ
            )
            - sEJ
            * (
                m * n * sFI * sFJ
                + m * n * sFd * sI * sJ
                - m * n * sFd * sIJ
                - m * sF * sFJ * sI
                - n * sF * sFI * sJ
                + sF**2 * sIJ
            )
        ) / den

        A = (
            m
            * n
            * sEF
            * (
                -m * n * sId * sJd
                + m * sI**2 * sJd
                + n * sId * sJ**2
                - 2 * sI * sIJ * sJ
                + sIJ**2
            )
            + m
            * sEJ
            * (
                m * n * sFJ * sId
                - m * sFJ * sI**2
                - n * sF * sId * sJ
                + n * sFI * sI * sJ
                - n * sFI * sIJ
                + sF * sI * sIJ
            )
            + n
            * sEI
            * (
                m * n * sFI * sJd
                - m * sF * sI * sJd
                + m * sFJ * sI * sJ
                - m * sFJ * sIJ
                - n * sFI * sJ**2
                + sF * sIJ * sJ
            )
            + sE
            * (
                m * n * sF * sId * sJd
                - m * n * sFI * sI * sJd
                - m * n * sFJ * sId * sJ
                + m * sFJ * sI * sIJ
                + n * sFI * sIJ * sJ
                - sF * sIJ**2
            )
        ) / den

        b = (
            m
            * sEF
            * (
                m * n * sFJ * sId
                - m * sFJ * sI**2
                - n * sF * sId * sJ
                + n * sFI * sI * sJ
                - n * sFI * sIJ
                + sF * sI * sIJ
            )
            + m
            * sEJ
            * (
                -m * n * sFd * sId
                + m * sFd * sI**2
                + n * sFI**2
                + sF**2 * sId
                - 2 * sF * sFI * sI
            )
            + sE
            * (
                m * n * sFd * sId * sJ
                - m * sF * sFJ * sId
                + m * sFI * sFJ * sI
                - m * sFd * sI * sIJ
                - n * sFI**2 * sJ
                + sF * sFI * sIJ
            )
            - sEI
            * (
                m * n * sFI * sFJ
                + m * n * sFd * sI * sJ
                - m * n * sFd * sIJ
                - m * sF * sFJ * sI
                - n * sF * sFI * sJ
                + sF**2 * sIJ
            )
        ) / den

        # in case the modeldata is zero, to avoid further errors
        if np.isnan(A):
            A = 0.0
        if np.isnan(a):
            a = 0.0
        if np.isnan(b):
            b = 0.0
        if np.isnan(c):
            c = 0.0
        return A, a, b, c

    # ----------------------------------------------------------------------------------
    # Public methods and properties
    # ----------------------------------------------------------------------------------
    def _transform(self, X=None):
        # X is ignored for Optimize
        # this method is present for coherence with other decomposition methods
        return self._outfit[0]

    def _inverse_transform(self, X_transform=None):
        # X_transform is ignored for Optimize
        # this method is present for coherence with other decomposition methods
        X_transform = self._outfit[2]
        if X_transform.ndim == 1:
            # we need a seconddimension of size 1 for the restoration of masks
            X_transform = X_transform[np.newaxis]
        return X_transform

    def _get_components(self):
        return self._outfit[1]  # the first is the baseline, the last is the sum

    # ----------------------------------------------------------------------------------
    # Public methods/properties
    # ----------------------------------------------------------------------------------
    @_docstring.dedent
    def fit(self, X):
        """
        Perform a non-linear optimization of the ``X`` dataset.

        Parameters
        ----------
        %(analysis_fit.parameters.X)s

        Returns
        -------
        %(analysis_fit.returns)s

        See Also
        --------
        %(analysis_fit.see_also)s
        """
        return super().fit(X, Y=None)


# ======================================================================================
def _optimize(
    func,
    fp0,
    args=(),
    constraints={},
    method="SIMPLEX",
    maxfun=None,
    maxiter=1000,
    ftol=1e-8,
    xtol=1e-8,
    callback=None,
):
    # """
    # Parameters
    # ----------
    # func
    # fp0
    # args
    # constraints
    # method
    # maxfun
    # maxiter
    # ftol
    # xtol
    # callback
    #
    #
    # #  Internal/external transformation
    # #  These transformations are used in the MINUIT package,
    # #  and described in detail
    # #  in the section 1.3.1 of the MINUIT User's Guide.
    # """

    global keys

    def restore_external(fp, p, keys):
        # restore external parameters
        for key in list(fp.keys()):
            keysp = key.split("_")
            if keysp[0] in fp.expvars:
                ps = []
                for i in range(fp.expnumber):
                    ks = "%s_exp%d" % (key, i)
                    if ks not in keys:
                        break
                    k = keys.index(ks)
                    ps.append(p[k])
                if len(ps) > 0:
                    fp.to_external(key, ps)
            else:
                if key not in keys:
                    continue
                k = keys.index(key)
                fp.to_external(key, p[k])
        return fp

    def internal_func(p, dat, fp, keys, *args):
        fp = restore_external(fp, p, keys)
        return func(fp, dat, *args)

    def internal_callback(*args):
        if callback is None:
            return
        return callback(*args)

    if not isinstance(fp0, FitParameters):
        raise TypeError("fp0 is not of FitParameter type")

    # make internal parameters
    par = []
    keys = []

    for key in sorted(fp0.keys()):
        if not fp0.fixed[key]:
            # we make internal parameters in case of bounding
            # We also take care of the multiple experiments
            keysp = key.split("_")[0]
            if keysp in fp0.expvars:
                for i in range(fp0.expnumber):
                    par.append(fp0.to_internal(key, i))
                    keys.append("%s_exp%d" % (key, i))
            else:
                par.append(fp0.to_internal(key))
                keys.append(key)

    args = list(args)
    args.append(fp0)
    args.append(keys)
    if constraints:
        args.append(constraints)

    if not maxfun:
        maxfun = 4 * maxiter
    if method.upper() == "SIMPLEX":
        result = scipy.optimize.fmin(
            internal_func,
            par,
            args=tuple(args),
            maxfun=maxfun,
            maxiter=maxiter,
            ftol=ftol,
            xtol=xtol,
            full_output=True,
            disp=False,
            callback=internal_callback,
        )
        res, fopt, iterations, funcalls, warnmess = result

    elif method.upper() == "HOPPING":
        result = scipy.optimize.basinhopping(
            internal_func,
            par,
            niter=100,
            T=1.0,
            stepsize=0.5,
            minimizer_kwargs={"args": tuple(args)},
            take_step=None,
            accept_test=None,
            callback=internal_callback,
            interval=50,
            disp=False,
            niter_success=None,
        )

        # fmin(func, par, args=args, maxfun=maxfun, maxiter=maxiter, ftol=ftol, xtol=xtol,
        #                                                full_output=True, disp=False, callback=callback)
        res, fopt, warnmess = result.x, result.fun, result.message

    elif method == "XXXX":
        raise NotImplementedError("method: %s" % method)
        # TODO: implement other algorithms
    else:
        raise NotImplementedError("method: %s" % method)

    # restore the external parameter
    fpe = restore_external(fp0, res, keys)
    # for i, key in enumerate(keys):
    #    fp0.to_external(key, res[i])

    if warnmess == 1:
        warning_("Maximum number of function evaluations made.")
    if warnmess == 2:
        warning_("Maximum number of iterations reached.")

    return fpe, fopt


# ======================================================================================
def getmodel(x, y=None, modelname=None, par=None, **kwargs):
    """
    Get the model for a given x vector.

    Parameters
    -----------
    x : ndarray
        Array of frequency where to evaluate the model values returned by the
        f function.
    y : ndarray or None
        None for 1D, or index for the second dimension.
    modelname : str
        Name of the model class to use.
    par : :class:`Parameters` instance
        Parameter to pass to the f function.
    kargs : any
        Keywords arguments to pass to the f function.

    Returns
    -------
    `~numpy.ndarray`
        Array containing the calculated model.
    """
    model = par.model[modelname]
    modelcls = getattr(models_, model)

    # take an instance of the model
    a = modelcls()

    # get the parameters for the given model
    args = []
    for p in a.args:
        try:
            args.append(par[f"{p}_{modelname}"])
        except KeyError as e:
            if p.startswith("c_"):
                # probably the end of the list
                # due to a limited polynomial degree
                pass
            else:
                raise ValueError(e)

    x = np.array(x, dtype=np.float64)
    if y is not None:
        y = np.array(y, dtype=np.float64)

    if y is None:
        val = a.f(x, *args, **kwargs)
    else:
        val = a.f(x, y, *args, **kwargs)

    # Return amplitude or area ? return calc is scaled by area by default
    amplitude_mode = kwargs.pop("amplitude_mode")
    if amplitude_mode.lower() == "height":
        # in this case ampl parameter is the height, so we need to rescale
        # calc
        ampl = args[0]
        val = ampl * val / val.max()
    return val
