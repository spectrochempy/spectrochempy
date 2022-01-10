# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
Module to perform fitting of 1D or n-D spectral data.
"""
__all__ = ["Fit"]

__dataset_methods__ = []

import sys
import re
from warnings import warn
from traitlets import HasTraits, Bool, Any, List, Instance
import numpy as np

# IPython
from IPython import display

from spectrochempy.core.fitting.parameters import ParameterScript
from spectrochempy.core.fitting.models import getmodel
from spectrochempy.core.fitting.optimization import optimize
from spectrochempy.utils import htmldoc
from spectrochempy.core import preferences, info_, INFO


# ======================================================================================================================
#  Fit: main object to handle a fit
# ======================================================================================================================


class Fit(HasTraits):
    """
    Fit a 1D or 2D dataset, or a list of datasets.


    Parameters
    ----------
    dataset : Dataset or list of Dataset instance
        The data to fit.

    mode : Unicode, optional
        Reserved - not used for now.

    Attributes
    ----------
    fp : Dict
        Fit parameters dictionary (read-only, but individual elements of the dict
        can be changed)

    script : Unicode
        A string representation of the fp dict,
        which can be used as input for other a fit (read-only)
    """

    silent = Bool(False)

    _ = Any()

    datasets = List(Instance("spectrochempy.core.dataset.nddataset.NDDataset"))

    parameterscript = Instance(ParameterScript)

    # *******************************************************************************
    # initialisation
    # *******************************************************************************

    def __init__(self, *args, **kwargs):

        if args:
            # look in args
            if not isinstance(args[0], list):
                self.datasets = [
                    args[0],
                ]
            else:
                self.datasets = args[0]
                # we create a list of dataset in all case
            script = args[1]

        else:
            return

        # get parameters form script
        self.parameterscript = ParameterScript(datasets=self.datasets, script=script)
        if self.fp is None:
            # for unknown reason for now, this sometimes happens during tests
            warn("error with fp")

        # sequence = kargs.get('sequence', 'ideal_pulse')
        # self.sequence = PulseSequence(type=sequence)

        self.mode = kwargs.pop("mode", None)
        self.method = kwargs.pop("method", None)
        self.silent = kwargs.pop("silent", False)

        for exp_idx, dataset in enumerate(self.datasets):
            (
                dataset.modeldata,
                dataset.modelnames,
                dataset.model_A,
                dataset.model_a,
                dataset.model_b,
            ) = self._get_modeldata(
                dataset, exp_idx
            )  # lgtm[py/mismatched-multiple-assignment]

    # *******************************************************************************
    # public methodss
    # *******************************************************************************

    @staticmethod
    def script_default():
        """
        Return a default script.
        """
        return """
        #-----------------------------------------------------------
        # syntax for parameters definition:
        # name: value, low_bound,  high_bound
        #  * for fixed parameters
        #  $ for variable parameters
        #  > for reference to a parameter in the COMMON block
        #    (> is forbidden in the COMMON block)
        # common block parameters should not have a _ in their names
        #-----------------------------------------------------------
        #

        COMMON:
        # common parameters ex.
        experiment_variables: ampl

        # $ gwidth: 1.0, 0.0, none
          $ gratio: 0.5, 0.0, 1.0

        MODEL: LINE_1
        shape: voigtmodel
            $ ampl:  1.0, 0.0, none
            $ pos:   0.0, -100.0, 100.0
            > ratio: gratio
            $ width: 1.0, 0, 100
        """

    def dry_run(self):
        return self.run(dry=True)

    def run(self, maxiter=100, maxfun=None, every=10, method="simplex", **kwargs):
        """
        Main fitting procedure.

        Parameters
        ----------
        maxiter : int, maximum number of iteration
        maxfun : int, maximum number of function calls
        every : int, number of function call between two displays
        method : str, ether 'simplex' or 'hopping'
        dryrun : bool
        """

        if not self.silent:
            level = preferences.log_level
            if level > INFO:
                preferences.log_level = INFO
            info_("*" * 50)
            info_("  Entering fitting procedure")
            info_("*" * 50)

        global niter, chi2, everyiter, ncalls
        ncalls = 0
        everyiter = every
        niter = 0

        # internally defined function chi2
        def funchi2(params, datasets, *constraints):
            """
            Return sum((y - x)**2)
            """
            global chi2, ncalls
            # model spectrum

            chi2 = 0
            som = 0
            ncalls += 1

            for exp_idx, dataset in enumerate(datasets):
                modeldata = self._get_modeldata(dataset, exp_idx)[0]
                # baseline is already summed with modeldata[-1]

                # important to work with the real component of dataset
                # not the complex number
                data = dataset.real.data.squeeze()

                # if not dataset.is_2d:
                mdata = modeldata[-1]  # modelsum

                # else:
                #    mdata = modeldata.values

                merror = 1.0
                # if dataset.is_2d:
                #     if constraints:
                #
                #         # Case of SQ-DQ experiments
                #         if self.kind == 'SQ-DQ' and \
                #                         'max_connections' in constraints[0]:
                #             # check connectivity numbers
                #             nbconnections = {}
                #             for key in params.keys():
                #                 if 'pos1' in key:
                #                     connect = key[-2:]
                #                     key = 'ampl_line_' + connect  # get amplitude
                #                     ki = connect[0].upper()
                #                     if ki not in nbconnections.keys():
                #                         nbconnections[ki] = 0
                #                     if int(params[key]) > 0:
                #                         nbconnections[ki] += 1
                #             for k, v in nbconnections.iteritems():
                #                 if v > constraints[0]['max_connections']:
                #                     merror *= v * 10.

                diff = data - mdata
                chi2 += np.sum(diff ** 2) * merror
                som += np.sum(data[0] ** 2)

            chi2 = np.sqrt(chi2 / som)
            # reset log_level
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

            if not self.silent:
                display.clear_output(wait=True)
                info_(
                    ("Iterations: %d, Calls: %d (chi2: %.5f)" % (niter, ncalls, chi2))
                )
                sys.stdout.flush()

        # end callback function ---------------------------------------------------

        fp = self.fp  # starting parameters

        dry = kwargs.get("dry", False)

        if not dry:
            fp, fopt = optimize(
                funchi2,
                fp,
                args=(self.datasets,),
                maxfun=maxfun,
                maxiter=maxiter,
                method=method,
                constraints=kwargs.get("constraints", None),
                callback=callback,
            )

        # replace the previous script with new fp parameters
        self.parameterscript.script = str(fp)

        if not self.silent:
            # log.info the results
            info_("\n")
            info_("*" * 50)
            if not dry:
                info_("  Result:")
            else:
                info_("  Starting parameters:")
            info_("*" * 50)
            info_(self.parameterscript.script)

        # store the models
        for exp_idx, dataset in enumerate(self.datasets):
            (
                dataset.modeldata,
                dataset.modelnames,
                dataset.model_A,
                dataset.model_a,
                dataset.model_b,
            ) = self._get_modeldata(dataset, exp_idx)

        # Reset Log_level
        if not self.silent:
            preferences.log_level = level

        return

    # *******************************************************************************
    # properties
    # *******************************************************************************

    @property
    def fp(self):
        return self.parameterscript.fp

    @property
    def script(self):
        return self.parameterscript.script

    # *******************************************************************************
    # Private functions
    # *******************************************************************************

    def _repr_html_(self):
        if not self.datasets:
            return htmldoc(self.__init__.__doc__)
        else:
            return self.message

    def _get_modeldata(self, dataset, exp_idx):

        # Prepare parameters
        parameters = self._prepare(self.fp, exp_idx)

        # Get the list of models
        models = self.fp.models
        nbmodels = len(models)

        # Make an array 'modeldata' with the size of the dataset of data
        # which will contains the data produced by the models
        # This name must always be 'modeldata'
        # which will be returned to the main program.

        expedata = dataset.real.data.squeeze()
        axis, dim = dataset.get_axis(-1)
        x = dataset.coordset[dim].data

        if expedata.ndim > 1:
            # nD data
            raise NotImplementedError("Fit not implemented for nD data yet!")

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
            calc = getmodel(x, modelname=model, par=parameters)  # , dataset=dataset)
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

        xi = np.arange(float(x.size))
        A, a, b = self._ampbas(xi, expedata, modeldata[-1])
        # (fitzone-fitzone[0], data.take(fitzone),
        # modeldata[-1].take(fitzone))

        modeldata = A * modeldata
        baseline = a * xi + b  # a*(xi-fitzone[0]) + b

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

    def _prepare(self, param, exp_idx):
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

            if isinstance(new_param[key], list):
                new_param.data[key] = new_param.data[key][exp_idx]

        return new_param

    # ===============================================================================
    # automatic calculation of amplitude and baseline
    # ===============================================================================
    @staticmethod
    def _ampbas(xi, expe, calc):
        # Automatically calculate correct amplitude A and baseline
        # (baseline linear model a*i+b) by detemining the zero of the first derivative
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
        sId = sum(xi ** 2)

        den = (
            n * sFI ** 2
            - n * sFd * sId
            + sF ** 2 * sId
            - 2 * sF * sFI * sI
            + sFd * sI ** 2
        )

        a = (
            -sE * (sF * sFI - sFd * sI)
            + sEF * (n * sFI - sF * sI)
            - sEI * (n * sFd - sF ** 2)
        ) / den

        A = (
            sE * (sF * sId - sFI * sI)
            - sEF * (n * sId - sI ** 2)
            + sEI * (n * sFI - sF * sI)
        ) / den

        b = (
            sE * (sFI ** 2 - sFd * sId)
            + sEF * (sF * sId - sFI * sI)
            - sEI * (sF * sFI - sFd * sI)
        ) / den

        # in case the modeldata is zero, to avoid further errors
        if np.isnan(A):
            A = 0.0
        if np.isnan(a):
            a = 0.0
        if np.isnan(b):
            b = 0.0
        return A, a, b

    @staticmethod
    def _ampbas2D(xi, yj, expe, calc):
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
        sId = sum(xi ** 2)
        sJd = sum(yj ** 2)

        den = (
            -(m ** 2) * n ** 2 * sFd * sId * sJd
            + m ** 2 * n * sFJ ** 2 * sId
            + m ** 2 * n * sFd * sI ** 2 * sJd
            - m ** 2 * sFJ ** 2 * sI ** 2
            + m * n ** 2 * sFI ** 2 * sJd
            + m * n ** 2 * sFd * sId * sJ ** 2
            + m * n * sF ** 2 * sId * sJd
            - 2 * m * n * sF * sFI * sI * sJd
            - 2 * m * n * sF * sFJ * sId * sJ
            + 2 * m * n * sFI * sFJ * sI * sJ
            - 2 * m * n * sFI * sFJ * sIJ
            - 2 * m * n * sFd * sI * sIJ * sJ
            + m * n * sFd * sIJ ** 2
            + 2 * m * sF * sFJ * sI * sIJ
            - n ** 2 * sFI ** 2 * sJ ** 2
            + 2 * n * sF * sFI * sIJ * sJ
            - sF ** 2 * sIJ ** 2
        )

        c = (
            sE
            * (
                -m * n * sFd * sId * sJd
                + m * sFJ ** 2 * sId
                + n * sFI ** 2 * sJd
                - 2 * sFI * sFJ * sIJ
                + sFd * sIJ ** 2
            )
            + sEF
            * (
                m * n * sF * sId * sJd
                - m * n * sFI * sI * sJd
                - m * n * sFJ * sId * sJ
                + m * sFJ * sI * sIJ
                + n * sFI * sIJ * sJ
                - sF * sIJ ** 2
            )
            + sEI
            * (
                m * n * sFd * sI * sJd
                - m * sFJ ** 2 * sI
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
                - n * sFI ** 2 * sJ
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
                - n * sFI * sJ ** 2
                + sF * sIJ * sJ
            )
            + n
            * sEI
            * (
                -m * n * sFd * sJd
                + m * sFJ ** 2
                + n * sFd * sJ ** 2
                + sF ** 2 * sJd
                - 2 * sF * sFJ * sJ
            )
            + sE
            * (
                m * n * sFd * sI * sJd
                - m * sFJ ** 2 * sI
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
                + sF ** 2 * sIJ
            )
        ) / den

        A = (
            m
            * n
            * sEF
            * (
                -m * n * sId * sJd
                + m * sI ** 2 * sJd
                + n * sId * sJ ** 2
                - 2 * sI * sIJ * sJ
                + sIJ ** 2
            )
            + m
            * sEJ
            * (
                m * n * sFJ * sId
                - m * sFJ * sI ** 2
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
                - n * sFI * sJ ** 2
                + sF * sIJ * sJ
            )
            + sE
            * (
                m * n * sF * sId * sJd
                - m * n * sFI * sI * sJd
                - m * n * sFJ * sId * sJ
                + m * sFJ * sI * sIJ
                + n * sFI * sIJ * sJ
                - sF * sIJ ** 2
            )
        ) / den

        b = (
            m
            * sEF
            * (
                m * n * sFJ * sId
                - m * sFJ * sI ** 2
                - n * sF * sId * sJ
                + n * sFI * sI * sJ
                - n * sFI * sIJ
                + sF * sI * sIJ
            )
            + m
            * sEJ
            * (
                -m * n * sFd * sId
                + m * sFd * sI ** 2
                + n * sFI ** 2
                + sF ** 2 * sId
                - 2 * sF * sFI * sI
            )
            + sE
            * (
                m * n * sFd * sId * sJ
                - m * sF * sFJ * sId
                + m * sFI * sFJ * sI
                - m * sFd * sI * sIJ
                - n * sFI ** 2 * sJ
                + sF * sFI * sIJ
            )
            - sEI
            * (
                m * n * sFI * sFJ
                + m * n * sFd * sI * sJ
                - m * n * sFd * sIJ
                - m * sF * sFJ * sI
                - n * sF * sFI * sJ
                + sF ** 2 * sIJ
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


if __name__ == "__main__":
    pass
