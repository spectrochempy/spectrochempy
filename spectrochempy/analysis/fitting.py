# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
Module to perform fitting of 1D or n-D spectral data.
"""
__all__ = ["Fit", "optimize", "FitParameters", "ParameterScript"]

__dataset_methods__ = []

import sys
import re
from collections import UserDict
from warnings import warn

import numpy as np
import scipy.optimize
from traitlets import HasTraits, Bool, Any, Unicode, Instance, List, observe
from IPython import display

from spectrochempy.utils import htmldoc
from spectrochempy.core import preferences, info_, warning_, INFO
from spectrochempy.analysis import models as models_


# ======================================================================================================================
def getmodel(x, y=None, modelname=None, par=None, **kargs):
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
    ndarray : float
        An array containing the calculated model.
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
        return a.f(x, *args, **kargs)
    else:
        return a.f(x, y, *args, **kargs)


# =============================================================================
class FitParameters(UserDict):
    """
    Allow passing a dictionary of parameters with additional properties
    to the fit function. Check if the parameter is between the specified bounds
    if any.
    """

    # ------------------------------------------------------------------------
    def __init__(self):
        UserDict.__init__(self)  # Create a dictionary class
        self.lob = {}  # Lower bound
        self.upb = {}  # Upper bound
        self.fixed = {}  # true for non-variable parameter
        self.reference = {}  #
        self.common = {}  # indicates if a parameters belong to a common block
        self.model = {}  # model to use
        self.models = []  # list of models
        self.sequence = ""  # sequence used in the experiment
        self.expvars = []  # list of parameters which are experiment dependent
        self.expnumber = 1  # number of experiments

    # ------------------------------------------------------------------------
    def __setitem__(self, key, value):
        key = str(key)
        if key not in self.reference:
            self.reference[key] = False
        if self.reference[key]:
            # we get a reference to another parameter
            self.data[key] = str(value)
            self.fixed[key] = True
        elif isinstance(value, tuple) or isinstance(value, list):
            self.data[key] = self._evaluate(value[0])
            self.lob[key] = None
            self.upb[key] = None
            try:
                if len(value) > 2:
                    self.lob[key] = self._evaluate(value[1])
                    self.upb[key] = self._evaluate(value[2])
                    self._checkerror(key)
            except Exception:
                pass
            self.fixed[key] = False
            if isinstance(value[-1], bool):
                self.fixed[key] = value[-1]
        else:
            self.data[key] = self._evaluate(value)
            self.lob[key] = None
            self.upb[key] = None
            self.fixed[key] = False

    # ------------------------------------------------------------------------
    def __getitem__(self, key):
        key = str(key)
        if key in self.data:
            return self.data[key]
        raise KeyError(f"parameter `{key}` is not found")

    # ------------------------------------------------------------------------
    def iteritems(self):
        return iter(self.data.items())

    # ------------------------------------------------------------------------
    def _checkerror(self, key):
        key = str(key)
        if self.lob[key] is None and self.upb[key] is None:
            return False
        elif (self.lob[key] is not None and self.data[key] < self.lob[key]) or (
            self.upb[key] is not None and self.data[key] > self.upb[key]
        ):
            raise ValueError(f"`{key}` value ({self.data[key]}) is out of bounds")

    # ------------------------------------------------------------------------
    def __str__(self):

        # .............................................................
        def makestr(key):

            keystring = key.split("_")[0]
            if self.reference[key]:
                return f"\t> {keystring}:{self.data[key]}\n"
            else:
                if self.fixed[key]:
                    keystring = f"\t* {keystring}"
                else:
                    keystring = f"\t$ {keystring}"
                lob = self.lob[key]
                upb = self.upb[key]
                if lob <= -0.1 / sys.float_info.epsilon:
                    lob = "none"
                if upb >= +0.1 / sys.float_info.epsilon:
                    upb = "none"
                val = str(self.data[key])

                return f"{keystring}: {float(val):10.4f}, {lob}, {upb}\n"

        # ..............................................................

        message = "#PARAMETER SCRIPT\n\nCOMMON:\n"

        var = ""
        for item in self.expvars:
            var += f" {item}"

        if var:
            message += f"\texperiment_number: {self.expnumber}\n"
            message += f"\texperiment_variables: {var}\n"

        # look for common parameters
        for key in list(self.keys()):
            keysp = key.split("_")[0]
            if self.common[keysp]:
                message += makestr(key)

        # model parameters
        models = self.models
        for model in models:
            message += f"\nMODEL: {model}\n"
            message += f"shape: {self.model[model]}\n"
            for key in sorted(self.keys()):
                keyspl = key.split("_")
                if model not in "_".join(keyspl[1:]):
                    continue
                message += makestr(key)
        return message

    # ------------------------------------------------------------------------
    @staticmethod
    def _evaluate(strg):
        """
        Allow the evaluation of strings containing some operations

        Parameters
        ----------
        strg : string
            A string to evaluate containing multiplier,
            e.g., '10 k' evaluate to 10 000.

        Return
        ------
        value : float or bool
            Value of the string, or False, if there is an error
        """
        res = False

        if isinstance(strg, str):
            # strg=string.upper(strg)
            p = re.compile(r"\s+")
            m = p.split(strg.strip())

            for i in range(len(m)):
                try:
                    res = eval(m[i])
                except NameError:
                    message = f"Cannot evaluate '{strg}' >> {m[i]} is not defined"
                    raise NameError(message)
                except SyntaxError:
                    message = f"Syntax error in '{strg}'"
                    raise SyntaxError(message)
        else:
            # not a string (probably a scalar that can be return as it is)
            res = strg

        return res

    # ------------------------------------------------------------------------
    def to_internal(self, key, expi=None):
        """
        If expi is not none, several parameters to create.
        """
        key = str(key)
        if key not in self.data:
            raise KeyError(f"parameter `{key}` is not found")

        if expi is not None:
            pe = self.data[key][expi]
        else:
            pe = self.data[key]
        lob = self.lob[key]
        upb = self.upb[key]

        is_lob = (
            lob is not None and lob > -0.1 / sys.float_info.epsilon
        )  # lob is not None
        is_upb = (
            lob is not None and upb < +0.1 / sys.float_info.epsilon
        )  # upb is not None

        if is_lob and is_upb:
            lob = min(pe, lob)
            upb = max(pe, upb)
            # With min and max bounds defined
            pi = np.arcsin((2 * (pe - lob) / (upb - lob)) - 1.0)
        elif is_upb:
            upb = max(pe, upb)
            # With only max defined
            pi = np.sqrt((upb - pe + 1.0) ** 2 - 1.0)
        elif is_lob:
            lob = min(pe, lob)
            # With only min defined
            pi = np.sqrt((pe - lob + 1.0) ** 2 - 1.0)
        else:
            pi = pe
        return pi

    # ------------------------------------------------------------------------
    def to_external(self, key, pi):

        key = str(key)
        if key not in self.data:
            raise KeyError(f"parameter `{key}` is not found")

        lob = self.lob[key]
        upb = self.upb[key]

        is_lob = (
            lob is not None and lob > -0.1 / sys.float_info.epsilon
        )  # lob is not None
        is_upb = (
            lob is not None and upb < +0.1 / sys.float_info.epsilon
        )  # upb is not None

        if not isinstance(pi, list):
            pi = [
                pi,
            ]  # make a list

        pe = []
        for item in pi:
            if is_lob and is_upb:
                #  With min and max bounds defined
                pei = lob + ((upb - lob) / 2.0) * (np.sin(item) + 1.0)
            elif is_upb:
                # With only max defined
                pei = upb + 1.0 - np.sqrt(item ** 2 + 1.0)
            elif is_lob:
                # With only min defined
                pei = lob - 1.0 + np.sqrt(item ** 2 + 1.0)
            else:
                pei = pi
            pe.append(pei)

        if len(pe) == 1:
            pe = pe[0]

        self.data[key] = pe

        return pe

    def copy(self):

        import copy as cpy

        data = cpy.copy(self.data)
        lob = cpy.copy(self.lob)
        upb = cpy.copy(self.upb)
        fixed = cpy.copy(self.fixed)
        reference = cpy.copy(self.reference)

        c = cpy.copy(self)

        c.data = data
        c.lob = lob
        c.upb = upb
        c.fixed = fixed
        c.reference = reference

        return c


# =============================================================================
class ParameterScript(HasTraits):
    """
    This class allow some manipulation of the parameter list for modelling.
    """

    fp = Instance(FitParameters)

    script = Unicode("")

    datasets = List(Instance("spectrochempy.core.dataset.nddataset.NDDataset"))

    # =========================================================================
    # properties
    # =========================================================================
    # ------------------------------------------------------------------------
    @observe("script")
    def _check_parameters(self, change):
        """
        Check the validity of the parameters.
        """
        self.fp = self._interpret(self.script)

    # ------------------------------------------------------------------------
    def _interpret(self, script):
        """
        Interpreter of the script content.
        """
        # init some flags
        modlabel = None
        common = False
        fixed = False
        reference = False

        # create a new FitParameters instance
        fp = FitParameters()

        # set the number of experiments
        fp.expnumber = len(self.datasets)
        info_(f"The number of experiment(s) is set to {fp.expnumber}")

        # start interpreting ------------------------------------------------------
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
                if (
                    shape is None
                ):  # or (shape not in self._list_of_models and shape not in self._list_of_baselines):
                    raise ValueError(
                        f"Shape of this model `{shape}` was not specified or is not implemented"
                    )
                fp.model[modlabel] = shape
                common = False
                continue
            elif key.startswith("experiment"):  # must be in common
                if not common:
                    raise ValueError(
                        "'experiment_...' specification was found outside the common block."
                    )
                if "variables" in key:
                    expvars = values.lower().strip()
                    expvars = expvars.replace(",", " ").replace(";", " ")
                    expvars = expvars.split()
                    fp.expvars.extend(expvars)
                continue
            else:
                if modlabel is None and not common:
                    raise ValueError(
                        "The first definition should be a label for a model or a block of variables or constants."
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
                        f"Cannot interpret line {lc}: A parameter definition must start with *,$ or >"
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
                    if isinstance(val, list):
                        # if the parameter is already a list, that's ok if the number of parameters is ok
                        if len(val) != fp.expnumber:
                            raise ValueError(
                                f"the number of parameters {len(val)} is not the number of experiments."
                            )
                        if key not in fp.expvars:
                            raise ValueError(
                                f"parameter {key} is not declared as variable"
                            )
                    else:
                        if key in fp.expvars:
                            # we create a list of parameters corresponding
                            val = [val] * fp.expnumber
                    fp[ks] = val, mini.strip(), maxi.strip(), fixed
                else:
                    fp[ks] = value.strip()

        return fp


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

        # get parameters from script
        self.parameterscript = ParameterScript(datasets=self.datasets, script=script)
        if self.fp is None:  # pragma: no cover
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
    # public methods
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


# ======================================================================================================================
def optimize(
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
    """
    Parameters
    ----------
    func
    fp0
    args
    constraints
    method
    maxfun
    maxiter
    ftol
    xtol
    callback


    #  Internal/external transformation
    #  These transformations are used in the MINUIT package,
    #  and described in detail
    #  in the section 1.3.1 of the MINUIT User's Guide.


    """

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
