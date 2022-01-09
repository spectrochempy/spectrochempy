# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
Model parameters handling.
"""

__all__ = ["FitParameters", "ParameterScript"]

# ==============
# python import
# ==============
import sys
import re  # For regular expression search
from collections import UserDict  # This is to be able to create a special dictionary
from spectrochempy.core import info_
from spectrochempy.core.dataset.nddataset import NDDataset
import numpy as np
from traitlets import HasTraits, Unicode, Instance, List, observe


# =============
# id_generator
# =============
# def _id_generator():
#     """Returns a sequence of numbers for the title of the objects.
#
#     Examples
#     --------
#
#     >>> id_generator
#     1
#
#     """
#     n = 1
#     while True:
#         yield n
#         n += 1
#
#
# id_generator = _id_generator()


# ==============
# FitParameters
# ==============
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
        raise KeyError("parameter %s is not found" % key)

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
            raise ValueError(
                "%s value %s is out of bounds" % (key, str(self.data[key]))
            )

    # ------------------------------------------------------------------------
    def __str__(self):

        # .............................................................
        def makestr(key):

            keystring = key.split("_")[0]
            if self.reference[key]:
                return "\t> %s:%s \n" % (keystring, self.data[key])
            else:
                if self.fixed[key]:
                    keystring = "\t* %s" % keystring
                else:
                    keystring = "\t$ %s" % keystring
                lob = self.lob[key]
                upb = self.upb[key]
                if lob <= -0.1 / sys.float_info.epsilon:
                    lob = "none"
                if upb >= +0.1 / sys.float_info.epsilon:
                    upb = "none"
                val = str(self.data[key])

                return "%s: %10.4f, %s, %s \n" % (keystring, float(val), lob, upb)

        # ..............................................................

        message = "#PARAMETER SCRIPT\n\nCOMMON: \n"

        var = ""
        for item in self.expvars:
            var += " %s" % item

        if var:
            message += "\texperiment_number: %s \n" % str(self.expnumber)
            message += "\texperiment_variables: %s \n" % var

        # look for common parameters
        for key in list(self.keys()):
            keysp = key.split("_")[0]
            if self.common[keysp]:
                message += makestr(key)

        # model parameters
        models = self.models
        for model in models:
            message += "\nMODEL: %s\n" % model
            message += "shape: %s\n" % self.model[model]
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
                    message = (
                        "Cannot evaluate '" + strg + "' >> " + m[i] + " is "
                        "not defined"
                    )
                    raise NameError(message)
                except SyntaxError:
                    message = "Syntax error in '" + strg + "'"
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
            raise KeyError("parameter %s is not found" % key)

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
            raise KeyError("parameter %s is not found" % key)

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


# ================
# ParameterScript
# ================
class ParameterScript(HasTraits):
    """
    This class allow some manipulation of the parameter list for modelling.
    """

    fp = Instance(FitParameters)

    script = Unicode("")

    datasets = List(Instance(NDDataset))

    # ===========================================================================
    # properties
    # ===========================================================================
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
        info_("The number of experiment(s) is set to %d" % fp.expnumber)

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
                    "Cannot interpret line %d: A semi-column is missing?" % lc
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
                        'Shape of this model "%s" was not specified or is not implemented'
                        % shape
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
                        "Cannot interpret line %d: A parameter definition must start with *,$ or >"
                        % lc
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
                        "line %d: value, min, max should be defined in this order" % lc
                    )
                elif len(s) == 2:
                    raise ValueError("only two items in line %d" % lc)
                    # s.append('none')
                elif len(s) == 1:
                    s.extend(["none", "none"])
                value, mini, maxi = s
                if mini.strip().lower() in ["none", ""]:
                    mini = str(-1.0 / sys.float_info.epsilon)
                if maxi.strip().lower() in ["none", ""]:
                    maxi = str(+1.0 / sys.float_info.epsilon)
                if modlabel != "common":
                    ks = "%s_%s" % (key, modlabel)
                    # print(ks)
                    # if "ratio_line_1" in ks:
                    #    print('xxxx'+ks)
                    fp.common[key] = False
                else:
                    ks = "%s" % key
                    fp.common[key] = True
                # if key in fp.expvars:
                #    for i in xrange(len(self.datasets)):
                #        ks = "%s_exp%d"%(ks, i)
                fp.reference[ks] = reference
                if not reference:
                    val = value.strip()
                    val = eval(val)
                    if isinstance(val, list):
                        # if the parameter is already a list, that's ok if the number of parameters is ok
                        if len(val) != fp.expnumber:
                            raise ValueError(
                                "the number of parameters for %s is not the number of experiments."
                                % len(val)
                            )
                        if key not in fp.expvars:
                            raise ValueError(
                                "parameter %s is not declared as variable" % key
                            )
                    else:
                        if key in fp.expvars:
                            # we create a list of parameters corresponding
                            val = [val] * fp.expnumber
                    fp[ks] = val, mini.strip(), maxi.strip(), fixed
                else:
                    fp[ks] = value.strip()

        return fp

        ###########################
