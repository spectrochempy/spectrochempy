# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Module to perform fitting of 1D or n-D spectral data.
"""
__all__ = []


import re
import sys
from collections import UserDict

import numpy as np


# =============================================================================
class FitParameters(UserDict):
    """
    Allow passing a dictionary of parameters with additional properties
    to the fit function. Check if the parameter is between the specified bounds
    if any.
    """

    # ----------------------------------------------------------------------------------
    def __init__(self):
        UserDict.__init__(self)  # Create a dictionary class
        self.lob = {}  # Lower bound
        self.upb = {}  # Upper bound
        self.fixed = {}  # true for non-variable parameter
        self.reference = {}  #
        self.common = {}  # indicates if a parameters belong to a common block
        self.model = {}  # model to use
        self.models = []  # list of models
        self.sequence = ""  # NMR sequence used in the experiment
        self.expvars = []  # list of parameters which are experiment dependent
        self.expnumber = 1  # number of experiments

    # ----------------------------------------------------------------------------------
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

    # ----------------------------------------------------------------------------------
    def __getitem__(self, key):
        key = str(key)
        if key in self.data:
            return self.data[key]
        raise KeyError(f"parameter `{key}` is not found")

    # ----------------------------------------------------------------------------------
    def iteritems(self):
        return iter(self.data.items())

    # ----------------------------------------------------------------------------------
    def _checkerror(self, key):
        key = str(key)
        if self.lob[key] is None and self.upb[key] is None:
            return False
        elif (self.lob[key] is not None and self.data[key] < self.lob[key]) or (
            self.upb[key] is not None and self.data[key] > self.upb[key]
        ):
            raise ValueError(f"`{key}` value ({self.data[key]}) is out of bounds")

    # ----------------------------------------------------------------------------------
    def __str__(self):
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

    # ----------------------------------------------------------------------------------
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

    # ----------------------------------------------------------------------------------
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

    # ----------------------------------------------------------------------------------
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
                pei = upb + 1.0 - np.sqrt(item**2 + 1.0)
            elif is_lob:
                # With only min defined
                pei = lob - 1.0 + np.sqrt(item**2 + 1.0)
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
