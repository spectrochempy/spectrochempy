# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================


# ===============================================================================
# imports
# ===============================================================================

import sys

import scipy.optimize
from warnings import warn
from spectrochempy.application import log

from spectrochempy.fitting.parameters import FitParameters

__all__ = ["optimize", ]
_methods = ['optimize', ]


# ===========================================================================
# _fitting
# ===========================================================================

#  Internal/external transformation
#  These transformations are used in the MINUIT package,
#  and described in detail
#  in the section 1.3.1 of the MINUIT User's Guide.

def optimize(func, fp0, args=(), constraints={}, method="SIMPLEX",
             maxfun=None, maxiter=1000,
             ftol=1e-8, xtol=1e-8,
             callback=None):
    """
    Optimization procedure

    :param func:
    :param x0:
    :param args:
    :param method:
    :param maxfun:
    :param maxiter:
    :param ftol:
    :param xtol:
    :param callback:
    :return:

    """

    global keys

    def restore_external(fp, p, keys):
        # restore external parameters
        for key in list(fp.keys()):
            keysp = key.split('_')
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
                if not key in keys:
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
        raise TypeError('fp0 is not of FitParameter type')

    # make internal parameters
    par = []
    keys = []

    for key in sorted(fp0.keys()):
        if not fp0.fixed[key]:
            # we make internal parameters in case of bounding
            # We also take care of the multiple experiments
            keysp = key.split('_')[0]
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
        result = scipy.optimize.fmin(internal_func, par,
                                     args=tuple(args),
                                     maxfun=maxfun, maxiter=maxiter,
                                     ftol=ftol, xtol=xtol,
                                     full_output=True, disp=False,
                                     callback=internal_callback)
        res, fopt, iterations, funcalls, warnmess = result

    elif method.upper() == "HOPPING":
        result = scipy.optimize.basinhopping(internal_func, par,
                                             niter=100, T=1.0, stepsize=0.5,
                                             minimizer_kwargs={
                                                 'args': tuple(args)},
                                             take_step=None,
                                             accept_test=None,
                                             callback=internal_callback,
                                             interval=50, disp=False,
                                             niter_success=None)

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
        warn("Maximum number of function evaluations made.")
    if warnmess == 2:
        warn("Maximum number of iterations reached.")

    return fpe, fopt


if __name__ == "__main__":
    pass
