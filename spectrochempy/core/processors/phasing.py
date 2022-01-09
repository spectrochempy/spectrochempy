# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================

"""
A collection of NMR spectral processing functions which operate on the last
dimension (1) of 2D arrays.
"""

__all__ = ["pk", "pk_exp"]
__dataset_methods__ = __all__

import functools
import numpy as np

from spectrochempy.units import ur, Quantity
from spectrochempy.core import error_

pi = np.pi


# ======================================================================================================================
# Decorators
# ======================================================================================================================


def _phase_method(method):
    @functools.wraps(method)
    def wrapper(dataset, **kwargs):

        # On which axis do we want to phase (get axis from arguments)
        axis, dim = dataset.get_axis(**kwargs, negative_axis=True)

        # output dataset inplace (by default) or not
        if not kwargs.pop("inplace", False):
            new = dataset.copy()  # copy to be sure not to modify this dataset
        else:
            new = dataset

        swaped = False
        if axis != -1:
            new.swapdims(axis, -1, inplace=True)  # must be done in  place
            swaped = True

        # Get the coordinates for the last dimension
        x = new.coordset[dim]

        # check if the dimensionality is compatible with this kind of functions
        if x.unitless or x.dimensionless or x.units.dimensionality != "[time]":

            # extract inital phase from metadata
            def _check_units(par, default_units, inv=False):
                if not isinstance(par, Quantity):
                    par *= Quantity(1.0, default_units)
                elif inv:
                    if par == 0:
                        return par
                    par = 1.0 / (1.0 / par).to(default_units)
                else:
                    par = par.to(default_units)
                return par

            # Set correct units for the parameters
            dunits = dataset.coordset[dim].units

            current = [new.meta.phc0[-1], new.meta.phc1[-1]]
            rel = kwargs.pop("rel", False)
            if rel:  # relative phase
                current = [0, 0]
            kwargs["phc0"] = (
                _check_units(kwargs.get("phc0", 0), "degree") - current[0]
            ).magnitude
            kwargs["phc1"] = (
                _check_units(kwargs.get("phc1", 0), "degree") - current[1]
            ).magnitude
            kwargs["pivot"] = _check_units(
                kwargs.get("pivot", new.meta.pivot[-1]), dunits
            ).magnitude
            kwargs["exptc"] = _check_units(
                kwargs.get("exptc", new.meta.get("exptc", [0] * new.ndim)[-1]),
                dunits,
                inv=True,
            ).magnitude

            if not new.meta.phased[-1]:
                # initial phase from topspin have not yet been used
                kwargs["phc0"] = -kwargs["phc0"]
                kwargs["phc1"] = -kwargs["phc1"]

            apod = method(new.data, **kwargs)
            new *= apod

            new.history = f"`{method.__name__}` applied to dimension `{dim}` with parameters: {kwargs}"

            if not new.meta.phased[-1]:
                new.meta.phased[-1] = True
                new.meta.phc0[-1] = 0 * ur.degree
                new.meta.phc1[-1] = 0 * ur.degree
                new.meta.exptc[-1] = 0 * (1 / dunits)
            else:
                if rel:
                    new.meta.phc0[-1] += kwargs["phc0"] * ur.degree
                    new.meta.phc1[-1] += kwargs["phc1"] * ur.degree
                else:
                    new.meta.phc0[-1] = kwargs["phc0"] * ur.degree
                    new.meta.phc1[-1] = kwargs["phc1"] * ur.degree

                    # TODO: to do for exptc too!
                new.meta.exptc[-1] = kwargs["exptc"] * (1 / dunits)

            new.meta.pivot[-1] = kwargs["pivot"] * dunits

        else:  # not (x.unitless or x.dimensionless or x.units.dimensionality != '[time]')
            error_(
                "This method apply only to dimensions with [frequency] or [dimensionless] dimensionality.\n"
                "Phase processing was thus cancelled"
            )

        # restore original data order if it was swaped
        if swaped:
            new.swapdims(axis, -1, inplace=True)  # must be done inplace

        return new

    return wrapper


# ======================================================================================================================
# Public methods
# ======================================================================================================================


@_phase_method
def pk(dataset, phc0=0.0, phc1=0.0, exptc=0.0, pivot=0.0, **kwargs):
    """
    Linear phase correction.

    For multidimensional NDDataset,
    the phase is by default applied on the last dimension.

    Parameters
    ----------
    dataset : nddataset
        Input dataset.
    phc0 : float or |Quantity|, optional, default=0 degree
        Zero order phase in degrees.
    phc1 : float or |Quantity|, optional, default=0 degree
        First order phase in degrees.
    exptc : float or |Quantity|, optional, default=0 us
        Exponential decay constant. If not 0, phc1 is ignored.
    pivot: float or |Quantity|, optional, default=0 in units of the x coordinate
        Units if any must be compatible with last dimension units.

    Returns
    -------
    phased
        Dataset.

    Other Parameters
    ----------------
    dim : str or int, keyword parameter, optional, default='x'.
        Specify on which dimension to apply the phase. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    inv : bool, keyword parameter, optional, default=False.
        True for inverse phasing.
    inplace : bool, keyword parameter, optional, default=False.
        True if we make the transform inplace.  If False, the function return a new dataset.

    See Also
    --------
    ps_exp : Exponential Phase Correction.
    pk : Automatic or manual phasing.
    """
    phc0 = pi * phc0 / 180.0
    size = dataset.shape[-1]

    if exptc > 0.0:
        apod = np.exp(1.0j * (phc0 * np.exp(-exptc * (np.arange(size) - pivot) / size)))

    else:
        phc1 = pi * phc1 / 180.0
        apod = np.exp(1.0j * (phc0 + (phc1 * (np.arange(size) - pivot) / size)))

    return apod


def pk_exp(dataset, phc0=0.0, pivot=0.0, exptc=0.0, **kwargs):
    """
    Exponential Phase Correction.

    For multidimensional NDDataset,
    the phase is by default applied on the last dimension.

    Parameters
    ----------
    dataset : nddataset
        Input dataset.
    phc0 : float or |Quantity|, optional, default=0 degree
        Zero order phase in degrees.
    exptc : float or |Quantity|, optional, default=0 us
        Exponential decay constant.

    Returns
    -------
    phased
        Dataset.

    Other Parameters
    ----------------
    dim : str or int, keyword parameter, optional, default='x'.
        Specify on which dimension to apply the phase. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    inv : bool, keyword parameter, optional, default=False.
        True for inverse phasing.
    inplace : bool, keyword parameter, optional, default=False.
        True if we make the transform inplace.  If False, the function return a new dataset.

    See Also
    --------
    ps : Linear Phase Correction.
    pk : Automatic or manual phasing.
    """

    return pk(dataset, phc0=phc0, phc1=0, pivot=pivot, exptc=exptc)


# # TODO: work on pk (below a copy from MASAI)
# @_phase_method
# def _apk(source=None, options='', axis=-1):
#     """
#     Automatic or manual phasing
#
#     Parameters
#     ----------
#     fit_phc1: bool, optional, default=False
#         also optimize first order phase if True
#     mode: String, optional, default='negmin'
#         method for automatic phase detection: 'negmin', 'entropy', ...
#     entropyd: int, optional, default=2
#         order of derivation for method entropy
#     gamma: float, optional
#         relative weigth for method entropy error with respect to negmin
#     axis: optional, default=-1
#
#     """
#     # options evaluation
#     parser = argparse.ArgumentParser(description='PK processing.', usage="""
#     pk [-h] [--auto] [--fit_phc1] [--ediff EDIFF]
#                            [--gamma GAMMA] [--select {standard,max,cols}]
#                            [--threshold THRESHOLD] [--mode MODE]
#                            [--optmode {simplex,hopping}]
#                            [--bound_phc0 BOUND_PHC0] [--bound_phc1 BOUND_PHC1]
#                            [--verbose]
#                            [phases [phases ...]]
#     """)
#
#     global data, par, interact
#
#     phc0 = None
#     phc1 = None
#
#     # positional arguments
#     parser.add_argument('phases', default=(phc0, phc1), nargs='*', type=float, help='zero and first order phase')
#     parser.add_argument('--pivot', '-pv', default=None, type=float, help='pivot position in spectral units')
#     parser.add_argument('--interactive', '-i', default=None, nargs='*',
#                         help='Interative mode on selected section here to check phase')
#     parser.add_argument('--pos', default=(0,), nargs='*', type=float,
#                         help='row or column position where to check phase')
#     parser.add_argument('--shifted', default=0.0, type=float, help="position of the top in units of time")
#     parser.add_argument('--exp', '-ex', action='store_true', help='perform an exponential phase correction')
#     parser.add_argument('--auto', '-a', action='store_true', help='set to automatic phase mode')
#     parser.add_argument('--fit_phc1', '-u1', action='store_true', help='use phc1 in automatic phasing', )
#     parser.add_argument('--ediff', '-ef', default=1, type=int,
#     help='order of the derivative for entropy calculation', )
#     parser.add_argument('--gamma', '-ga', default=1.0, type=float, help='weight', )
#     parser.add_argument('--select', '-st', default='standard', choices=['standard', 'max', 'cols', 'pos'],
#                         help='selection mode in automatic phasing', )
#     parser.add_argument('--threshold', '-th', default=50.0, type=int,
#     help='default threshold for columns selection', )
#     parser.add_argument('--mode', '-m', default='negmin+entropy', help="position of the top in units of time")
#     parser.add_argument('--optmode', '-om', default='simplex', choices=['simplex', 'hopping'],
#                         help="method of optimisation")
#     parser.add_argument('--bound_phc0', '-bp0', default=360., type=float, help="phc0 boundary")
#     parser.add_argument('--bound_phc1', '-bp1', default=10., type=float, help="phc1 boundary")
#     parser.add_argument('--byrow', '-br', action='store_true', help='to phase each row separately for series or 2D')
#     parser.add_argument('--verbose', action='store_true', help='verbose flag')
#     # parser.add_argument('--absolute', action='store_true', help='absolute flag: take the absolute value of phases')
#
#     args = parser.parse_args(options.split())
#
#     if not source:
#         return
#
#     if source:
#
#         if axis == -1 or axis == 1:
#             par = source.par
#         else:
#             par = source.par2
#
#         if not par.isfreq:
#             print('This source is not already transformed: are you sure this is what you want?')
#
#     # set data depending on the axis
#
#     data = source.data
#     if axis == 0:
#         # transpose temporarily the data for indirect dimension ft
#         data = data.T
#
#     # get initial phase and pivot
#     # get the initial phase setting
#     if DEBUG:
#         print('Current phases for axis %d : %f,%f' % (axis, par.PHC0, par.PHC1))
#
#     phc0, phc1 = args.phases
#
#     if phc0 is None and phc1 is None:
#         # phase were not given, read stored phase
#         args.phases = phc0, phc1 = par.PHC0, par.PHC1
#
#     # relative phases
#     phc0, phc1 = phc0 - par.PHC0, phc1 - par.PHC1
#
#     # absolute phases
#     par.PHC0, par.PHC1 = args.phases
#
#     # phases to apply
#     args.phases = phc0, phc1
#
#     # read pivot
#     ppivot = args.pivot
#     if ppivot is None:
#         ppivot = par.pivot
#     par.pivot = ppivot
#
#     # we need to transform into index
#     pivot = position2index(data, ppivot)
#     args.pivot = pivot
#
#     if DEBUG:
#         print('Phases demanded for axis %d : %f,%f with pivot: %f' % (axis, par.PHC0, par.PHC1, ppivot))
#         print('Actual phases for axis %d  to apply: %f,%f with pivot: %f' % (axis, phc0, phc1, ppivot))
#
#     sw = source.par.SW_h
#     p_shifted = args.shifted = 0.360 * args.shifted * sw  # time are in ms
#
#     # if the correction is exponential we need a second parameter tc=phc1
#     if args.exp:
#         args.fit_phc1 = True
#
#     # INTERACTIVE MODE ---------------------
#     if args.interactive is not None:
#
#         if DEBUG:
#             print('INTERACTIVE PHASING MODE')
#
#         interact = args.interactive
#         if interact == []:
#             interact = ['0']
#
#         # interactive phasing
#
#         ps0, ps1 = par.PHC0, par.PHC1
#
#         def phasing(ph0, ph1, pivot):
#
#             global interact
#
#             # set data depending on the axis
#             data = source.data.copy()
#             if axis == 0:
#                 # transpose temporarily the data for indirect dimension ft
#                 data = data.T
#
#             if axis == -1 or axis == 1:
#                 par = source.par
#             else:
#                 par = source.par2
#
#             p0 = (ph0 - ps0) * np.pi / 180.  # convert to radians
#             p1 = (ph1 - ps1) * np.pi / 180.
#
#             size = data.shape[-1]
#             pivot_ = position2index(data, pivot)
#             apod = np.exp(1.0j * (p0 + (p1 * (np.arange(size) - pivot_) / size)))
#             data = apod * data
#             fig = pl.figure(figsize=(4, 2))
#             ax = fig.add_subplot(1, 1, 1)
#             ax.set_xlim((data.columns.max(), data.columns.min()))
#
#             if interact[0] == 'max':
#                 i, j = np.unravel_index(np.abs(data.values).argmax(), data.values.shape)
#                 # dat = data.values[i]
#                 dat = getsection(data, i, width=1., axis=0)
#                 ax.plot(data.columns, dat.values)
#             else:
#                 for item in interact:
#                     pos = float(item)
#                     i = position2index(data.T, pos)  # index of a row
#                     dat = data.values[i]
#                     ax.plot(data.columns, dat)
#
#             ax.axvline(pivot, color='r', lw='.1')
#             ax.axhline(0, color='g', lw='.1')
#             pl.show()
#
#             print('ph0: {}  ph1: {}  pivot: {} (PHC0: {}'.format(ph0, ph1, pivot,
#                                                                  par.PHC0) + '\nWARNING THESE PHASE VALUES ARE NOT '
#                                                                            STORED IN THE SOURCE!' + '\nSo you must '
#                                                                                                        'use them in '
#                                                                                                        'another `pk` '
#                                                                                                        'command)')
#
#         w = interactive(phasing,
#                         ph0=FloatSlider(min=ps0 - 45, max=ps0 + 45, step=0.001, value=ps0, continuous_update=False),
#                         ph1=FloatSlider(min=ps1 - 180, max=ps1 + 180, step=0.01, value=ps1, continuous_update=False),
#                         pivot=FloatSlider(min=data.columns.min(), max=data.columns.max(), step=0.001, value=ppivot,
#                                           continuous_update=False))
#
#         output = w.children[-1]
#         display(w)
#
#         return w
#
#     # Manual mode ------------------------------
#
#     elif not args.auto:
#
#         if DEBUG:
#             print('MANUAL PHASE MODE')
#
#         # not interactive and not automatic leaves here...
#         data = ps(data.values, phc0, phc1 + p_shifted, pivot=pivot,
#                   is_exp=args.exp)  # this return an array not a dataframe
#         source.history.append('Manual phasing phc0:%.2f, phc1:%.2f, pivot:%.2f ' % (par.PHC0, par.PHC1, par.pivot))
#
#     # Automatic mode ------------------------------
#     else:
#         if DEBUG:
#             print('AUTOMATIC PHASE MODE')
#
#         # APK...
#         if args.select == 'cols' and source.is_2d and axis == 0:
#             ar = picking(source, args.threshold, index=True)
#             args.cols = zip(*ar)[1]
#
#         if args.byrow and source.is_2d:
#             rows = []
#             for index in range(data.index.size):
#                 row = data.iloc[index:index + 1].values
#                 row, phc0, phc1 = autophase(row, args)
#                 row = row - basecorr(row)
#                 rows.append(row)
#             # merge all rows to recreate data
#             data = np.vstack(rows)
#         else:
#             data, phc0, phc1 = autophase(data.values, args, par)  # return an array not a dataframe
#
#         atxt = '(not optimized)' if not args.fit_phc1 else ''
#
#         sbyrow = 'byrow' if args.byrow else ''
#         source.history.append(
#                 'Auto-phasing %s:  phc0 = %.3f, phc1%s = %.3f, pivot:%.2f' % (sbyrow, phc0, atxt, phc1, pivot))
#         # store the new phases
#         par.PHC0, par.PHC1 = phc0, phc1
#
#     try:
#         data = data - basecorr(data)
#     except Exception:
#         pass
#
#     # un-transpose the data if needed
#     if axis == 0:
#         data = data.T
#
#     source.data = data
#
#
# #
# # autophase
# #
# def _checkin(ph, bp):
#     if ph < -bp:
#         ph = -bp
#     if ph > bp:
#         ph = bp
#     return ph
#
#
# def _autophase(data, args, par):
#     """
#     Automatic phasing of 1D or 2D spectra
#     """
#
#     dat = data.copy()
#
#     # get parameters
#     # --------------
#     phc0, phc1 = args.phases
#     pivot = args.pivot
#     is_exp = args.exp
#     verbose = args.verbose
#
#     if DEBUG:
#         print('performing autophase')
#
#     select = args.select
#
#     if select == 'standard' or data.shape[0] == 1:
#         # select the first row
#         dat = dat[0]
#         select = 'standard'
#
#     if select == 'max' and data.ndim > 1:
#         i, j = np.unravel_index(data.real.argmax(), data.shape)
#         dat = data[i]
#
#     if select == 'cols' and data.ndim > 1:
#         l = []
#         cols = args.cols
#         if verbose:
#             print('columns selected:', cols)
#         for col in cols:
#             i = int(col)
#             l.append(data[i])
#         dat = np.vstack(l)
#
#     # apkmode
#     # --------
#     if DEBUG:
#         print('\nselect: ', select)
#
#     mode = args.mode
#     ediff = args.ediff
#     gamma = args.gamma
#     optmode = args.optmode
#     p_shifted = args.shifted
#     # prepare the phase to optimized with bound
#     fp = FitParameters()
#     bp0 = args.bound_phc0
#     bp1 = args.bound_phc1
#     phc0 = _checkin(phc0, bp0)
#     # fixed is false
#     fp['phc0'] = phc0, -bp0, bp0, False
#
#     fit_phc1 = args.fit_phc1
#     phc1 = _checkin(phc1, bp1)
#     fixed = not fit_phc1
#     fp['phc1'] = phc1, -bp1, bp1, fixed
#
#     # optim
#     global ni, nas, spe, err, niter
#     ni = 0
#     err = 0
#     niter = 0
#
#     # internal error function ----------------------------------------------------------------------------------
#     def _phase_error(p, s):
#
#         global nas, spe, ni, err
#
#         p0 = p['phc0']
#         p1 = p['phc1']
#         sc = s.copy()
#         sc = ps(sc, p0, p1 + p_shifted, pivot=pivot, is_exp=is_exp)
#
#         # baseline correction
#         scp = sc - basecorr(sc)
#
#         # Negative area minimization
#         nas = 0.
#         nam = 0.
#         if "negmin" in mode:
#             fm = scp[scp.real <= 0].real
#             fm = fm - np.min(fm)
#             nas = np.sum(fm ** 2)
#
#             # normalisation
#             fp = scp.real
#             fp = fp - np.min(fm)
#             nas = nas * 100. / (np.sum(fp ** 2) + 1.e-30)
#
#         # entropy minimization
#         spe = 0.
#         if 'entropy' in mode:
#             h = np.diff(scp.real, ediff)
#             h = np.abs(h)
#             h = abs(h / np.sum(np.abs(h))) + 1.e-30
#             spe = -np.sum(h * np.log(h))
#         ni += 1
#         err = nas + spe * gamma
#
#         return err
#
#     # end _phase_error function -------------------------------------------------------------------------------------
#
#     # callback function--------------------------------------------------------
#     def callback(x, f=None, accepted=None):
#         """
#         callback print function
#         """
#         global niter, err
#         return
#
#         if f is not None:  # hopping
#             pass
#         if f is None:  # simplex
#             # display.clear_output(wait=True)
#
#             msg = ("Iteration: %d (chi2: %.5f)" % (niter, err) +
#             ' Negative area (NA):%.3g Entropy (S) * gamma:%.3g' % (
#                     nas, spe * gamma))
#             auto_close_message(msg, title='Information', time=.001)
#
#             niter += 1
#
#     # end callback function ---------------------------------------------------
#
#     # convergence is not insured depending on the starting values
#     fp, err1 = optimize(_phase_error, fp, args=[dat, ], method=optmode, callback=None)
#     nas_save = nas
#     spe_save = spe
#     fp_save = fp.copy()
#
#     if optmode.upper() != 'HOPPING':
#         fp['phc0'] = (fp['phc0'] - 180.0) % 360.0, -bp0, bp0, False
#         ni = 0
#         fp, err2 = optimize(_phase_error, fp, args=(dat,), method=optmode, callback=callback)
#
#         # select the best
#         if err2 > err1:
#             fp = fp_save.copy()
#             nas = nas_save
#             spe = spe_save
#
#     # extract results
#     phc0 = fp['phc0']
#     phc1 = fp['phc1']
#
#     if verbose:
#         print('Phase:')
#
#     if fit_phc1:
#         if verbose:
#             print('phc0: %.3f' % phc0)
#             print('phc1: %.3f' % phc1)
#     else:
#         if verbose:
#             print('phc0: %.3f' % phc0)
#             print('phc1: %.3f (not optimized)' % phc1)
#
#     if verbose:
#         print('Negative area (NA):%.3g Entropy (S) * gamma:%.3g' % (nas, spe * gamma))
#
#     # apply to the original data and return
#     data = pk(data, phc0, phc1 + p_shifted, pivot=pivot, is_exp=is_exp)
#     return data, phc0, phc1
