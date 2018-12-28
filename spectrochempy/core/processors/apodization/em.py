# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

__all__ = ["em"]

__dataset_methods__ = __all__

# =============================================================================
# Third party imports
# =============================================================================
import numpy as np

# =============================================================================
# Local imports
# =============================================================================
from spectrochempy.core.processors.apodization.apodize import apodize
from spectrochempy.utils import epsilon

# =============================================================================
# em function
# =============================================================================

def em(dataset, *args, **kwargs):
    """Calculate an exponential apodization function

    The exponential apodization is calculated in the last dimension on NDDatasets.
    The data in the last dimension MUST be time-domain,
    or an error is raised.

    Functional form of apodization window:

    .. math::
        em(x) = \\exp(-pi * x * lb)

    Parameters
    ----------
    dataset : |NDDataset|.
        Dataset we want to apodize using exponential multiplication
    lb : float or `quantity`
        Exponential line broadening,
        If it is not a quantity with units,
        it is assumed to be a broadening expressed in Hz.
    shifted : float or `quantity`
        Shift the data time origin by this amount. If it is not a quantity
        it is assumed to be expressed in the data units of the last
        dimension.
    inv : bool, optional
        True for inverse apodization.  False (default) for standard.
    rev : bool, optional.
        True to reverse the apodization before applying it to the data.
    apply : `bool`, optional, default = True
        Should we apply the calculated apodization to the dataset (default)
        or just return the apodization ndarray.
    inplace : `bool`, optional, default = True
        Should we make the transform in place or return a new dataset
    axis : optional, default is -1

    Returns
    -------
    out : |NDDataset|.
        The apodized dataset if apply is True, the apodization array if not True.

    """
    args = list(args)  # important (args is a tuple)

    # what's the line broadening ?
    lb = kwargs.pop('lb', 0)
    if lb == 0:
        # let's try the args if the kwargs was not passed.
        # In this case it should be the first arg

        if len(args) > 0:
            lb = args.pop(0)

    # is it a shifted broadening?
    shifted = kwargs.pop('shifted', 0)

    def func(x, lb, tc2, shifted):
        # tc2 not used here
        if lb.magnitude <= epsilon:
            e = np.ones_like(x)
        else:
            e = np.pi * np.abs(x - shifted) / lb
        return np.exp(-e.data)

    kwargs['method'] = func
    kwargs['method_name'] = 'em'
    kwargs['apod'] = lb
    kwargs['shifted'] = shifted

    out = apodize(dataset, **kwargs)

    return out

# =============================================================================
if __name__ == '__main__':

    from spectrochempy import *

    dataset1D = NDDataset()
    path = os.path.join(datadir.path, 'nmrdata', 'bruker', 'tests', 'nmr', 'bruker_1d')
    dataset1D.read_bruker_nmr(path, expno=1, remove_digital_filter=True)

    dataset1D /= dataset1D.real.data.max() # normalize
    #TODO: workaround error with the max function which retur a dataset (issue #7)

    p = dataset1D.plot()
    apodfunc = dataset1D.em(lb=100. * ur.Hz, apply=False)

    apodfunc.plot(xlim=(0, 25000), zlim=(-2, 2), data_only=True, clear=False)

    show()