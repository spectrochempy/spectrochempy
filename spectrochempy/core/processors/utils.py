# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory
# ======================================================================================================================

import functools


# ======================================================================================================================
# Decorators
# ======================================================================================================================


def _units_agnostic_method(method):
    @functools.wraps(method)
    def wrapper(dataset, **kwargs):

        # On which axis do we want to shift (get axis from arguments)
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

        data = method(new.data, **kwargs)
        new._data = data

        new.history = f"`{method.__name__}` shift performed on dimension `{dim}` with parameters: {kwargs}"

        # restore original data order if it was swaped
        if swaped:
            new.swapdims(axis, -1, inplace=True)  # must be done inplace

        return new

    return wrapper
