# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to provide a general
# API for displaying, processing and analysing spectrochemical data.
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
# =============================================================================



import warnings

from spectrochempy.utils.exceptions import SpectroChemPyDeprecationWarning

__all__ = ['deprecated']

def deprecated(message):
    """
    Deprecation decorator

    Parameters
    ----------
    message: str,
        the deprecation message

    """

    def deprecation_decorator(func):

        def wrapper(*args, **kwargs):
            warnings.warn("The function `{} is deprecated: {}".format(
                    func.__name__, message),
                    SpectroChemPyDeprecationWarning)
            return func(*args, **kwargs)

        return wrapper

    return deprecation_decorator
