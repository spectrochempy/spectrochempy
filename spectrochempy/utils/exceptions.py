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


__all__ =  ['SpectroChemPyWarning',
            'SpectroChemPyDeprecationWarning',
            'deprecated',
            ]



class SpectroChemPyWarning(Warning):
    """
    The base warning class from which all SpectroChemPy warnings should inherit.

    Any warning inheriting from this class is handled by the SpectroChemPy log.
    """


class SpectroChemPyDeprecationWarning(SpectroChemPyWarning):
    """
    A warning class to indicate a deprecated feature.
    """



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