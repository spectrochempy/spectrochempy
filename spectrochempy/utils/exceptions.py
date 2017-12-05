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




"""
This module contains errors/exceptions and warnings of general use for
spectrochempy. Exceptions that are specific to a given subpackage should *not*
be here, but rather in the particular subpackage.
"""

__all__ =  ['SpectroChemPyWarning',
            'SpectroChemPyDeprecationWarning',
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
