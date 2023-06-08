# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

# Utility functions

import numpy as np


def lls(data):
    """
    Log-log-square operator.

    Compress the dynamic range of data by applying the log-log-square transform.
    It is used in the context of the baseline correction but can be used
    in other contexts, such as de-noising for instances.

    Parameters
    ----------
    data : array-like
        Data to be transformed.

    Returns
    -------
    array-like
        Log-log-square transformed data.

    See Also
    --------
    lls_inv : Inverse log-log-square operator.
    """

    return np.log(np.log(np.sqrt(data + 1) + 1) + 1)


def lls_inv(data):
    """
    Inverse log-log-square operator.

    Parameters
    ----------
    data : array-like
        Data to be transformed.

    Returns
    -------
    array-like
        Inverse log-log-square transformed data.

    See Also
    --------
    lls : Log-log-square operator.
    """

    return (np.exp(np.exp(data) - 1) - 1) ** 2 - 1
