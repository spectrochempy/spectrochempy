# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================


__all__ = ['interleaved2complex']


def interleaved2complex(data):
    """
    Make a complex array from interleaved data

    """
    return data[..., ::2] + 1j * data[..., 1::2]


