# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Various objects
"""


class Adict(dict):
    """
    A simple dict with attribute-access
    """

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self.__setitem__(key, value)
