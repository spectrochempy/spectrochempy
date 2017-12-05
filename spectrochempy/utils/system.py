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
"""
__all__ = ['get_user_and_node',
                      'get_user',
                      'get_node',
                      'is_kernel'
                      ]

import getpass
import platform
import inspect
import os
import sys


def get_user():
    return getpass.getuser()


def get_node():
    return platform.node()


def get_user_and_node():
    return "{0}@{1}".format(get_user(), get_node())


def is_kernel():
    """ Check if we are running from IPython

    """
    # from http://stackoverflow.com
    # /questions/34091701/determine-if-were-in-an-ipython-notebook-session
    if 'IPython' not in sys.modules:
        # IPython hasn't been imported, definitely not
        return False
    from IPython import get_ipython
    # check for `kernel` attribute on the IPython instance
    return getattr(get_ipython(), 'kernel', None) is not None
